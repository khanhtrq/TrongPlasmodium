import random
import warnings
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

# Try to import timm modules
try:
    import timm
    from timm.data import create_transform
    from timm.data.auto_augment import auto_augment_policy_v0, auto_augment_transform
    from timm.data.random_erasing import RandomErasing
    from timm.data.mixup import Mixup
    from timm.data.transforms import RandomResizedCropAndInterpolation
    from timm.data.transforms_factory import create_transform
    TIMM_AVAILABLE = True

except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn(
        "⚠️ timm library not available. Some advanced augmentations will not work.")


class RandomPad:
    def __init__(self, max_pad_ratio=0.1, fill=0, p=0.5):
        """
        Random padding transform using ratio-based padding.

        Args:
            max_pad_ratio (float): Maximum padding ratio relative to image dimensions (0.0-1.0)
            fill (int or tuple): Pixel fill value for padding
            p (float): Probability of applying padding (0.0-1.0)
        """
        self.max_pad_ratio = max_pad_ratio
        self.fill = fill
        self.p = p

    def __call__(self, img):
        # Apply padding based on probability
        if random.random() > self.p:
            return img

        if isinstance(img, Image.Image):
            width, height = img.size
        else:
            # Assume tensor format (C, H, W)
            height, width = img.shape[-2:]

        # Calculate max padding pixels based on image dimensions
        max_pad_x = int(width * self.max_pad_ratio)
        max_pad_y = int(height * self.max_pad_ratio)

        # Generate random padding for each side
        left = random.randint(0, max_pad_x)
        top = random.randint(0, max_pad_y)
        right = random.randint(0, max_pad_x)
        bottom = random.randint(0, max_pad_y)

        padding = [left, top, right, bottom]
        return F.pad(img, padding, fill=self.fill)


class BrightPixelStatistics:
    def __init__(self, threshold=(190, 190, 190), dark_threshold=(20, 20, 20)):
        """
        Initialize BrightPixelStatistics transform.

        Args:
            threshold (tuple): RGB threshold values for identifying bright pixels (0-255)
            dark_threshold (tuple): RGB threshold values for identifying dark pixels to replace (0-255)
        """
        if not isinstance(threshold, (tuple, list)) or len(threshold) != 3:
            raise ValueError(
                "Threshold must be a tuple or list of 3 integers (R, G, B).")
        if not isinstance(dark_threshold, (tuple, list)) or len(dark_threshold) != 3:
            raise ValueError(
                "Dark threshold must be a tuple or list of 3 integers (R, G, B).")

        self.threshold = threshold
        self.dark_threshold = dark_threshold

    def __call__(self, img):
        """
        Apply bright pixel statistics replacement to image before normalization.

        Args:
            img: PIL Image or numpy array (values in 0-255 range)

        Returns:
            PIL Image with dark pixels replaced
        """
        if isinstance(img, Image.Image):
            # Convert PIL to numpy array
            img_array = np.array(img)
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            raise TypeError("Input image must be a PIL Image or numpy array.")

        # Ensure we're working with uint8 values (0-255)
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        # Find bright pixels (all channels above threshold)
        bright_mask = np.all(img_array >= np.array(self.threshold), axis=2)
        bright_pixels = img_array[bright_mask]

        if len(bright_pixels) == 0:
            return img if isinstance(img, Image.Image) else Image.fromarray(img_array)

        # Calculate mean and std from bright pixels
        bright_mean = np.mean(bright_pixels, axis=0)
        bright_std = np.std(bright_pixels, axis=0)

        # Find dark pixels to replace (all channels below dark threshold)
        dark_mask = np.all(img_array <= np.array(self.dark_threshold), axis=2)

        if not np.any(dark_mask):
            return img if isinstance(img, Image.Image) else Image.fromarray(img_array)

        # Generate replacement values from bright pixel distribution
        num_dark_pixels = np.sum(dark_mask)
        replacement_values = np.random.normal(
            loc=bright_mean,
            scale=bright_std,
            size=(num_dark_pixels, 3)
        )

        # Clamp values to valid range [0, 255] and convert to uint8
        replacement_values = np.clip(
            replacement_values, 0, 255).astype(np.uint8)

        # Replace dark pixels
        modified_img = img_array.copy()
        modified_img[dark_mask] = replacement_values

        # Return PIL Image
        return Image.fromarray(modified_img)


class TimmAugmentationStrategy:
    """
    A comprehensive augmentation strategy using timm's advanced techniques.

    This class provides various augmentation policies that can be easily integrated
    into existing data loading pipelines.
    """

    def __init__(self,
                 strategy='light',
                 input_size=224,
                 interpolation='bicubic',
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 auto_augment_policy='original',
                 randaugment_magnitude=9,
                 randaugment_num_ops=2,
                 mixup_alpha=0.2,
                 cutmix_alpha=1.0,
                 random_erase_prob=0.25,
                 **kwargs):
        """
        Initialize augmentation strategy.

        Args:
            strategy (str): Augmentation intensity ['minimal', 'light', 'medium', 'heavy', 'extreme']
            input_size (int): Target image size
            interpolation (str): Interpolation method
            mean (tuple): Normalization mean
            std (tuple): Normalization std
            auto_augment_policy (str): AutoAugment policy name
            randaugment_magnitude (int): RandAugment magnitude (0-30)
            randaugment_num_ops (int): Number of RandAugment operations
            mixup_alpha (float): MixUp alpha parameter
            cutmix_alpha (float): CutMix alpha parameter
            random_erase_prob (float): Random erasing probability
        """
        self.strategy = strategy
        self.input_size = input_size if isinstance(
            input_size, (tuple, list)) else (input_size, input_size)
        self.interpolation = interpolation
        self.mean = mean
        self.std = std
        self.auto_augment_policy = auto_augment_policy
        self.randaugment_magnitude = randaugment_magnitude
        self.randaugment_num_ops = randaugment_num_ops
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.random_erase_prob = random_erase_prob

        # Validate timm availability for advanced features
        if not TIMM_AVAILABLE and strategy in ['heavy', 'extreme']:
            warnings.warn(
                "⚠️ Advanced augmentation strategies require timm. Falling back to 'medium' strategy.")
            self.strategy = 'medium'

    def get_train_transform(self):
        """Get training transform pipeline based on strategy."""
        if self.strategy == 'minimal':
            return self._get_minimal_transform()
        elif self.strategy == 'light':
            return self._get_light_transform()
        elif self.strategy == 'medium':
            return self._get_medium_transform()
        elif self.strategy == 'heavy':
            return self._get_heavy_transform()
        elif self.strategy == 'extreme':
            return self._get_extreme_transform()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_eval_transform(self):
        """Get evaluation/validation transform pipeline."""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def print_transform_details(self, transform_type='both'):
        """
        Print detailed information about the transforms being used.

        Args:
            transform_type (str): 'train', 'eval', or 'both'
        """
        print("🔍 Transform Analysis")
        print("=" * 50)

        if transform_type in ['eval', 'both']:
            print(
                f"\n📊 EVALUATION/TEST TRANSFORMS ({self.strategy.upper()} strategy)")
            print("-" * 40)
            eval_transform = self.get_eval_transform()
            self._print_transform_pipeline(eval_transform, "Evaluation")

        if transform_type in ['train', 'both']:
            print(
                f"\n🏋️ TRAINING TRANSFORMS ({self.strategy.upper()} strategy)")
            print("-" * 40)
            train_transform = self.get_train_transform()
            self._print_transform_pipeline(train_transform, "Training")

    def _print_transform_pipeline(self, transform_pipeline, pipeline_name):
        """Print details of a transform pipeline."""
        if hasattr(transform_pipeline, 'transforms'):
            transforms_list = transform_pipeline.transforms
        else:
            transforms_list = [transform_pipeline]

        print(f"\n{pipeline_name} Pipeline ({len(transforms_list)} steps):")
        for i, transform in enumerate(transforms_list, 1):
            print(f"  {i:2d}. {self._get_transform_description(transform)}")

        # Print summary statistics
        print(f"\n📋 {pipeline_name} Summary:")
        print(f"   • Total steps: {len(transforms_list)}")
        print(f"   • Input size: {self.input_size}")
        print(f"   • Interpolation: {self.interpolation}")
        print(f"   • Normalization: mean={self.mean}, std={self.std}")

    def _get_transform_description(self, transform):
        """Get a detailed description of a transform."""
        transform_name = transform.__class__.__name__

        # Handle different transform types
        if transform_name == 'Resize':
            size = getattr(transform, 'size', 'Unknown')
            interpolation = getattr(transform, 'interpolation', 'Unknown')
            return f"Resize(size={size}, interpolation={interpolation})"

        elif transform_name == 'RandomResizedCrop':
            size = getattr(transform, 'size', 'Unknown')
            scale = getattr(transform, 'scale', 'Unknown')
            ratio = getattr(transform, 'ratio', 'Unknown')
            interpolation = getattr(transform, 'interpolation', 'Unknown')
            return f"RandomResizedCrop(size={size}, scale={scale}, ratio={ratio}, interpolation={interpolation})"

        elif transform_name == 'RandomHorizontalFlip':
            p = getattr(transform, 'p', 'Unknown')
            return f"RandomHorizontalFlip(p={p})"

        elif transform_name == 'RandomVerticalFlip':
            p = getattr(transform, 'p', 'Unknown')
            return f"RandomVerticalFlip(p={p})"

        elif transform_name == 'RandomRotation':
            degrees = getattr(transform, 'degrees', 'Unknown')
            return f"RandomRotation(degrees={degrees})"

        elif transform_name == 'RandomAffine':
            degrees = getattr(transform, 'degrees', 'Unknown')
            translate = getattr(transform, 'translate', 'Unknown')
            scale = getattr(transform, 'scale', 'Unknown')
            shear = getattr(transform, 'shear', 'Unknown')
            return f"RandomAffine(degrees={degrees}, translate={translate}, scale={scale}, shear={shear})"

        elif transform_name == 'ColorJitter':
            brightness = getattr(transform, 'brightness', 'Unknown')
            contrast = getattr(transform, 'contrast', 'Unknown')
            saturation = getattr(transform, 'saturation', 'Unknown')
            hue = getattr(transform, 'hue', 'Unknown')
            return f"ColorJitter(brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue})"

        elif transform_name == 'Normalize':
            mean = getattr(transform, 'mean', 'Unknown')
            std = getattr(transform, 'std', 'Unknown')
            return f"Normalize(mean={mean}, std={std})"

        elif transform_name == 'RandomErasing':
            p = getattr(transform, 'p', 'Unknown')
            scale = getattr(transform, 'scale', 'Unknown')
            ratio = getattr(transform, 'ratio', 'Unknown')
            value = getattr(transform, 'value', 'Unknown')
            return f"RandomErasing(p={p}, scale={scale}, ratio={ratio}, value={value})"

        elif transform_name == 'ToTensor':
            return "ToTensor() - Convert PIL Image to Tensor [0,1]"

        elif hasattr(transform, '__class__'):
            # For timm transforms or custom transforms
            return f"{transform_name} - {str(transform)}"

        else:
            return f"{transform_name} - Unknown transform"

    def compare_strategies(self, strategies=None):
        """
        Compare transforms across different strategies.

        Args:
            strategies (list): List of strategies to compare. If None, compare all.
        """
        if strategies is None:
            strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']

        print("🔍 Strategy Comparison")
        print("=" * 60)

        for strategy in strategies:
            print(f"\n🎯 {strategy.upper()} STRATEGY:")
            print("-" * 30)

            try:
                # Create temporary strategy instance
                temp_strategy = TimmAugmentationStrategy(
                    strategy=strategy,
                    input_size=self.input_size,
                    mean=self.mean,
                    std=self.std
                )

                # Get transforms
                train_transform = temp_strategy.get_train_transform()
                eval_transform = temp_strategy.get_eval_transform()

                # Count transforms
                train_count = len(train_transform.transforms) if hasattr(
                    train_transform, 'transforms') else 1
                eval_count = len(eval_transform.transforms) if hasattr(
                    eval_transform, 'transforms') else 1

                print(f"   Training transforms: {train_count} steps")
                print(f"   Evaluation transforms: {eval_count} steps")

                # List key transforms
                if hasattr(train_transform, 'transforms'):
                    key_transforms = [
                        t.__class__.__name__ for t in train_transform.transforms]
                    print(
                        f"   Key training ops: {', '.join(key_transforms[:5])}")
                    if len(key_transforms) > 5:
                        print(
                            f"                     ... and {len(key_transforms) - 5} more")

            except Exception as e:
                print(f"   ❌ Error with {strategy}: {e}")

        print(f"\n📊 Evaluation transforms are the same for all strategies:")
        eval_transform = self.get_eval_transform()
        if hasattr(eval_transform, 'transforms'):
            eval_ops = [
                t.__class__.__name__ for t in eval_transform.transforms]
            print(f"   {' → '.join(eval_ops)}")

    def _get_minimal_transform(self):
        """Minimal augmentation for conservative training."""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _get_light_transform(self):
        """Light augmentation with basic geometric transforms."""
        return transforms.Compose([
            # Thêm padding ngẫu nhiên
            RandomPad(max_pad_ratio=0.1, fill=0, p=0.5),
            BrightPixelStatistics(),
            transforms.RandomAffine(
                degrees=0,  # Độ xoay sẽ được xử lý bởi RandomRotation
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5,
                fill=0,
                interpolation=self._get_interpolation()
            ),
            BrightPixelStatistics(),
            transforms.RandomRotation(
                degrees=45, fill=0, interpolation=self._get_interpolation()),  # Thêm xoay ngẫu nhiên
            BrightPixelStatistics(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAdjustSharpness(
                sharpness_factor=5.0,
                p=0.2,

            ),
            transforms.GaussianBlur(
                kernel_size=(5, 5),
                sigma=(0.1, 2.0),
            ),
            transforms.RandomResizedCrop(
                self.input_size,
                scale=(0.7, 1),
                ratio=(0.8, 1.2),
                interpolation=self._get_interpolation(),
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0,
                hue=0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.05),
                ratio=(0.3, 3.3),
                value=0
            ),

        ])

    def _get_medium_transform(self):
        """Medium augmentation with more aggressive transforms."""
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.input_size,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                interpolation=self._get_interpolation()
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomPerspective(
                distortion_scale=0.1,
                p=0.2,
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=5.0,
                p=0.2
            ),
            transforms.GaussianBlur(
                kernel_size=(5, 5),
                sigma=(0.1, 2.0),
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                value=0
            ),

        ])

    def _get_heavy_transform(self):
        """Heavy augmentation using timm's advanced techniques."""
        if not TIMM_AVAILABLE:
            warnings.warn(
                "⚠️ timm not available, falling back to medium augmentation")
            return self._get_medium_transform()

        # Create base transforms
        base_transforms = [
            transforms.RandomResizedCrop(
                self.input_size,
                scale=(0.6, 1.0),
                ratio=(0.75, 1.33),
                interpolation=self._get_interpolation()
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # Add AutoAugment if available
        try:
            auto_augment = auto_augment_transform(
                self.auto_augment_policy,
                hparams={'translate_const': int(
                    0.45 * min(self.input_size)), 'img_mean': tuple([min(255, round(255 * x)) for x in self.mean])}
            )
            base_transforms.append(auto_augment)
        except Exception as e:
            warnings.warn(f"⚠️ Could not create AutoAugment: {e}")
            # Fallback to manual augmentation (geometric only)
            base_transforms.extend([
                transforms.RandomRotation(degrees=20),
            ])

        # Add final transforms
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # Add RandomErasing
        try:
            random_erase = RandomErasing(
                probability=self.random_erase_prob,
                mode='pixel',
                max_count=1,
                num_splits=0,
                cube=False
            )
            base_transforms.append(random_erase)
        except Exception:
            # Fallback to torchvision's RandomErasing
            base_transforms.append(
                transforms.RandomErasing(
                    p=self.random_erase_prob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            )

        return transforms.Compose(base_transforms)

    def _get_extreme_transform(self):
        """Extreme augmentation for very challenging scenarios."""
        if not TIMM_AVAILABLE:
            warnings.warn(
                "⚠️ timm not available, falling back to medium augmentation")
            return self._get_medium_transform()

        try:            # Use timm's create_transform with aggressive settings
            transform = create_transform(
                input_size=self.input_size,
                is_training=True,
                use_prefetcher=False,
                no_aug=False,
                scale=(0.5, 1.0),
                ratio=(0.7, 1.4),
                hflip=0.5,
                vflip=0.2,
                color_jitter=0.0,  # Disable color jitter
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation=self.interpolation,
                mean=self.mean,
                std=self.std,
                re_prob=self.random_erase_prob,
                re_mode='pixel',
                re_count=1,
                re_num_splits=0,
                separate=False,
                color_jitter_prob=0,
                grayscale_prob=0,
            )
            return transform
        except Exception as e:
            warnings.warn(
                f"⚠️ Could not create extreme transform: {e}, falling back to heavy")
            return self._get_heavy_transform()

    def _get_interpolation(self):
        """Get PIL interpolation mode."""
        interpolation_map = {
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'nearest': Image.NEAREST,
            'lanczos': Image.LANCZOS,
        }
        return interpolation_map.get(self.interpolation, Image.BICUBIC)


class MixupCutmixWrapper:
    """
    Wrapper for MixUp and CutMix augmentations that can be applied during training.

    This should be used in the training loop, not in the transform pipeline.
    """

    def __init__(self,
                 mixup_alpha=0.2,
                 cutmix_alpha=1.0,
                 cutmix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 label_smoothing=0.1,
                 num_classes=1000):
        """
        Initialize MixUp/CutMix wrapper.

        Args:
            mixup_alpha (float): MixUp alpha parameter
            cutmix_alpha (float): CutMix alpha parameter  
            cutmix_minmax (tuple): CutMix min/max ratio
            prob (float): Overall probability of applying augmentation
            switch_prob (float): Probability of switching between MixUp and CutMix
            mode (str): 'batch' or 'pair'
            label_smoothing (float): Label smoothing factor
            num_classes (int): Number of classes
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        if TIMM_AVAILABLE:
            try:
                self.mixup_fn = Mixup(
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                    cutmix_minmax=cutmix_minmax,
                    prob=prob,
                    switch_prob=switch_prob,
                    mode=mode,
                    label_smoothing=label_smoothing,
                    num_classes=num_classes
                )
                self.enabled = True
            except Exception as e:
                warnings.warn(f"⚠️ Could not initialize timm Mixup: {e}")
                self.enabled = False
        else:
            self.enabled = False
            self.mixup_fn = None

    def __call__(self, x, target):
        """Apply MixUp/CutMix augmentation."""
        if self.enabled and self.mixup_fn is not None:
            return self.mixup_fn(x, target)
        return x, target

    def is_enabled(self):
        """Check if MixUp/CutMix is enabled."""
        return self.enabled


class RandAugmentTransform:
    """
    RandAugment implementation for use in transform pipelines.
    """

    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31):
        """
        Initialize RandAugment transform.

        Args:
            num_ops (int): Number of operations to apply
            magnitude (int): Magnitude of operations (0-30)
            num_magnitude_bins (int): Number of magnitude bins
        """
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        # Define available operations (geometric and shape-only)
        self.operations = [
            # self._auto_contrast,
            # self._equalize,
            # self._invert,  # Removed - color inversion
            self._rotate,
            self._posterize,
            # self._solarize,  # Removed - color manipulation
            # self._solarize_add,  # Removed - color manipulation
            # self._color,  # Removed - saturation adjustment
            # self._contrast,  # Removed - contrast adjustment
            # self._brightness,  # Removed - brightness adjustment
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img):
        """Apply RandAugment to an image."""
        ops = random.choices(self.operations, k=self.num_ops)
        for op in ops:
            img = op(img)
        return img

    def _get_magnitude(self):
        """Get magnitude value normalized to [0, 1]."""
        return self.magnitude / self.num_magnitude_bins

    def _auto_contrast(self, img):
        """Apply auto contrast."""
        return ImageOps.autocontrast(img)

    def _equalize(self, img):
        """Apply histogram equalization."""
        return ImageOps.equalize(img)

    def _invert(self, img):
        """Invert image colors."""
        return ImageOps.invert(img)

    def _rotate(self, img):
        """Apply rotation."""
        magnitude = self._get_magnitude()
        angle = magnitude * 30  # Max 30 degrees
        if random.random() > 0.5:
            angle = -angle
        return img.rotate(angle, fillcolor=(128, 128, 128))

    def _posterize(self, img):
        """Apply posterization."""
        magnitude = self._get_magnitude()
        bits = int(8 - magnitude * 4)  # 4-8 bits
        return ImageOps.posterize(img, bits)

    def _solarize(self, img):
        """Apply solarization."""
        magnitude = self._get_magnitude()
        threshold = int(256 - magnitude * 256)
        return ImageOps.solarize(img, threshold)

    def _solarize_add(self, img):
        """Apply solarization with addition."""
        magnitude = self._get_magnitude()
        threshold = int(magnitude * 110)
        img_array = np.array(img)
        img_array = np.where(img_array < threshold,
                             img_array + threshold, img_array)
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _color(self, img):
        """Adjust color saturation."""
        magnitude = self._get_magnitude()
        factor = 1 + magnitude * 0.9  # 1.0 to 1.9
        if random.random() > 0.5:
            factor = 1 - magnitude * 0.9  # 0.1 to 1.0
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _contrast(self, img):
        """Adjust contrast."""
        magnitude = self._get_magnitude()
        factor = 1 + magnitude * 0.9
        if random.random() > 0.5:
            factor = 1 - magnitude * 0.9
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _brightness(self, img):
        """Adjust brightness."""
        magnitude = self._get_magnitude()
        factor = 1 + magnitude * 0.9
        if random.random() > 0.5:
            factor = 1 - magnitude * 0.9
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _sharpness(self, img):
        """Adjust sharpness."""
        magnitude = self._get_magnitude()
        factor = 1 + magnitude * 0.9
        if random.random() > 0.5:
            factor = 1
        if random.random() > 0.5:
            factor = 1 - magnitude * 0.9
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

    def _shear_x(self, img):
        """Apply horizontal shear."""
        magnitude = self._get_magnitude()
        shear = magnitude * 0.3
        if random.random() > 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(128, 128, 128))

    def _shear_y(self, img):
        """Apply vertical shear."""
        magnitude = self._get_magnitude()
        shear = magnitude * 0.3
        if random.random() > 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), fillcolor=(128, 128, 128))

    def _translate_x(self, img):
        """Apply horizontal translation."""
        magnitude = self._get_magnitude()
        translate = magnitude * img.size[0] * 0.45
        if random.random() > 0.5:
            translate = -translate
        return img.transform(img.size, Image.AFFINE, (1, 0, translate, 0, 1, 0), fillcolor=(128, 128, 128))

    def _translate_y(self, img):
        """Apply vertical translation."""
        magnitude = self._get_magnitude()
        translate = magnitude * img.size[1] * 0.45
        if random.random() > 0.5:
            translate = -translate
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, translate), fillcolor=(128, 128, 128))


def create_augmentation_strategy(config, model_config=None):
    """
    Factory function to create augmentation strategy from config.

    Args:
        config (dict): Configuration dictionary with augmentation settings
        model_config (dict): Model configuration with input size, mean, std

    Returns:
        TimmAugmentationStrategy: Configured augmentation strategy
    """
    # Extract augmentation config
    aug_config = config.get('augmentation', {})

    # Get model-specific parameters
    if model_config:
        input_size = model_config.get('input_size', 224)
        mean = model_config.get('mean', (0.485, 0.456, 0.406))
        std = model_config.get('std', (0.229, 0.224, 0.225))
        interpolation = model_config.get('interpolation', 'bilinear')
    else:
        input_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = 'bilinear'

    # Create strategy
    strategy = TimmAugmentationStrategy(
        strategy=aug_config.get('strategy', 'light'),
        input_size=input_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        auto_augment_policy=aug_config.get('auto_augment_policy', 'original'),
        randaugment_magnitude=aug_config.get('randaugment_magnitude', 9),
        randaugment_num_ops=aug_config.get('randaugment_num_ops', 2),
        mixup_alpha=aug_config.get('mixup_alpha', 0.2),
        cutmix_alpha=aug_config.get('cutmix_alpha', 1.0),
        random_erase_prob=aug_config.get('random_erase_prob', 0.25),
    )

    return strategy


def get_timm_transform(model_name, input_size=224, is_training=True, **kwargs):
    """
    Get timm's recommended transform for a specific model.

    Args:
        model_name (str): Name of the timm model
        input_size (int): Input image size
        is_training (bool): Whether for training or evaluation
        **kwargs: Additional arguments for create_transform

    Returns:
        Transform pipeline or None if timm not available
    """
    if not TIMM_AVAILABLE:
        return None

    try:
        # Get model's data config
        model = timm.create_model(model_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(model)

        # Create transform
        transform = create_transform(
            input_size=input_size,
            is_training=is_training,
            **{**data_config, **kwargs}
        )

        return transform
    except Exception as e:
        warnings.warn(
            f"⚠️ Could not create timm transform for {model_name}: {e}")
        return None


# Example usage and testing functions
def test_augmentations():
    """Test different augmentation strategies."""
    print("🧪 Testing augmentation strategies...")

    # Create test strategies
    strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']

    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        try:
            aug_strategy = TimmAugmentationStrategy(strategy=strategy)
            train_transform = aug_strategy.get_train_transform()
            eval_transform = aug_strategy.get_eval_transform()

            print(f"✅ {strategy} strategy created successfully")
            print(
                f"   Train transform steps: {len(train_transform.transforms)}")
            print(f"   Eval transform steps: {len(eval_transform.transforms)}")

        except Exception as e:
            print(f"❌ Error with {strategy} strategy: {e}")

    # Test MixUp/CutMix
    print(f"\n--- Testing MixUp/CutMix ---")
    try:
        mixup_cutmix = MixupCutmixWrapper(num_classes=6)
        print(f"✅ MixUp/CutMix enabled: {mixup_cutmix.is_enabled()}")
    except Exception as e:
        print(f"❌ Error with MixUp/CutMix: {e}")


def test_image_augmentation(image_path, output_dir="augmentation_test_results"):
    """
    Test augmentation on a single image and display results.

    Args:
        image_path (str): Path to the test image
        output_dir (str): Directory to save augmented images
    """
    import matplotlib.pyplot as plt
    import os
    from PIL import Image

    print(f"🧪 Testing image augmentation on: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the original image
    try:
        original_img = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded image: {original_img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Test different augmentation strategies
    strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']

    # Create figure for comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f'Augmentation Test Results - {os.path.basename(image_path)}', fontsize=16)

    # Show original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Test each strategy
    for idx, strategy in enumerate(strategies):
        row = (idx + 1) // 3
        col = (idx + 1) % 3

        try:
            print(f"\n--- Testing {strategy} strategy ---")

            # Create augmentation strategy
            aug_strategy = TimmAugmentationStrategy(
                strategy=strategy,
                input_size=224,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

            # Get training transform
            train_transform = aug_strategy.get_train_transform()

            # Apply augmentation
            augmented_tensor = train_transform(original_img.copy())

            # Convert tensor back to PIL Image for display
            if isinstance(augmented_tensor, torch.Tensor):
                # Denormalize the tensor
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denormalized = augmented_tensor * std + mean
                denormalized = torch.clamp(denormalized, 0, 1)

                # Convert to numpy and PIL
                img_array = denormalized.permute(1, 2, 0).numpy()
                augmented_img = Image.fromarray(
                    (img_array * 255).astype(np.uint8))
            else:
                augmented_img = augmented_tensor

            # Display in subplot
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f'{strategy.capitalize()} Strategy')
            axes[row, col].axis('off')

            # Save augmented image
            save_path = os.path.join(output_dir, f'{strategy}_augmented.jpg')
            augmented_img.save(save_path)
            print(f"✅ Saved {strategy} augmented image to: {save_path}")

        except Exception as e:
            print(f"❌ Error with {strategy} strategy: {e}")
            # Show error placeholder
            axes[row, col].text(0.5, 0.5, f'Error:\n{strategy}',
                                ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(
                f'{strategy.capitalize()} Strategy (Error)')
            axes[row, col].axis('off')

    # Adjust layout and save comparison
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'augmentation_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison image to: {comparison_path}")

    # Show the plot
    plt.show()

    print(f"\n🎉 Augmentation test completed! Results saved in: {output_dir}")


def test_multiple_augmentations(image_path, strategy='medium', num_samples=9, output_dir="multiple_augmentations"):
    """
    Apply the same augmentation strategy multiple times to see variation.

    Args:
        image_path (str): Path to the test image
        strategy (str): Augmentation strategy to use
        num_samples (int): Number of augmented samples to generate
        output_dir (str): Directory to save results
    """
    import matplotlib.pyplot as plt
    import os
    from PIL import Image

    print(f"🔄 Testing multiple {strategy} augmentations on: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the original image
    try:
        original_img = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded image: {original_img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Create augmentation strategy
    try:
        aug_strategy = TimmAugmentationStrategy(
            strategy=strategy,
            input_size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        train_transform = aug_strategy.get_train_transform()
    except Exception as e:
        print(f"❌ Error creating augmentation strategy: {e}")
        return

    # Create figure
    rows = int(np.ceil(np.sqrt(num_samples + 1)))
    cols = int(np.ceil((num_samples + 1) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(
        f'Multiple {strategy.capitalize()} Augmentations - {os.path.basename(image_path)}', fontsize=16)

    # Flatten axes for easier indexing
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Show original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # hardcode transform for testing - FIXED VERSION
    # train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Move BEFORE ToTensor
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ])

    # Generate multiple augmented versions
    for i in range(num_samples):
        try:
            # Apply augmentation
            augmented_tensor = train_transform(original_img.copy())

            # Convert tensor back to PIL Image for display
            if isinstance(augmented_tensor, torch.Tensor):
                # Denormalize the tensor
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denormalized = augmented_tensor * std + mean
                denormalized = torch.clamp(denormalized, 0, 1)

                # Convert to numpy and PIL
                img_array = denormalized.permute(1, 2, 0).numpy()
                augmented_img = Image.fromarray(
                    (img_array * 255).astype(np.uint8))
            else:
                augmented_img = augmented_tensor

            # Display in subplot
            axes[i + 1].imshow(augmented_img)
            axes[i + 1].set_title(f'Sample {i + 1}')
            axes[i + 1].axis('off')

            # Save augmented image
            save_path = os.path.join(
                output_dir, f'{strategy}_sample_{i+1:02d}.jpg')
            augmented_img.save(save_path)

        except Exception as e:
            print(f"❌ Error generating sample {i + 1}: {e}")
            axes[i + 1].text(0.5, 0.5, f'Error\nSample {i+1}',
                             ha='center', va='center', transform=axes[i + 1].transAxes)
            axes[i + 1].axis('off')

    # Hide unused subplots
    for i in range(num_samples + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    comparison_path = os.path.join(
        output_dir, f'multiple_{strategy}_augmentations.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved multiple augmentations to: {comparison_path}")

    # Show the plot
    plt.show()

    print(f"🎉 Multiple augmentation test completed!")


def interactive_augmentation_test():
    """
    Interactive function to test augmentations with user input.
    """
    import os

    print("🎮 Interactive Augmentation Test")
    print("=" * 50)

    default_image_path = r"X:\datn\v2_malaria_full_class_classification\v2_malaria_full_class_classification\train\064\rbc_parasitized_F_S1\cell2.jpg"
    # Get image path from user
    while True:
        image_path = input(
            "\n📁 Enter the path to your test image (or default): ").strip()
        if not image_path:
            image_path = default_image_path
            print(f"Using default image path: {image_path}")
            break
        elif os.path.exists(image_path):
            break
        else:
            print("❌ File not found. Please enter a valid image path.")

    # Choose test type
    print("\n🔧 Choose test type:")
    print("1. Compare all strategies")
    print("2. Multiple samples of one strategy")
    print("3. Both tests")

    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

    # Execute tests based on choice
    if choice in ['1', '3']:
        print("\n🚀 Running strategy comparison test...")
        test_image_augmentation(image_path)

    if choice in ['2', '3']:
        print("\n🔧 Choose strategy for multiple samples:")
        strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")

        while True:
            strategy_choice = input("Enter strategy number (1-5): ").strip()
            if strategy_choice in ['1', '2', '3', '4', '5']:
                selected_strategy = strategies[int(strategy_choice) - 1]
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")

        while True:
            try:
                num_samples = int(
                    input("Enter number of samples (default 9): ").strip() or "9")
                if num_samples > 0:
                    break
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a valid number.")

        print(f"\n🚀 Running multiple {selected_strategy} augmentation test...")
        test_multiple_augmentations(image_path, selected_strategy, num_samples)

    print("\n🎉 All tests completed!")


def test_single_image_augmentation(image_path):
    """
    Simple test function that takes an image path, applies different augmentation strategies,
    and displays the results in a grid.

    Args:
        image_path (str): Path to the test image
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    print(f"🧪 Testing augmentation strategies on: {image_path}")

    # Load the original image
    try:
        original_img = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded image: {original_img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Test different augmentation strategies
    strategies = ['minimal', 'light', 'medium', 'heavy']

    # Create figure for comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f'Augmentation Test Results - {os.path.basename(image_path)}', fontsize=16)

    # Show original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Test each strategy
    for idx, strategy in enumerate(strategies):
        row = (idx + 1) // 3
        col = (idx + 1) % 3

        try:
            print(f"--- Testing {strategy} strategy ---")

            # Create augmentation strategy
            aug_strategy = TimmAugmentationStrategy(
                strategy=strategy,
                input_size=224,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

            # Get training transform
            train_transform = aug_strategy.get_train_transform()

            # Apply augmentation
            augmented_tensor = train_transform(original_img.copy())

            # Convert tensor back to PIL Image for display
            if isinstance(augmented_tensor, torch.Tensor):
                # Denormalize the tensor
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denormalized = augmented_tensor * std + mean
                denormalized = torch.clamp(denormalized, 0, 1)

                # Convert to numpy and PIL
                img_array = denormalized.permute(1, 2, 0).numpy()
                augmented_img = Image.fromarray(
                    (img_array * 255).astype(np.uint8))
            else:
                augmented_img = augmented_tensor

            # Display in subplot
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f'{strategy.capitalize()} Strategy')
            axes[row, col].axis('off')

            print(f"✅ {strategy} strategy applied successfully")

        except Exception as e:
            print(f"❌ Error with {strategy} strategy: {e}")
            # Show error placeholder
            axes[row, col].text(0.5, 0.5, f'Error:\n{strategy}',
                                ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(
                f'{strategy.capitalize()} Strategy (Error)')
            axes[row, col].axis('off')

    # Hide the last subplot since we only have 4 strategies + 1 original
    axes[1, 2].axis('off')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    print(f"🎉 Augmentation test completed!")


def test_mixup_cutmix_wrapper(image_path1, image_path2, num_classes=6, output_dir="mixup_cutmix_test"):
    """
    Test MixupCutmixWrapper với 2 ảnh đầu vào và tạo ra 3 ảnh kết quả.

    Args:
        image_path1 (str): Đường dẫn đến ảnh thứ nhất
        image_path2 (str): Đường dẫn đến ảnh thứ hai  
        num_classes (int): Số lượng classes cho wrapper
        output_dir (str): Thư mục lưu kết quả

    Returns:
        tuple: (mixup_result, cutmix_result, comparison_image)
    """
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    import torch.nn.functional as F

    print(f"🧪 Testing MixUp/CutMix with:")
    print(f"   Image 1: {image_path1}")
    print(f"   Image 2: {image_path2}")

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Load và preprocess 2 ảnh
    try:
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        print(f"✅ Loaded images: {img1.size}, {img2.size}")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return None, None, None

    # Tạo transform cho preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # Preprocess images
    tensor1 = preprocess(img1).unsqueeze(0)  # Add batch dimension
    tensor2 = preprocess(img2).unsqueeze(0)

    # Tạo batch từ 2 ảnh
    # Shape: [2, 3, 224, 224]
    batch_images = torch.cat([tensor1, tensor2], dim=0)

    # Tạo fake labels
    batch_labels = torch.tensor([0, 1])  # 2 classes khác nhau

    print(f"📊 Batch shape: {batch_images.shape}")
    print(f"📊 Labels shape: {batch_labels.shape}")

    # Test MixUp riêng biệt
    print(f"\n🔄 Testing MixUp...")
    try:
        mixup_wrapper = MixupCutmixWrapper(
            mixup_alpha=0.2,
            cutmix_alpha=0.0,  # Tắt CutMix
            prob=1.0,
            switch_prob=0.0,  # Chỉ dùng MixUp
            num_classes=num_classes
        )

        if mixup_wrapper.is_enabled():
            mixed_images_mixup, mixed_labels_mixup = mixup_wrapper(
                batch_images.clone(), batch_labels.clone())
            print(f"✅ MixUp applied successfully")
            mixup_result = mixed_images_mixup[0]  # Lấy ảnh đầu tiên
        else:
            print(f"⚠️ MixUp not available, using original")
            mixup_result = batch_images[0]

    except Exception as e:
        print(f"❌ Error with MixUp: {e}")
        mixup_result = batch_images[0]

    # Test CutMix riêng biệt
    print(f"\n✂️ Testing CutMix...")
    try:
        cutmix_wrapper = MixupCutmixWrapper(
            mixup_alpha=0.0,  # Tắt MixUp
            cutmix_alpha=0.5,
            prob=1.0,
            switch_prob=1.0,  # Chỉ dùng CutMix
            num_classes=num_classes
        )

        if cutmix_wrapper.is_enabled():
            mixed_images_cutmix, mixed_labels_cutmix = cutmix_wrapper(
                batch_images.clone(), batch_labels.clone())
            print(f"✅ CutMix applied successfully")
            cutmix_result = mixed_images_cutmix[0]  # Lấy ảnh đầu tiên
        else:
            print(f"⚠️ CutMix not available, using original")
            cutmix_result = batch_images[0]

    except Exception as e:
        print(f"❌ Error with CutMix: {e}")
        cutmix_result = batch_images[0]

    # Hàm để denormalize tensor thành PIL Image
    def tensor_to_pil(tensor):
        """Convert normalized tensor to PIL Image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)

        # Convert to PIL
        img_array = denormalized.permute(1, 2, 0).numpy()
        return Image.fromarray((img_array * 255).astype(np.uint8))

    # Convert results to PIL Images
    img1_pil = tensor_to_pil(batch_images[0])
    img2_pil = tensor_to_pil(batch_images[1])
    mixup_pil = tensor_to_pil(mixup_result)
    cutmix_pil = tensor_to_pil(cutmix_result)

    # Tạo ảnh comparison
    print(f"\n🖼️ Creating comparison visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MixUp/CutMix Test Results', fontsize=16)

    # Hàng đầu: Input images và MixUp result
    axes[0, 0].imshow(img1_pil)
    axes[0, 0].set_title('Input Image 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_pil)
    axes[0, 1].set_title('Input Image 2')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mixup_pil)
    axes[0, 2].set_title('MixUp Result\n(Blended)')
    axes[0, 2].axis('off')

    # Hàng thứ hai: CutMix result và combined view
    axes[1, 0].imshow(cutmix_pil)
    axes[1, 0].set_title('CutMix Result\n(Patch Mixed)')
    axes[1, 0].axis('off')

    # So sánh side-by-side MixUp vs CutMix
    comparison_img = Image.new('RGB', (mixup_pil.width * 2, mixup_pil.height))
    comparison_img.paste(mixup_pil, (0, 0))
    comparison_img.paste(cutmix_pil, (mixup_pil.width, 0))

    axes[1, 1].imshow(comparison_img)
    axes[1, 1].set_title('MixUp vs CutMix\n(Side by Side)')
    axes[1, 1].axis('off')

    # Thống kê
    axes[1, 2].text(0.1, 0.9, 'Augmentation Stats:', fontsize=12,
                    weight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(
        0.1, 0.8, f'✅ MixUp: {"Enabled" if mixup_wrapper.is_enabled() else "Disabled"}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(
        0.1, 0.7, f'✅ CutMix: {"Enabled" if cutmix_wrapper.is_enabled() else "Disabled"}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(
        0.1, 0.6, f'📊 Input size: {batch_images.shape}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(
        0.1, 0.5, f'🎯 Classes: {num_classes}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'🔧 MixUp α: 0.8',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'✂️ CutMix α: 1.0',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    # Lưu ảnh
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'mixup_cutmix_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison to: {comparison_path}")

    # Lưu từng ảnh riêng lẻ
    mixup_path = os.path.join(output_dir, 'mixup_result.jpg')
    cutmix_path = os.path.join(output_dir, 'cutmix_result.jpg')
    comparison_only_path = os.path.join(
        output_dir, 'side_by_side_comparison.jpg')

    mixup_pil.save(mixup_path)
    cutmix_pil.save(cutmix_path)
    comparison_img.save(comparison_only_path)

    print(f"✅ Saved individual results:")
    print(f"   MixUp: {mixup_path}")
    print(f"   CutMix: {cutmix_path}")
    print(f"   Comparison: {comparison_only_path}")

    # Hiển thị plot
    plt.show()

    print(f"\n🎉 MixUp/CutMix test completed!")
    print(f"📁 All results saved in: {output_dir}")

    return mixup_pil, cutmix_pil, comparison_img


def interactive_mixup_cutmix_test():
    """
    Hàm interactive để test MixUp/CutMix với input từ user.
    """
    import os

    print("🎮 Interactive MixUp/CutMix Test")
    print("=" * 50)

    # Default paths
    default_img1 = r"X:\datn\v2_malaria_full_class_classification\v2_malaria_full_class_classification\train\064\rbc_parasitized_F_S1\cell2.jpg"
    default_img2 = r"X:\datn\v2_malaria_full_class_classification\v2_malaria_full_class_classification\test\084\rbc_parasitized_F_TJ\cell2.jpg"

    # Get first image path
    print(f"\n📁 Enter path for first image:")
    img1_path = input(f"(default: {os.path.basename(default_img1)}): ").strip()
    if not img1_path:
        img1_path = default_img1
    elif not os.path.exists(img1_path):
        print(f"❌ File not found, using default")
        img1_path = default_img1

    # Get second image path
    print(f"\n📁 Enter path for second image:")
    img2_path = input(f"(default: {os.path.basename(default_img2)}): ").strip()
    if not img2_path:
        img2_path = default_img2
    elif not os.path.exists(img2_path):
        print(f"❌ File not found, using default")
        img2_path = default_img2

    # Get number of classes
    try:
        num_classes = int(
            input("\n🎯 Enter number of classes (default 6): ").strip() or "6")
    except ValueError:
        num_classes = 6
        print("Using default: 6 classes")

    # Run test
    print(f"\n🚀 Running MixUp/CutMix test...")
    mixup_result, cutmix_result, comparison = test_mixup_cutmix_wrapper(
        img1_path, img2_path, num_classes=num_classes
    )

    if mixup_result is not None:
        print(f"✅ Test completed successfully!")
    else:
        print(f"❌ Test failed!")


if __name__ == "__main__":
    import sys

    print("🧪 Augmentation Testing Module")
    print("=" * 40)

    # Check if image path provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"📸 Testing with provided image: {image_path}")
            test_image_augmentation(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
            print("🎮 Starting interactive mode...")
            interactive_augmentation_test()
    else:
        # Run interactive test or basic function tests
        print("Choose an option:")
        print("1. Test augmentation functions")
        print("2. Interactive image augmentation test")
        print("3. Test MixUp/CutMix wrapper")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            test_augmentations()
        elif choice == "2":
            interactive_augmentation_test()
        elif choice == "3":
            interactive_mixup_cutmix_test()
        else:
            print("Running basic augmentation tests...")
            test_augmentations()


def print_test_transforms(strategy='medium', input_size=224):
    """
    Print out the transforms used during testing/evaluation.

    Args:
        strategy (str): Augmentation strategy to analyze
        input_size (int): Input image size
    """
    print("🧪 Test/Evaluation Transform Analysis")
    print("=" * 50)

    # Create augmentation strategy
    aug_strategy = TimmAugmentationStrategy(
        strategy=strategy,
        input_size=input_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    # Print only evaluation transforms
    aug_strategy.print_transform_details(transform_type='eval')

    return aug_strategy


def compare_all_transforms():
    """
    Compare transforms across all strategies for both training and testing.
    """
    print("🔍 Complete Transform Comparison")
    print("=" * 60)

    strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"🎯 {strategy.upper()} STRATEGY ANALYSIS")
        print(f"{'='*60}")

        try:
            aug_strategy = TimmAugmentationStrategy(
                strategy=strategy,
                input_size=224,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

            # Print detailed analysis for this strategy
            aug_strategy.print_transform_details(transform_type='both')

        except Exception as e:
            print(f"❌ Error analyzing {strategy} strategy: {e}")


def test_transforms_only():
    """
    Test and display only the evaluation/test transforms.
    """
    print("🔬 Test Transform Analysis")
    print("=" * 50)

    print("📊 EVALUATION/TEST TRANSFORMS")
    print("-" * 30)
    print("Note: Test transforms are the same for all strategies")
    print()

    # Create a strategy instance
    aug_strategy = TimmAugmentationStrategy(strategy='medium')

    # Get and analyze evaluation transforms
    eval_transform = aug_strategy.get_eval_transform()

    print("Test Transform Pipeline:")
    if hasattr(eval_transform, 'transforms'):
        for i, transform in enumerate(eval_transform.transforms, 1):
            description = aug_strategy._get_transform_description(transform)
            print(f"  {i}. {description}")

    print(f"\n📋 Test Transform Summary:")
    print(f"   • Total steps: {len(eval_transform.transforms)}")
    print(f"   • Input size: {aug_strategy.input_size}")
    print(f"   • Interpolation: {aug_strategy.interpolation}")
    print(
        f"   • Normalization: mean={aug_strategy.mean}, std={aug_strategy.std}")

    print(f"\n✨ Purpose of each transform:")
    print(f"   1. Resize: Standardizes input image to model's expected size")
    print(f"   2. ToTensor: Converts PIL Image to PyTorch tensor (0-1 range)")
    print(f"   3. Normalize: Normalizes pixel values using ImageNet statistics")


def analyze_transform_impact():
    """
    Analyze the impact and purpose of test transforms.
    """
    print("🔍 Test Transform Impact Analysis")
    print("=" * 50)

    aug_strategy = TimmAugmentationStrategy(strategy='medium')
    eval_transform = aug_strategy.get_eval_transform()

    print("📊 EVALUATION/TEST TRANSFORM BREAKDOWN:")
    print("-" * 40)

    if hasattr(eval_transform, 'transforms'):
        for i, transform in enumerate(eval_transform.transforms, 1):
            transform_name = transform.__class__.__name__
            print(f"\n{i}. {transform_name}")

            if transform_name == 'Resize':
                size = getattr(transform, 'size', 'Unknown')
                print(f"   📐 Purpose: Resize image to {size}")
                print(f"   🎯 Impact: Ensures consistent input dimensions for model")
                print(f"   ⚙️  Method: Maintains aspect ratio, centers image")

            elif transform_name == 'ToTensor':
                print(f"   📐 Purpose: Convert PIL Image to PyTorch tensor")
                print(
                    f"   🎯 Impact: Changes data type from uint8 [0,255] to float32 [0,1]")
                print(
                    f"   ⚙️  Method: Divides pixel values by 255, changes HWC to CHW format")

            elif transform_name == 'Normalize':
                mean = getattr(transform, 'mean', 'Unknown')
                std = getattr(transform, 'std', 'Unknown')
                print(f"   📐 Purpose: Normalize pixel values using ImageNet statistics")
                print(
                    f"   🎯 Impact: Centers data around 0, reduces training instability")
                print(f"   ⚙️  Method: (pixel - mean) / std for each channel")
                print(f"   📊 Values: mean={mean}, std={std}")

    print(f"\n🎯 Overall Test Pipeline Impact:")
    print(f"   • Ensures model receives consistently formatted input")
    print(f"   • No data augmentation during testing (maintains original image content)")
    print(f"   • Preserves image quality while meeting model requirements")
    print(f"   • Uses same normalization as training for consistent feature space")
