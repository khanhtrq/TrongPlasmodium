#!/usr/bin/env python3
"""
Quick test script for debugging GradCAM issues.
"""

import torch
import sys
import os
sys.path.append('src')

from src.gradcam import setup_gradcam, debug_target_layers, generate_and_save_gradcam_per_class
from src.model_initializer import initialize_model
from src.data_loader import get_data_loaders
from src.device_handler import get_device

def test_gradcam_quick():
    """Quick test with debug mode enabled."""
    
    print("🧪 Quick GradCAM Debug Test")
    print("=" * 50)
    
    # Setup device
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Load model (you'll need to adjust this path to your actual model)
    model_path = "path/to/your/trained/model.pth"  # CHANGE THIS!
    
    if not os.path.exists(model_path):
        print("❌ Please update model_path in test_gradcam.py")
        print("   Currently looking for:", model_path)
        return
    
    # Initialize model (adjust these parameters)
    model_name = "deit_tiny_patch16_224"  # CHANGE THIS to match your model!
    num_classes = 4  # CHANGE THIS to match your dataset!
    
    try:
        model = initialize_model(
            model_name=model_name,
            num_classes=num_classes,
            device=device
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create dummy input for testing
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust size if needed
    
    print("\n🔍 Running target layer debugging...")
    
    try:
        debug_results, best_layer = debug_target_layers(model, dummy_input, class_idx=0)
        
        print(f"\n📊 Debug Results Summary:")
        for name, result in debug_results.items():
            if result['success']:
                print(f"   ✅ {name}: range [{result['min']:.6f}, {result['max']:.6f}]")
            else:
                print(f"   ❌ {name}: {result['error']}")
        
        if best_layer:
            print(f"\n🏆 Best layer found: {type(best_layer).__name__}")
            
            # Test with the best layer
            print("\n🧪 Testing GradCAM with best layer...")
            _, gradcam_data, compute_func = setup_gradcam(
                model, 
                target_layers=[best_layer], 
                cam_algorithm='gradcam'
            )
            
            # Generate test CAM
            cam_result = compute_func(dummy_input, class_idx=0)
            
            if cam_result is not None:
                print("✅ GradCAM computation successful!")
                print(f"   CAM shape: {cam_result.shape}")
                print(f"   CAM range: [{cam_result.min():.6f}, {cam_result.max():.6f}]")
            else:
                print("❌ GradCAM computation failed")
        else:
            print("😞 No suitable target layer found")
            
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

def test_gradcam_with_dataset():
    """Test with actual dataset (requires data setup)."""
    
    print("\n🧪 GradCAM Test with Dataset")
    print("=" * 50)
    
    # You'll need to update these paths
    data_dir = "path/to/your/data"  # CHANGE THIS!
    model_path = "path/to/your/model.pth"  # CHANGE THIS!
    
    if not os.path.exists(data_dir):
        print("❌ Please update data_dir in test_gradcam.py")
        print("   Currently looking for:", data_dir)
        return
    
    if not os.path.exists(model_path):
        print("❌ Please update model_path in test_gradcam.py")
        print("   Currently looking for:", model_path)
        return
    
    # Load data
    try:
        device = get_device()
        _, _, test_loader = get_data_loaders(
            data_dir=data_dir,
            batch_size=32,
            num_workers=4
        )
        
        dataset = test_loader.dataset
        print(f"✅ Dataset loaded: {len(dataset.classes)} classes")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Load model (adjust parameters as needed)
    try:
        model = initialize_model(
            model_name="deit_tiny_patch16_224",  # CHANGE THIS!
            num_classes=len(dataset.classes),
            device=device
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Generate GradCAM with debug mode
    try:
        generate_and_save_gradcam_per_class(
            model=model,
            dataset=dataset,
            save_dir="gradcam_debug_results",
            device=device,
            cam_algorithm='gradcam',
            debug_layers=True  # Enable debug mode!
        )
        
        print("✅ GradCAM generation complete! Check 'gradcam_debug_results' folder.")
        
    except Exception as e:
        print(f"❌ GradCAM generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 GradCAM Debug Tool")
    print("Choose test mode:")
    print("1. Quick test with dummy data")
    print("2. Test with actual dataset")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_gradcam_quick()
    elif choice == "2":
        test_gradcam_with_dataset()
    else:
        print("Invalid choice. Running quick test...")
        test_gradcam_quick()
