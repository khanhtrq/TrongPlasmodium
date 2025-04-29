import copy
import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import warnings
import torch.nn.utils as torch_utils  # For gradient clipping

try:
    import torch_xla.core.xla_model as xm
    _tpu_available = True
except ImportError:
    _tpu_available = False

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=25, patience=5, use_amp=True, save_path='best_model.pth',
                log_path='training_log.csv', clip_grad_norm=1.0):
    """Trains the model, tracks history, handles early stopping, and saves the best weights."""
    since = time.time()

    # --- Input Validation ---
    if not isinstance(clip_grad_norm, (float, int)) or clip_grad_norm <= 0:
        print(f"⚠️ Invalid clip_grad_norm value ({clip_grad_norm}). Disabling gradient clipping.")
        clip_grad_norm = None  # Disable clipping if value is invalid

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_metric = 0.0  # Use a generic name, decided by primary metric below
    primary_metric = 'val_acc_macro'  # Metric to monitor for improvement and early stopping
    epochs_no_improve = 0
    nan_inf_counter = 0
    max_nan_inf_tolerance = 5  # Number of NaN/Inf batches tolerated before warning/action

    is_cuda = device.type == 'cuda'
    is_tpu = _tpu_available and 'xla' in str(device)

    # Enable GradScaler only if using CUDA and AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_cuda))

    history = {
        'epoch': [],
        'train_loss': [], 'train_acc_macro': [], 'train_acc_weighted': [],
        'val_loss': [], 'val_acc_macro': [], 'val_acc_weighted': [],
        'val_precision_macro': [], 'val_recall_macro': [], 'val_f1_macro': [],
        'val_precision_weighted': [], 'val_recall_weighted': [], 'val_f1_weighted': [],
        'lr': []
    }

    print(f"\n🚀 Starting Training Configuration:")
    print(f"   Model: {type(model).__name__}")
    print(f"   Epochs: {num_epochs}, Patience: {patience}")
    print(f"   Device: {device}, AMP: {use_amp}, Grad Clip Norm: {clip_grad_norm}")
    print(f"   Optimizer: {type(optimizer).__name__}, LR Scheduler: {type(scheduler).__name__}")
    print(f"   Criterion: {type(criterion).__name__}")
    print(f"   Best Model Path: {save_path}")
    print(f"   Log Path: {log_path}")
    print(f"   Primary Metric for Improvement: {primary_metric}")
    print("-" * 30)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        history['epoch'].append(epoch + 1)
        history['lr'].append(optimizer.param_groups[0]['lr'])  # Log LR at start of epoch

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                # --- Debug: Check validation dataloader before loop ---
                try:
                    print(f"Debug VAL: len(dataloaders['val']) = {len(dataloaders['val'])}")
                    print(f"Debug VAL: len(dataloaders['val'].dataset) = {len(dataloaders['val'].dataset)}")
                except Exception as e:
                    print(f"Debug VAL: Error getting dataloader/dataset length - {e}")
                # --- End Debug ---

            running_loss = 0.0
            all_preds = []
            all_labels = []
            batch_count = 0
            phase_start_time = time.time()

            # Use tqdm for progress bar
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}', leave=False, unit="batch")

            for i, (inputs, labels) in enumerate(pbar):  # Use enumerate for batch index
                # --- Debug: Check if validation loop is entered ---
                if phase == 'val' and i == 0:
                    print(f"Debug VAL: Entered validation loop (batch 0). inputs.shape={inputs.shape if hasattr(inputs, 'shape') else 'N/A'}, labels.shape={labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                # --- End Debug ---

                # Skip batch if data loading failed (indicated by empty tensors)
                if inputs.numel() == 0 or labels.numel() == 0:
                    warnings.warn(f"Skipping empty batch in {phase} phase (epoch {epoch+1}). Check data loading.")
                    continue

                inputs = inputs.to(device, non_blocking=True)  # Use non_blocking for potential speedup
                labels = labels.to(device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Use autocast for mixed precision if enabled
                    # Determine dtype based on device
                    amp_dtype = torch.float16 if is_cuda else torch.bfloat16
                    with torch.cuda.amp.autocast(enabled=(use_amp and is_cuda), dtype=amp_dtype):
                        outputs = model(inputs)
                        # Ensure outputs and labels are compatible with criterion
                        loss = criterion(outputs, labels)

                    # Check for NaN/Inf loss *before* backward pass
                    if not torch.isfinite(loss).item():
                        nan_inf_counter += 1
                        warnings.warn(f"⚠️ NaN/Inf loss detected in {phase} phase (epoch {epoch+1}, batch {batch_count+1}). Loss: {loss.item()}. Skipping update for this batch.")
                        if nan_inf_counter > max_nan_inf_tolerance:
                            warnings.warn(f"   Exceeded NaN/Inf tolerance ({max_nan_inf_tolerance}). Consider checking model stability, learning rate, or data.")
                        del outputs, loss
                        torch.cuda.empty_cache()  # Try to clear cache if OOM might be related
                        continue  # Skip backprop and metric calculation for this batch

                    # Get predictions for metrics calculation (use cpu for sklearn)
                    preds = outputs.argmax(dim=1).detach().cpu().numpy()
                    labels_cpu = labels.detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels_cpu)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        if is_cuda and use_amp:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            if clip_grad_norm is not None:
                                torch_utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        elif is_tpu:  # Specific handling for TPU
                            loss.backward()
                            if clip_grad_norm is not None:
                                torch_utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            xm.optimizer_step(optimizer, barrier=True)  # TPU optimizer step
                        else:  # Standard CPU or CUDA without AMP
                            loss.backward()
                            if clip_grad_norm is not None:
                                torch_utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                batch_count += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                del inputs, labels, outputs, loss, preds, labels_cpu
                if is_cuda:
                    torch.cuda.empty_cache()

            # --- Epoch End Calculation ---
            phase_duration = time.time() - phase_start_time
            # --- Debug: Print final batch count ---
            if phase == 'val':
                print(f"Debug VAL: Finished validation loop. Final batch_count = {batch_count}")
            # --- End Debug ---
            try:
                num_samples = len(dataloaders[phase].dataset)
            except TypeError:  # Handle cases where dataset might not have __len__ (though unlikely for ImageFolder)
                print(f"⚠️ Could not determine dataset size for phase '{phase}'. Using len(all_labels) if available.")
                num_samples = len(all_labels) if len(all_labels) > 0 else 0  # Fallback

            if batch_count == 0 or num_samples == 0:
                # --- Enhanced Warning ---
                print(f"⚠️ No valid batches processed in {phase} phase for epoch {epoch+1} (batch_count={batch_count}, num_samples={num_samples}). Skipping metrics calculation.")
                # --- End Enhanced Warning ---
                history[f'{phase}_loss'].append(float('nan'))
                history[f'{phase}_acc_macro'].append(float('nan'))
                history[f'{phase}_acc_weighted'].append(float('nan'))
                if phase == 'val':
                    history['val_precision_macro'].append(float('nan'))
                    history['val_recall_macro'].append(float('nan'))
                    history['val_f1_macro'].append(float('nan'))
                    history['val_precision_weighted'].append(float('nan'))
                    history['val_recall_weighted'].append(float('nan'))
                    history['val_f1_weighted'].append(float('nan'))
                continue

            epoch_loss = running_loss / len(all_labels)

            if len(all_preds) > 0 and len(all_labels) > 0:
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='macro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )
                epoch_acc_macro = accuracy_score(all_labels, all_preds)
                epoch_acc_macro_avg_per_class = recall_macro
                epoch_acc_weighted = accuracy_score(all_labels, all_preds)
            else:
                print(f"⚠️ No predictions generated for {phase} phase in epoch {epoch+1}. Setting metrics to 0.")
                epoch_acc_macro_avg_per_class, epoch_acc_weighted = 0.0, 0.0
                precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0
                precision_weighted, recall_weighted, f1_weighted = 0.0, 0.0, 0.0

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc_macro'].append(epoch_acc_macro_avg_per_class)
            history[f'{phase}_acc_weighted'].append(epoch_acc_weighted)

            if phase == 'val':
                history['val_precision_macro'].append(precision_macro)
                history['val_recall_macro'].append(recall_macro)
                history['val_f1_macro'].append(f1_macro)
                history['val_precision_weighted'].append(precision_weighted)
                history['val_recall_weighted'].append(recall_weighted)
                history['val_f1_weighted'].append(f1_weighted)

            print(f'{phase.capitalize():<5} Loss: {epoch_loss:.4f} | Acc (Mac): {epoch_acc_macro_avg_per_class:.4f} | Acc (Wgt): {epoch_acc_weighted:.4f} | Time: {phase_duration:.2f}s')
            if phase == 'val':
                print(f'      P(Mac): {precision_macro:.4f} | R(Mac): {recall_macro:.4f} | F1(Mac): {f1_macro:.4f}')
                print(f'      P(Wgt): {precision_weighted:.4f} | R(Wgt): {recall_weighted:.4f} | F1(Wgt): {f1_weighted:.4f}')

            if phase == 'val':
                # --- Correctly select the metric value based on primary_metric ---
                if primary_metric == 'val_acc_macro':
                    current_val_metric = epoch_acc_macro_avg_per_class  # Use average recall for macro accuracy
                elif primary_metric == 'val_acc_weighted':
                    current_val_metric = epoch_acc_weighted
                elif primary_metric == 'val_f1_macro':
                    current_val_metric = f1_macro
                elif primary_metric == 'val_f1_weighted':
                    current_val_metric = f1_weighted
                # Add other metrics like precision/recall if needed
                else:
                    # Default or fallback if primary_metric is misconfigured
                    warnings.warn(f"Unrecognized primary_metric '{primary_metric}'. Defaulting to 'val_acc_macro'.")
                    current_val_metric = epoch_acc_macro_avg_per_class

                # --- Scheduler Step ---
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_metric)
                elif scheduler is not None:
                    # Step other schedulers like StepLR, CosineAnnealingLR
                    scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']
                print(f"   LR after scheduler step: {current_lr:.6f}")

                # --- Check for Improvement and Save Best Model ---
                if current_val_metric > best_val_metric:
                    print(f'✅ {primary_metric} improved ({best_val_metric:.4f} --> {current_val_metric:.4f}). Saving model to {save_path}')
                    best_val_metric = current_val_metric
                    # Save the model state dict
                    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                    if is_tpu:
                        # Save for TPU using xm.save
                        xm.save(model_to_save.state_dict(), save_path)
                    else:
                        # Standard PyTorch save
                        torch.save(model_to_save.state_dict(), save_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'📉 {primary_metric} did not improve for {epochs_no_improve} epoch(s). Best: {best_val_metric:.4f}')

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f}s")

        if epochs_no_improve >= patience:
            print(f'\n⏰ Early stopping triggered after {epoch+1} epochs ({patience} epochs without improvement on {primary_metric}).')
            break

    time_elapsed = time.time() - since
    print(f'\n🏁 Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'🏆 Best {primary_metric}: {best_val_metric:.4f} (achieved after epoch {epoch + 1 - epochs_no_improve})')

    try:
        max_len = max(len(v) for v in history.values())
        for k, v in history.items():
            if len(v) < max_len:
                padding_val = float('nan') if 'loss' in k or 'acc' in k or 'f1' in k or 'precision' in k or 'recall' in k else -1
                history[k].extend([padding_val] * (max_len - len(v)))

        history_df = pd.DataFrame(history)
        history_df.set_index('epoch', inplace=True)
        history_df.to_csv(log_path, index=True, float_format='%.6f')
        print(f"💾 Training log saved to {log_path}")
    except Exception as e:
        print(f"⚠️ Could not save training log to {log_path}: {e}")
        print("   History data:", history)

    print(f"🔄 Loading best model weights from {save_path} into model...")
    try:
        best_wts_loaded = torch.load(save_path, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(best_wts_loaded)
        else:
            model.load_state_dict(best_wts_loaded)
        print("   Best weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Best model file not found at {save_path}. Returning model with last epoch weights.")
    except Exception as e:
        print(f"❌ Error loading best model weights: {e}. Returning model with last epoch weights.")

    return model, history
