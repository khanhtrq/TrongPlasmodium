import copy
import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import warnings

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
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    is_cuda = device.type == 'cuda'
    is_tpu = _tpu_available and 'xla' in str(device)

    # Enable GradScaler only if using CUDA and AMP
    scaler = torch.amp.GradScaler(enabled=(use_amp and is_cuda))

    history = {
        'train_loss': [], 'train_acc_macro': [], 'train_acc_weighted': [],
        'val_loss': [], 'val_acc_macro': [], 'val_acc_weighted': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    nan_count = 0
    max_nan_threshold = 5 # Tolerance for NaN/Inf losses before reducing LR

    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"   Device: {device}, AMP: {use_amp}, Patience: {patience}, Clip Norm: {clip_grad_norm}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # Reduce learning rate if too many NaNs occurred
        if nan_count >= max_nan_threshold:
            print(f"⚠️ Detected {nan_count} NaN/Inf losses. Reducing learning rate by 10x.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"   New learning rate: {[pg['lr'] for pg in optimizer.param_groups]}")
            nan_count = 0 # Reset counter

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            all_preds = []
            all_labels = []
            batch_count = 0

            # Use tqdm for progress bar
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}', leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Use autocast for mixed precision if enabled
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Check for NaN/Inf loss
                    if torch.isnan(loss).item() or torch.isinf(loss).item():
                        warnings.warn(f"⚠️ NaN/Inf loss detected in {phase} phase (epoch {epoch+1}, batch {batch_count}). Skipping batch.")
                        if phase == 'train':
                            nan_count += 1
                        continue # Skip backprop for this batch

                    # Get predictions for metrics calculation
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        if is_cuda and use_amp:
                            scaler.scale(loss).backward()
                            # Unscale gradients before clipping
                            scaler.unscale_(optimizer)
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            # Scaler step
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            # Clip gradients for non-AMP training too
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            optimizer.step()

                        # Special step for TPU
                        if is_tpu:
                            xm.mark_step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                batch_count += 1
                # Update tqdm progress bar description
                pbar.set_postfix({'loss': loss.item()})


            # Ensure dataset is not empty and batches were processed
            if batch_count == 0 or len(dataloaders[phase].dataset) == 0:
                print(f"⚠️ No valid batches processed in {phase} phase for epoch {epoch+1}. Skipping metrics calculation.")
                continue

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # Calculate metrics only if predictions were made
            if len(all_preds) > 0 and len(all_labels) > 0:
                # Use macro average for primary accuracy metric (treats all classes equally)
                # Use weighted average for secondary accuracy metric (accounts for class imbalance)
                # Use zero_division=0 to return 0 for metrics where the denominator is 0 (e.g., precision in a class with no predictions)
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='macro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )
                # Accuracy is equivalent to recall in multiclass classification
                epoch_acc_macro = recall_macro
                epoch_acc_weighted = recall_weighted
            else:
                print(f"⚠️ No predictions generated for {phase} phase in epoch {epoch+1}. Setting metrics to 0.")
                epoch_acc_macro, epoch_acc_weighted = 0.0, 0.0
                precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0 # Needed for history

            # Log metrics
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc_macro'].append(epoch_acc_macro)
            history[f'{phase}_acc_weighted'].append(epoch_acc_weighted)

            if phase == 'val':
                history['val_precision'].append(precision_macro)
                history['val_recall'].append(recall_macro) # Same as epoch_acc_macro
                history['val_f1'].append(f1_macro)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Macro Acc: {epoch_acc_macro:.4f} | Weighted Acc: {epoch_acc_weighted:.4f}')
            if phase == 'val':
                 print(f'          Macro P: {precision_macro:.4f} | Macro R: {recall_macro:.4f} | Macro F1: {f1_macro:.4f}')


            # Early stopping and best model saving based on validation macro accuracy
            if phase == 'val':
                # Step the scheduler based on validation performance
                # Note: Some schedulers like ReduceLROnPlateau need the metric value
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                     scheduler.step(epoch_acc_macro) # Or epoch_loss, depending on goal
                else:
                     scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Current LR: {current_lr:.6f}")

                if epoch_acc_macro > best_acc:
                    print(f'✅ Validation Macro Accuracy improved ({best_acc:.4f} --> {epoch_acc_macro:.4f}). Saving model...')
                    best_acc = epoch_acc_macro
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, save_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'📉 Validation Macro Accuracy did not improve for {epochs_no_improve} epoch(s). Best: {best_acc:.4f}')

        # Check for early stopping
        if epochs_no_improve >= patience:
            print(f'\n⏰ Early stopping triggered after {epoch+1} epochs.')
            break

    time_elapsed = time.time() - since
    print(f'\n🏁 Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'🏆 Best Validation Macro Accuracy: {best_acc:.4f}')

    # Save training history log
    try:
        # Ensure all history lists have the same length (pad if early stopping occurred)
        max_len = max(len(v) for v in history.values())
        for k, v in history.items():
            if len(v) < max_len:
                # Pad with NaN for missing epochs due to early stopping
                history[k].extend([float('nan')] * (max_len - len(v)))

        history_df = pd.DataFrame(history)
        history_df.index.name = 'epoch' # Add epoch column
        history_df.to_csv(log_path, index=True)
        print(f"💾 Training log saved to {log_path}")
    except Exception as e:
        print(f"⚠️ Could not save training log to {log_path}: {e}")

    # Load best model weights back
    print(f"🔄 Loading best model weights from {save_path}")
    model.load_state_dict(torch.load(save_path))
    return model, history
