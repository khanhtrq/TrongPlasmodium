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
                log_path='training_log.csv', clip_grad_norm=1.0):  # Added clip_grad_norm parameter
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    is_cuda = device.type == 'cuda'
    is_tpu = _tpu_available and 'xla' in str(device)

    scaler = torch.amp.GradScaler(enabled=(use_amp and is_cuda))

    history = {
        'train_loss': [], 'train_acc_macro': [], 'train_acc_weighted': [],
        'val_loss': [], 'val_acc_macro': [], 'val_acc_weighted': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    # Initialize nan counter for monitoring
    nan_count = 0
    max_nan_threshold = 5  # How many NaNs to tolerate before reducing learning rate

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # Check for too many NaNs and reduce learning rate if needed
        if nan_count >= max_nan_threshold:
            print(f"⚠️ Detected {nan_count} NaN losses. Reducing learning rate by 10x")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            nan_count = 0  # Reset counter after adjusting

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []
            batch_count = 0  # Track the number of batches

            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} {epoch+1}', leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Check for NaN loss
                    if torch.isnan(loss).item() or torch.isinf(loss).item():
                        if phase == 'train':
                            nan_count += 1
                            warnings.warn(f"⚠️ NaN/Inf loss detected in batch (epoch {epoch+1})")
                            # Skip this batch for backprop
                            continue
                    
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    if phase == 'train':
                        if is_cuda and use_amp:
                            scaler.scale(loss).backward()
                            
                            # Apply gradient clipping to prevent explosion
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            
                            # Apply gradient clipping for non-AMP training too
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            
                            optimizer.step()

                        if is_tpu:
                            xm.mark_step()

                running_loss += loss.item() * inputs.size(0)
                batch_count += 1

            # Safety check for zero batches
            if batch_count == 0:
                print(f"⚠️ No valid batches in {phase} phase, epoch {epoch+1}")
                continue
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # Handle empty prediction lists (could happen if all batches had NaN loss)
            if len(all_preds) == 0 or len(all_labels) == 0:
                print(f"⚠️ No valid predictions in {phase} phase")
                precision_macro, recall_macro, f1_macro = 0, 0, 0
                precision_weighted, recall_weighted, f1_weighted = 0, 0, 0
            else:
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='macro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )

            epoch_acc_macro = recall_macro
            epoch_acc_weighted = recall_weighted

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc_macro'].append(epoch_acc_macro)
            history[f'{phase}_acc_weighted'].append(epoch_acc_weighted)

            if phase == 'val':
                history['val_precision'].append(precision_macro)
                history['val_recall'].append(recall_macro)
                history['val_f1'].append(f1_macro)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Macro Acc: {epoch_acc_macro:.4f} Weighted Acc: {epoch_acc_weighted:.4f}')

            if phase == 'val':
                # Only step scheduler at end of validation phase
                scheduler.step()
                if epoch_acc_macro > best_acc:
                    best_acc = epoch_acc_macro
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, save_path)
                    print(f'✅ Best model saved at epoch {epoch+1}')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'⏰ Early stopping at epoch {epoch+1}')
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Convert history to DataFrame and save as CSV
    try:
        # Ensure all lists in history have the same length (pad if early stopping)
        max_len = max(len(v) for v in history.values())
        for k, v in history.items():
            if len(v) < max_len:
                # Pad with NaN or the last value, NaN is generally better for missing data
                history[k].extend([float('nan')] * (max_len - len(v)))

        history_df = pd.DataFrame(history)
        history_df.to_csv(log_path, index=False)
        print(f"💾 Training log saved to {log_path}")
    except Exception as e:
        print(f"⚠️ Could not save training log to {log_path}: {e}")

    model.load_state_dict(best_model_wts)
    return model, history
