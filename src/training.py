import copy
import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support

try:
    import torch_xla.core.xla_model as xm
    _tpu_available = True
except ImportError:
    _tpu_available = False

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=25, patience=5, use_amp=True, save_path='best_model.pth'):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    is_cuda = device.type == 'cuda'
    is_tpu = _tpu_available and 'xla' in str(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_cuda))

    history = {
        'train_loss': [], 'train_acc_macro': [], 'train_acc_weighted': [],
        'val_loss': [], 'val_acc_macro': [], 'val_acc_weighted': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} {epoch+1}', leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_cuda and use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    if phase == 'train':
                        if is_cuda and use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        if is_tpu:
                            xm.mark_step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

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

    model.load_state_dict(best_model_wts)
    return model, history
