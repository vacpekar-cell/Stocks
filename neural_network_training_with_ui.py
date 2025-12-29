import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import gc
import time
import warnings
import re
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
warnings.filterwarnings('ignore')

# CUDA optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Efficient data augmentation
class FinancialDataset(Dataset):
    def __init__(self, features, targets, transform=False, noise_level=0.03, mixup_prob=0.2):
        self.features = features
        self.targets = targets
        self.transform = transform
        self.noise_level = noise_level
        self.mixup_prob = mixup_prob
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        
        if self.transform:
            if np.random.random() < 0.5:
                noise = torch.randn_like(x) * self.noise_level
                x = x + noise
                
            if np.random.random() < self.mixup_prob:
                mix_idx = np.random.randint(0, len(self.features))
                mix_x = self.features[mix_idx]
                mix_y = self.targets[mix_idx]
                lam = np.random.beta(0.2, 0.2)
                x = lam * x + (1 - lam) * mix_x
                y = lam * y + (1 - lam) * mix_y
                
        return x, y

# Optimized Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.leaky_relu(out, 0.1)
        out = self.dropout2(out)
        
        return out

# Enhanced ResNet model for financial predictions
class EfficientResNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.25, use_sigmoid=True, output_dim=3):
        super(EfficientResNet, self).__init__()
        self.use_sigmoid = use_sigmoid

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ]
        
        for i in range(len(hidden_dims)-1):
            layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_rate=dropout_rate*(0.9**i)))

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        
        if self.use_sigmoid:
            x = torch.sigmoid(x)
            
        return x


class ErrorEstimator(nn.Module):
    """Predict the absolute error of the primary model for a given horizon."""

    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = scaled_hidden_dims(input_dim)

        self.backbone = EfficientResNet(
            input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_sigmoid=False,
            output_dim=1,
        )

    def forward(self, x):
        raw = self.backbone(x)
        return F.softplus(raw)


def scaled_hidden_dims(input_dim: int) -> list[int]:
    base_input = 66
    base_dims = [512, 256, 128, 64]
    ratio = input_dim / base_input
    scaled = [max(32, int(round(dim * ratio))) for dim in base_dims]
    return scaled


def inverse_target_transform(values):
    if torch.is_tensor(values):
        clipped = torch.clamp(values, 1e-6, 1 - 1e-6)
        z = torch.atanh(1 - 2 * clipped)
        return torch.sign(z) * torch.pow(torch.abs(z), 4.0 / 3.0)

    arr = np.asarray(values)
    clipped = np.clip(arr, 1e-6, 1 - 1e-6)
    z = np.arctanh(1 - 2 * clipped)
    return np.sign(z) * (np.abs(z) ** (4.0 / 3.0))


HORIZON_ORDER = ["1w", "4w", "13w", "26w", "52w"]
TARGET_COLUMN_ALIASES = {
    "1w": {"target_1w", "1w", "1_week", "1-week", "1 week"},
    "4w": {"target_4w", "4w", "4_week", "4-week", "4 week"},
    "13w": {"target_13w", "13w", "13_week", "13-week", "13 week"},
    "26w": {"target_26w", "26w", "26_week", "26-week", "26 week"},
    "52w": {"target_52w", "52w", "52_week", "52-week", "52 week"},
}


def normalize_target_name(name: str) -> str:
    lower = name.strip().lower()
    if lower.startswith("target_"):
        lower = lower.split("target_", 1)[1]
    lower = lower.replace("weeks", "w").replace("week", "w").replace(" ", "")
    return lower


def map_columns_to_horizons(columns) -> dict:
    mapping = {}
    for col in columns:
        normalized = normalize_target_name(str(col))
        for horizon, aliases in TARGET_COLUMN_ALIASES.items():
            if normalized in aliases:
                mapping[horizon] = col
                break
    return mapping


def infer_horizon_from_filename(path: str) -> str | None:
    base = os.path.basename(path).lower()
    for horizon in HORIZON_ORDER:
        token = horizon.lower()
        if f"_{token}_" in base or base.startswith(f"{token}_") or base.endswith(f"_{token}.pt"):
            return horizon
    return None


def infer_timestamp_from_model(path: str) -> str | None:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    match = re.search(r"(\d{8}_\d{6})", name)
    if match:
        return match.group(1)

    parts = name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and len(parts[-2]) == 8 and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"

    return None

def infer_output_dim_from_state_dict(state_dict) -> int:
    """Try to infer the number of output nodes stored in a checkpoint."""

    if "output_layer.weight" in state_dict:
        return state_dict["output_layer.weight"].shape[0]

    if "output_layer.bias" in state_dict:
        return state_dict["output_layer.bias"].shape[0]

    raise ValueError("Nelze určit výstupní dimenzi z poskytnutého checkpointu.")

# Thread-safe logger
class ThreadSafeLogger:
    def __init__(self, callback, update_interval=100):
        self.queue = queue.Queue()
        self.callback = callback
        self.update_interval = update_interval
        self.last_update = time.time()
        
    def log(self, message):
        self.queue.put(message)
        current_time = time.time()
        if current_time - self.last_update > 0.1:
            self.process_queue()
            self.last_update = current_time
    
    def process_queue(self):
        messages = []
        while not self.queue.empty():
            try:
                messages.append(self.queue.get_nowait())
            except queue.Empty:
                break
        
        if messages:
            combined_message = "\n".join(messages)
            self.callback(combined_message)

# Training logger with visualization
class TrainingLogger:
    def __init__(self, thread_safe_logger, network_id, log_interval=5, patience=20):
        self.thread_safe_logger = thread_safe_logger
        self.network_id = network_id
        self.log_interval = log_interval
        self.patience = patience
        self.start_time = time.time()
        self.reset()
        
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.early_stopped = False
        self.val_metrics = {'rmse': []}
        
    def start_training(self):
        self.start_time = time.time()
        self.reset()
        
    def log(self, epoch, epochs, train_loss, val_loss, lr, val_rmse):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.val_metrics['rmse'].append(val_rmse)
        
        if val_loss < self.best_val_loss:
            is_best = True
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.no_improve_count = 0
        else:
            is_best = False
            self.no_improve_count += 1
        
        if epoch > 0:
            remain_epochs = epochs - epoch - 1
            eta = (elapsed / (epoch + 1)) * remain_epochs
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
        else:
            eta_min, eta_sec = 0, 0
        
        if epoch % self.log_interval == 0 or epoch < 5 or epoch == epochs-1 or is_best:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s"
            
            log_str = f"Síť {self.network_id}, Epocha {epoch+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.6f}"
            if is_best:
                log_str += " [NEJLEPŠÍ]"
            log_str += f" | Čas: {time_str}, ETA: {eta_min}m {eta_sec}s, RMSE: {val_rmse:.6f}"
            
            self.thread_safe_logger.log(log_str)
        
        if self.no_improve_count >= self.patience:
            self.early_stopped = True
            return True
            
        return False
    
    def finish(self):
        total_time = time.time() - self.start_time
        finish_str = f"Trénink dokončen za {total_time:.1f}s. Nejlepší val_loss={self.best_val_loss:.6f} (epocha {self.best_epoch+1})"
        if self.early_stopped:
            finish_str += " - Early stopping aktivován"
        self.thread_safe_logger.log(finish_str)
        return self.best_val_loss, self.best_epoch
    
    def get_plotting_data(self):
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'val_rmse': self.val_metrics['rmse']
        }

# Training thread
class TrainingThread(threading.Thread):
    def __init__(self, app, params):
        threading.Thread.__init__(self)
        self.app = app
        self.params = params
        self.stop_event = threading.Event()
        self.daemon = True
        
    def run(self):
        try:
            self.app.thread_safe_logger.log("Trénování spuštěno v novém vlákně...")
            
            seed = self.params['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            use_gpu = torch.cuda.is_available() and self.params['use_gpu']
            use_mixed_precision = use_gpu and self.params['use_mixed_precision']
            device = torch.device("cuda:0" if use_gpu else "cpu")
            
            timestamp = self.params['timestamp']
            
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.app.thread_safe_logger.log(f"Trénování na: {device} (ID: {timestamp})")
            
            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                self.app.thread_safe_logger.log(f"GPU: {gpu_name}")
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                self.app.thread_safe_logger.log(f"Výchozí využití GPU paměti: {mem_allocated:.1f}MB")
            
            target_tasks = self.params['target_tasks']
            batch_size = self.params['batch_size']
            epochs = self.params['epochs']
            dropout_rate = self.params['dropout_rate']
            use_augmentation = self.params['use_augmentation']
            noise_level = self.params['noise_level']
            mixup_prob = self.params['mixup_prob']
            patience = self.params['patience']
            scheduler_type = self.params['scheduler_type']
            min_lr = self.params['min_lr']
            learning_rate = self.params['learning_rate']
            weight_decay = self.params['weight_decay']
            use_sigmoid = self.params['use_sigmoid']

            results = []
            trained_models = []

            for task in target_tasks:
                if self.stop_event.is_set():
                    self.app.thread_safe_logger.log("Trénování přerušeno uživatelem.")
                    break

                horizon = task['horizon']
                self.app.current_network_id = horizon
                fold_tasks = task.get('folds', [])
                if not fold_tasks:
                    self.app.thread_safe_logger.log(f"Varování: Žádné foldy pro {horizon}, přeskočeno.")
                    continue

                self.app.thread_safe_logger.log(
                    f"\n--- Trénování horizontu {horizon} ({len(fold_tasks)} rolling foldů) ---"
                )

                best_fold = None
                best_val_loss = float('inf')
                best_entry = None

                for fold_task in fold_tasks:
                    if self.stop_event.is_set():
                        break

                    fold_id = fold_task['fold_id']
                    display_id = f"{horizon}/fold{fold_id}"
                    file_tag = f"{horizon}_fold{fold_id}"
                    has_meta = len(fold_task['meta_train_X']) > 0 and len(fold_task['meta_val_X']) > 0

                    self.app.thread_safe_logger.log(
                        f"\n--- Fold {fold_id}: trénink {display_id} ---"
                    )

                    train_outcome = self.train_single_model(
                        network_id=horizon,
                        display_id=display_id,
                        file_tag=file_tag,
                        X_train=fold_task['X_train'],
                        y_train=fold_task['y_train'],
                        X_val=fold_task['X_val'],
                        y_val=fold_task['y_val'],
                        meta_train_X=fold_task['meta_train_X'],
                        meta_val_X=fold_task['meta_val_X'],
                        meta_train_y=fold_task['meta_train_y'],
                        meta_val_y=fold_task['meta_val_y'],
                        timestamp=timestamp,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        device=device,
                        use_mixed_precision=use_mixed_precision,
                        use_augmentation=use_augmentation,
                        noise_level=noise_level,
                        mixup_prob=mixup_prob,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        use_sigmoid=use_sigmoid,
                        dropout_rate=dropout_rate,
                        scheduler_type=scheduler_type,
                        min_lr=min_lr,
                        output_dim=1,
                        run_meta=has_meta,
                    )

                    if self.stop_event.is_set():
                        break

                    fold_val = train_outcome.get('val_loss', float('inf'))
                    if fold_val < best_val_loss:
                        best_val_loss = fold_val
                        best_fold = fold_id
                        best_entry = train_outcome

                if self.stop_event.is_set():
                    break

                if not best_entry:
                    self.app.thread_safe_logger.log(f"Varování: {horizon} nemá žádný úspěšný fold.")
                    continue

                self.app.thread_safe_logger.log(
                    f"Nejlepší fold pro {horizon}: {best_fold} (val_loss {best_val_loss:.6f})"
                )

                best_model_state = best_entry.get('model_state')
                if best_model_state is not None:
                    model_file = f"model_resnet_{horizon}_{timestamp}.pt"
                    torch.save(best_model_state, model_file)
                    self.app.thread_safe_logger.log(f"Uloženo nejlepší modelové váhy jako '{model_file}'")

                best_meta_state = best_entry.get('meta_state')
                if best_meta_state is not None:
                    meta_file = f"meta_resnet_{horizon}_{timestamp}.pt"
                    torch.save(best_meta_state, meta_file)
                    self.app.thread_safe_logger.log(f"Uloženo nejlepší váhy metasítě jako '{meta_file}'")

                if self.stop_event.is_set():
                    break

                results.append(
                    {
                        'timestamp': timestamp,
                        'model_type': 'resnet',
                        'horizon': horizon,
                        'val_loss': best_entry.get('val_loss', float('inf')),
                        'meta_val_loss': best_entry.get('meta_val_loss'),
                        'best_fold': best_fold,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'dropout_rate': dropout_rate,
                        'use_augmentation': use_augmentation,
                        'use_sigmoid': use_sigmoid,
                        'scheduler_type': scheduler_type,
                        'noise_level': noise_level,
                        'mixup_prob': mixup_prob,
                        'seed': seed,
                    }
                )
                trained_models.append(best_entry)

            if not self.stop_event.is_set():
                self.app.thread_safe_logger.log(f"\n=== Trénink dokončen (ID: {timestamp}) ===")

                try:
                    results_file = "training_results.csv"
                    result_df = pd.DataFrame(results)

                    if os.path.exists(results_file):
                        existing_results = pd.read_csv(results_file)
                        updated_results = pd.concat([existing_results, result_df], ignore_index=True)
                        updated_results.to_csv(results_file, index=False)
                    else:
                        result_df.to_csv(results_file, index=False)

                    self.app.thread_safe_logger.log(f"Výsledky tréninku uloženy do '{results_file}'")

                except Exception as e:
                    self.app.thread_safe_logger.log(f"Varování: Chyba při ukládání výsledků: {str(e)}")

                try:
                    bundle = {
                        'bundle_type': 'training_bundle',
                        'timestamp': timestamp,
                        'use_sigmoid': use_sigmoid,
                        'models': {},
                    }

                    for entry in trained_models:
                        if not entry or not entry.get('horizon'):
                            continue
                        bundle['models'][entry['horizon']] = {
                            'model_state': entry.get('model_state'),
                            'meta_state': entry.get('meta_state'),
                            'val_loss': entry.get('val_loss'),
                            'meta_val_loss': entry.get('meta_val_loss'),
                            'output_dim': entry.get('output_dim', 1),
                            'input_dim': entry.get('input_dim'),
                        }

                    bundle_file = f"models_bundle_{timestamp}.pt"
                    torch.save(bundle, bundle_file)
                    self.app.thread_safe_logger.log(
                        f"Modely a metasítě z běhu {timestamp} uloženy do '{bundle_file}'."
                    )
                except Exception as e:
                    self.app.thread_safe_logger.log(f"Varování: Nepodařilo se uložit bundle modelů: {e}")

            self.app.root.after(0, self.app.enable_buttons)
            
        except Exception as e:
            import traceback
            error_msg = f"\nCHYBA v trénovacím vlákně: {str(e)}\n{traceback.format_exc()}"
            self.app.thread_safe_logger.log(error_msg)
            self.app.root.after(0, self.app.enable_buttons)
    
    def train_single_model(self, network_id, X_train, y_train, X_val, y_val, meta_train_X, meta_val_X,
                           meta_train_y, meta_val_y, timestamp, batch_size, epochs, patience, device,
                           use_mixed_precision=False, use_augmentation=False, noise_level=0.03,
                           mixup_prob=0.2, learning_rate=0.001, weight_decay=0.0001, use_sigmoid=True,
                           dropout_rate=0.25, scheduler_type='cosine', min_lr='1e-6',
                           output_dim=3, display_id=None, file_tag=None, run_meta=True):
        try:
            if self.stop_event.is_set():
                return {'horizon': network_id, 'val_loss': float('inf'), 'meta_val_loss': None}
                
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                self.app.thread_safe_logger.log(f"GPU využití před tréninkem: {mem_allocated:.1f}MB")
            
            display_id = display_id or network_id
            file_tag = file_tag or network_id

            if use_augmentation:
                train_dataset = FinancialDataset(X_train, y_train, transform=True,
                                              noise_level=noise_level, mixup_prob=mixup_prob)
            else:
                train_dataset = TensorDataset(X_train, y_train)
                
            val_dataset = TensorDataset(X_val, y_val)
            
            num_workers = 0
            pin_memory = device.type == 'cuda'
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size*2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            model = EfficientResNet(
                X_train.shape[1],
                hidden_dims=scaled_hidden_dims(X_train.shape[1]),
                dropout_rate=dropout_rate,
                use_sigmoid=use_sigmoid,
                output_dim=output_dim
            )
            
            model.to(device)
            
            activation = "Sigmoid" if use_sigmoid else "LeakyReLU"
            self.app.thread_safe_logger.log(f"Architektura: resnet s aktivací {activation}")
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.app.thread_safe_logger.log(f"Počet trénovatelných parametrů: {num_params:,}")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=float(min_lr)
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=float(min_lr)
                )
            
            criterion = nn.MSELoss()
            
            logger = TrainingLogger(self.app.thread_safe_logger, display_id, patience=patience)
            logger.start_training()
            self.app.training_loggers[str(display_id)] = logger
            self.app.root.after(0, lambda: self.app.add_network_option(display_id))
            
            scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
            
            best_val_loss = float('inf')
            best_model_state = None
            meta_state = None
            meta_val_loss = None
            
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    self.app.thread_safe_logger.log(f"Trénink sítě {display_id} přerušen.")
                    break
                
                model.train()
                train_loss = 0.0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if self.stop_event.is_set():
                        break
                        
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    
                    if epoch == 0 and batch_idx == 0:
                        if device.type == 'cuda':
                            mem_allocated = torch.cuda.memory_allocated() / 1024**2
                            self.app.thread_safe_logger.log(f"Epocha {epoch+1}, Využití GPU: {mem_allocated:.1f}MB")
                
                if self.stop_event.is_set():
                    break
                
                train_loss /= len(train_loader.dataset)
                
                model.eval()
                val_loss = 0.0
                val_outputs_all = []
                val_targets_all = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        if self.stop_event.is_set():
                            break
                            
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        
                        if use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                        val_loss += loss.item() * inputs.size(0)
                        val_outputs_all.append(outputs.cpu())
                        val_targets_all.append(targets.cpu())
                
                if self.stop_event.is_set():
                    break
                
                val_loss /= len(val_loader.dataset)
                
                val_outputs_cat = torch.cat(val_outputs_all)
                val_targets_cat = torch.cat(val_targets_all)
                
                mse = F.mse_loss(val_outputs_cat, val_targets_cat).item()
                rmse = np.sqrt(mse)
                
                scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                
                early_stop = logger.log(epoch, epochs, train_loss, val_loss, current_lr, rmse)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
                if early_stop:
                    self.app.thread_safe_logger.log(f"Early stopping na epoše {epoch+1}")
                    break
                
                if epoch % 5 == 0:
                    plot_data = logger.get_plotting_data()
                    self.app.root.after(0, lambda: self.app.update_plot_from_thread(str(display_id), plot_data))
            
            if self.stop_event.is_set():
                return {'horizon': network_id, 'val_loss': float('inf'), 'meta_val_loss': None}

            logger.finish()

            model_file = f"model_resnet_{file_tag}_{timestamp}.pt"
            model.load_state_dict(best_model_state)
            model.to(device)
            model.eval()

            torch.save(best_model_state, model_file)

            self.app.thread_safe_logger.log(f"Síť {display_id} dokončena. Nejlepší validační ztráta: {best_val_loss:.6f}")
            self.app.thread_safe_logger.log(f"Model uložen jako '{model_file}'")

            try:
                if not run_meta or len(meta_train_X) == 0 or len(meta_val_X) == 0:
                    self.app.thread_safe_logger.log(
                        f"Metasíť pro {display_id} přeskočena: není k dispozici meta segment."
                    )
                else:
                    meta_val_loss, meta_state = self.train_meta_model(
                        display_id,
                        timestamp,
                        model_state=best_model_state,
                        X_train=meta_train_X,
                        y_train=meta_train_y,
                        X_val=meta_val_X,
                        y_val=meta_val_y,
                        batch_size=batch_size,
                        device=device,
                        use_mixed_precision=use_mixed_precision,
                        dropout_rate=dropout_rate,
                        use_sigmoid=use_sigmoid,
                        file_tag=file_tag,
                    )
                    if meta_state is not None:
                        self.app.thread_safe_logger.log(
                            f"Metasíť pro {display_id} dokončena. Nejlepší validační ztráta: {meta_val_loss:.6f}"
                        )
            except Exception as exc:
                self.app.thread_safe_logger.log(
                    f"Varování: trénink metasítě pro {display_id} selhal: {exc}"
                )

            plot_data = logger.get_plotting_data()
            self.app.root.after(0, lambda: self.app.update_plot_from_thread(str(display_id), plot_data))
            
            del model, optimizer, criterion
            if scaler is not None:
                del scaler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            return {
                'horizon': network_id,
                'val_loss': best_val_loss,
                'meta_val_loss': meta_val_loss,
                'model_state': best_model_state,
                'meta_state': meta_state,
                'output_dim': output_dim,
                'input_dim': X_train.shape[1],
            }

        except Exception as e:
            import traceback
            error_msg = f"\nCHYBA při tréninku sítě {network_id}: {str(e)}\n{traceback.format_exc()}"
            self.app.thread_safe_logger.log(error_msg)
            return {'horizon': network_id, 'val_loss': float('inf'), 'meta_val_loss': None}

    def train_meta_model(
        self,
        network_id,
        timestamp,
        model_state,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        device,
        use_mixed_precision,
        dropout_rate,
        use_sigmoid,
        file_tag=None,
    ):
        try:
            base_model = EfficientResNet(
                X_train.shape[1],
                hidden_dims=scaled_hidden_dims(X_train.shape[1]),
                dropout_rate=dropout_rate,
                use_sigmoid=use_sigmoid,
                output_dim=1,
            )
            base_model.load_state_dict(model_state)
            base_model.to(device)
            base_model.eval()

            def build_meta_parts(features: torch.Tensor, targets: torch.Tensor):
                with torch.no_grad():
                    preds = base_model(features.to(device, non_blocking=True))
                    preds_raw = inverse_target_transform(preds).cpu()
                target_raw = inverse_target_transform(targets).cpu()
                meta_inputs = torch.cat([features.cpu(), preds_raw], dim=1)
                meta_targets = torch.abs(preds_raw - target_raw)
                return meta_inputs, meta_targets

            train_meta_X, train_meta_y = build_meta_parts(X_train, y_train)
            val_meta_X, val_meta_y = build_meta_parts(X_val, y_val)

            meta_model = ErrorEstimator(
                train_meta_X.shape[1],
                hidden_dims=scaled_hidden_dims(train_meta_X.shape[1]),
                dropout_rate=dropout_rate * 0.5,
            )
            meta_model.to(device)

            optimizer = torch.optim.AdamW(meta_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
            criterion = nn.MSELoss()
            scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

            train_loader = DataLoader(
                TensorDataset(train_meta_X, train_meta_y),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_meta_X, val_meta_y),
                batch_size=batch_size * 2,
                shuffle=False,
            )

            best_val_loss = float('inf')
            best_state = None
            patience = 20
            patience_counter = 0
            meta_epochs = 150

            meta_logger_id = f"{network_id}-meta"
            logger = TrainingLogger(self.app.thread_safe_logger, meta_logger_id, patience=patience)
            logger.start_training()
            self.app.training_loggers[str(meta_logger_id)] = logger
            self.app.root.after(0, lambda: self.app.add_network_option(meta_logger_id))

            for epoch in range(meta_epochs):
                if self.stop_event.is_set():
                    break

                meta_model.train()
                train_loss = 0.0

                for inputs, targets in train_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = meta_model(inputs)
                            loss = criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = meta_model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
                        optimizer.step()

                    train_loss += loss.item() * inputs.size(0)

                train_loss /= len(train_loader.dataset)

                meta_model.eval()
                val_loss = 0.0
                val_outputs = []
                val_targets = []
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        if use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = meta_model(inputs)
                                loss = criterion(outputs, targets)
                        else:
                            outputs = meta_model(inputs)
                            loss = criterion(outputs, targets)

                        val_loss += loss.item() * inputs.size(0)
                        val_outputs.append(outputs.cpu())
                        val_targets.append(targets.cpu())

                val_loss /= len(val_loader.dataset)
                scheduler.step()
                val_outputs_cat = torch.cat(val_outputs)
                val_targets_cat = torch.cat(val_targets)
                mse = F.mse_loss(val_outputs_cat, val_targets_cat).item()
                rmse = np.sqrt(mse)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu() for k, v in meta_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.log(epoch, meta_epochs, train_loss, val_loss, optimizer.param_groups[0]['lr'], rmse)

                if patience_counter >= patience:
                    break

                if epoch % 5 == 0:
                    plot_data = logger.get_plotting_data()
                    self.app.root.after(0, lambda: self.app.update_plot_from_thread(str(meta_logger_id), plot_data))

            if best_state is not None:
                meta_model.load_state_dict(best_state)

            logger.finish()

            file_tag = file_tag or network_id
            meta_model_file = f"meta_resnet_{file_tag}_{timestamp}.pt"
            torch.save(meta_model.state_dict(), meta_model_file)
            self.app.thread_safe_logger.log(f"Metasíť uložena jako '{meta_model_file}'")

            del base_model, meta_model, optimizer, criterion
            if scaler is not None:
                del scaler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            final_state = best_state if best_state is not None else {k: v.cpu() for k, v in meta_model.state_dict().items()}
            return best_val_loss, final_state

        except Exception as e:
            import traceback
            error_msg = f"CHYBA při tréninku metasítě {network_id}: {str(e)}\n{traceback.format_exc()}"
            self.app.thread_safe_logger.log(error_msg)
            return float('inf'), None

# Main application
class NeuralNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimalizované neuronové sítě pro finanční predikce v2.1")
        self.root.geometry("1200x800")
        
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.predict_file = tk.StringVar()
        self.model_type = tk.StringVar(value="resnet")
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.use_mixed_precision = tk.BooleanVar(value=torch.cuda.is_available())
        self.use_augmentation = tk.BooleanVar(value=True)
        self.use_sigmoid = tk.BooleanVar(value=True)
        self.training_columns = None
        self.training_loggers = {}
        self.best_models = {}
        self.current_network_id = None
        self.training_thread = None
        self.loaded_models = []  # Seznam pro načtené modely
        
        self.create_widgets()
        
        self.thread_safe_logger = ThreadSafeLogger(self.update_log_from_thread)
    
    def create_widgets(self):
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tab_control = ttk.Notebook(left_frame)
        
        train_tab = ttk.Frame(tab_control)
        predict_tab = ttk.Frame(tab_control)
        visualize_tab = ttk.Frame(tab_control)
        
        tab_control.add(train_tab, text="Trénování")
        tab_control.add(predict_tab, text="Predikce")
        tab_control.add(visualize_tab, text="Vizualizace")
        
        tab_control.pack(expand=1, fill=tk.BOTH)
        
        file_frame = ttk.LabelFrame(train_tab, text="Výběr CSV souborů", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Vstupní CSV (trénování):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Procházet", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="Výstupní CSV (trénování):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Procházet", command=self.browse_output).grid(row=1, column=2, padx=5)
        
        self.loaded_models_var = tk.StringVar()
        
        param_frame = ttk.LabelFrame(train_tab, text="Parametry trénování", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Počet epoch:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs = ttk.Entry(param_frame, width=8)
        self.epochs.insert(0, "500")
        self.epochs.grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Velikost dávky:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(15,0))
        self.batch_size = ttk.Entry(param_frame, width=8)
        self.batch_size.insert(0, "256")
        self.batch_size.grid(row=0, column=3, padx=5)
        
        ttk.Label(param_frame, text="Počet sítí:").grid(row=0, column=4, sticky=tk.W, pady=5, padx=(15,0))
        self.num_networks = ttk.Entry(param_frame, width=8, state='disabled')
        self.num_networks.insert(0, "1")
        self.num_networks.grid(row=0, column=5, padx=5)
        
        ttk.Label(param_frame, text="Learning rate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.learning_rate = ttk.Entry(param_frame, width=8)
        self.learning_rate.insert(0, "0.001")
        self.learning_rate.grid(row=1, column=1, padx=5)
        
        ttk.Label(param_frame, text="Patience:").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(15,0))
        self.patience = ttk.Entry(param_frame, width=8)
        self.patience.insert(0, "20")
        self.patience.grid(row=1, column=3, padx=5)
        
        ttk.Label(param_frame, text="Weight decay:").grid(row=1, column=4, sticky=tk.W, pady=5, padx=(15,0))
        self.weight_decay = ttk.Entry(param_frame, width=8)
        self.weight_decay.insert(0, "0.0001")
        self.weight_decay.grid(row=1, column=5, padx=5)
        
        ttk.Label(param_frame, text="Dropout rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dropout_rate = ttk.Entry(param_frame, width=8)
        self.dropout_rate.insert(0, "0.25")
        self.dropout_rate.grid(row=2, column=1, padx=5)
        
        ttk.Label(param_frame, text="Random seed:").grid(row=2, column=2, sticky=tk.W, pady=5, padx=(15,0))
        self.random_seed = ttk.Entry(param_frame, width=8)
        self.random_seed.insert(0, "42")
        self.random_seed.grid(row=2, column=3, padx=5)
        
        ttk.Checkbutton(param_frame, text="Použít GPU", 
                      variable=self.use_gpu).grid(row=3, column=0, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(param_frame, text="Mixed precision", 
                      variable=self.use_mixed_precision).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(param_frame, text="Datová augmentace", 
                      variable=self.use_augmentation).grid(row=3, column=2, sticky=tk.W, pady=5, padx=(15,0))
        
        ttk.Checkbutton(param_frame, text="Sigmoid výstup", 
                      variable=self.use_sigmoid).grid(row=3, column=3, sticky=tk.W, pady=5)
        
        aug_frame = ttk.LabelFrame(train_tab, text="Parametry augmentace", padding=10)
        aug_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(aug_frame, text="Úroveň šumu:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.noise_level = ttk.Entry(aug_frame, width=8)
        self.noise_level.insert(0, "0.03")
        self.noise_level.grid(row=0, column=1, padx=5)
        
        ttk.Label(aug_frame, text="Mixup pravděpodobnost:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(15,0))
        self.mixup_prob = ttk.Entry(aug_frame, width=8)
        self.mixup_prob.insert(0, "0.2")
        self.mixup_prob.grid(row=0, column=3, padx=5)
        
        scheduler_frame = ttk.LabelFrame(train_tab, text="Scheduler parametry", padding=10)
        scheduler_frame.pack(fill=tk.X, pady=5)
        
        self.scheduler_type = tk.StringVar(value="cosine")
        ttk.Label(scheduler_frame, text="Typ scheduleru:").grid(row=0, column=0, sticky=tk.W, pady=5)
        scheduler_combo = ttk.Combobox(scheduler_frame, textvariable=self.scheduler_type, width=15)
        scheduler_combo['values'] = ('cosine',)
        scheduler_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(scheduler_frame, text="Min LR:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(15,0))
        self.min_lr = ttk.Entry(scheduler_frame, width=8)
        self.min_lr.insert(0, "1e-6")
        self.min_lr.grid(row=0, column=3, padx=5)
        
        buttons_frame = ttk.Frame(train_tab)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(buttons_frame, text="Spustit trénování", 
                                     command=self.train_models, width=20)
        self.train_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = ttk.Button(buttons_frame, text="Zastavit trénování", 
                                     command=self.stop_training, width=20, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(buttons_frame, text="Vymazat log", 
                 command=self.clear_log, width=15).pack(side=tk.RIGHT, padx=10)
        
        predict_frame = ttk.LabelFrame(predict_tab, text="Výběr souboru pro predikci", padding=10)
        predict_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(predict_frame, text="Vstupní CSV (bez hlavičky):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(predict_frame, textvariable=self.predict_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(predict_frame, text="Procházet", command=self.browse_predict).grid(row=0, column=2, padx=5)
        
        # Frame pro načtení modelů pro predikci
        load_frame = ttk.LabelFrame(predict_tab, text="Načtení modelů", padding=10)
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_frame, text="Načtené modely pro predikci:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(load_frame, textvariable=self.loaded_models_var, width=50, state='readonly').grid(row=0, column=1, padx=5)
        ttk.Button(load_frame, text="Načíst modely", command=self.load_models).grid(row=0, column=2, padx=5)
        
        pred_settings = ttk.LabelFrame(predict_tab, text="Nastavení predikce", padding=10)
        pred_settings.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(pred_settings, text="Použít GPU", 
                      variable=self.use_gpu).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(pred_settings, text="Mixed precision",
                      variable=self.use_mixed_precision).grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(pred_settings, text="Predikční batch size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pred_batch_size = ttk.Entry(pred_settings, width=8)
        self.pred_batch_size.insert(0, "256")
        self.pred_batch_size.grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Label(pred_settings, text="Počet simulací (vzorků):").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.pred_samples = ttk.Entry(pred_settings, width=8)
        self.pred_samples.insert(0, "50")
        self.pred_samples.grid(row=1, column=3, padx=5, sticky=tk.W)
        
        ttk.Label(pred_settings, text="Konkrétní model (pokud nepoužíváte načtené modely):").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.model_file = ttk.Entry(pred_settings, width=50)
        self.model_file.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(pred_settings, text="Vybrat model", command=self.browse_model).grid(row=3, column=2, padx=5)
        
        ttk.Button(predict_tab, text="Spustit predikci", command=self.predict, width=20).pack(pady=10)
        
        viz_frame = ttk.LabelFrame(visualize_tab, text="Vizualizace průběhu trénování", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        control_frame = ttk.Frame(viz_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Vyberte síť:").pack(side=tk.LEFT, padx=5)
        self.network_selector = ttk.Combobox(control_frame, width=10)
        self.network_selector.pack(side=tk.LEFT, padx=5)
        self.network_selector.bind("<<ComboboxSelected>>", self.update_visualization)
        
        ttk.Button(control_frame, text="Aktualizovat", command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        self.figure = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(log_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.output_text.insert(tk.END, f"Program připraven ({current_time}) - Verze 2.1 s podporou pokračování tréninku\n")
        if torch.cuda.is_available():
            self.output_text.insert(tk.END, f"GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            self.output_text.insert(tk.END, "GPU není dostupné, bude použit pouze CPU.\n")
        
        self.output_text.see(tk.END)
    
    def browse_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV soubory", "*.csv"), ("Všechny soubory", "*.*")])
        if file_path:
            self.input_file.set(file_path)
    
    def browse_output(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV soubory", "*.csv"), ("Všechny soubory", "*.*")])
        if file_path:
            self.output_file.set(file_path)
    
    def browse_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV soubory", "*.csv"), ("Všechny soubory", "*.*")])
        if file_path:
            self.predict_file.set(file_path)
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt"), ("Všechny soubory", "*.*")])
        if file_path:
            self.model_file.delete(0, tk.END)
            self.model_file.insert(0, file_path)
    
    def load_models(self):
        model_files = filedialog.askopenfilenames(filetypes=[("PyTorch model", "*.pt"), ("Všechny soubory", "*.*")])
        if model_files:
            self.loaded_models = list(model_files)
            self.loaded_models_var.set(f"Načteno {len(self.loaded_models)} modelů")
            self.output_text.insert(tk.END, f"Načteno {len(self.loaded_models)} modelů.\n")
            
            # Vypíše načtené modely
            for i, model in enumerate(self.loaded_models):
                self.output_text.insert(tk.END, f"  {i+1}: {os.path.basename(model)}\n")
                
        else:
            self.loaded_models = []
            self.loaded_models_var.set("Žádné modely načteny")
            self.output_text.insert(tk.END, "Žádné modely nebyly načteny.\n")
        self.output_text.see(tk.END)

    def clear_log(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Log vymazán.\n")
        self.output_text.see(tk.END)
    
    def update_log_from_thread(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
    
    def update_plot_from_thread(self, network_id, plot_data):
        if network_id == self.network_selector.get():
            self.figure.clear()
            
            ax1 = self.figure.add_subplot(311)
            ax2 = self.figure.add_subplot(312)
            ax3 = self.figure.add_subplot(313)
            
            epochs = plot_data['epochs']
            ax1.plot(epochs, plot_data['train_losses'], label='Train Loss')
            ax1.plot(epochs, plot_data['val_losses'], label='Val Loss')
            ax1.set_ylabel('Loss (MSE)')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            if len(plot_data['val_rmse']) > 0:
                ax2.plot(epochs, plot_data['val_rmse'], label='Val RMSE', color='green')
                ax2.set_ylabel('RMSE')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            if len(plot_data['learning_rates']) > 0:
                ax3.plot(epochs, plot_data['learning_rates'], label='Learning Rate', color='red')
                ax3.set_ylabel('Learning Rate')
                ax3.set_xlabel('Epocha')
                ax3.legend()
                ax3.grid(True, linestyle='--', alpha=0.7)
            
            self.figure.tight_layout()
            self.canvas.draw()

    def add_network_option(self, network_id):
        if not network_id:
            return
        current_values = list(self.network_selector["values"]) if self.network_selector["values"] else []
        if network_id not in current_values:
            current_values.append(network_id)
            self.network_selector["values"] = current_values
            if not self.network_selector.get():
                self.network_selector.set(network_id)
    
    def update_visualization(self, event=None):
        selected_network = self.network_selector.get()
        if selected_network and selected_network in self.training_loggers:
            logger = self.training_loggers[selected_network]
            plot_data = logger.get_plotting_data()
            self.update_plot_from_thread(selected_network, plot_data)
    
    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.output_text.insert(tk.END, "\nZastavování tréninku, čekejte...\n")
            self.output_text.see(tk.END)
            
            self.training_thread.stop_event.set()
            
            self.output_text.insert(tk.END, "Signál pro zastavení odeslán. Trénink se brzy ukončí.\n")
            self.output_text.see(tk.END)
    
    def disable_buttons(self):
        self.train_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
    
    def enable_buttons(self):
        self.train_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
    
    def train_models(self):
        try:
            input_path = self.input_file.get()
            output_path = self.output_file.get()
            if not input_path or not output_path:
                messagebox.showerror("Chyba", "Vyberte vstupní i výstupní CSV soubory!")
                return
            
            self.disable_buttons()
            
            self.training_loggers = {}
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            self.output_text.insert(tk.END, "Začátek nového tréninku\n")
            
            self.output_text.insert(tk.END, "Načítání dat...\n")
            self.output_text.see(tk.END)
            
            try:
                X_df = pd.read_csv(input_path)
                y_df = pd.read_csv(output_path)
                
                self.training_columns = X_df.columns.tolist()
                
                self.output_text.insert(tk.END, "Předzpracování dat...\n")
                
                for col in X_df.columns:
                    if X_df[col].isnull().any() or (X_df[col] == 0).any():
                        non_zero_vals = X_df[col][(X_df[col] != 0) & ~X_df[col].isnull()]
                        if len(non_zero_vals) > 0:
                            median_val = non_zero_vals.median()
                            X_df[col] = X_df[col].replace(0, median_val)
                            X_df[col] = X_df[col].fillna(median_val)
                
                if X_df.shape[0] != y_df.shape[0]:
                    messagebox.showerror("Chyba", "Počet řádků vstupních a výstupních dat se neshoduje!")
                    self.enable_buttons()
                    return
                
                X = X_df.values
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                scaler_file = f"scaler_{timestamp}.pkl"
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)

                seed = int(self.random_seed.get())
                epochs = int(self.epochs.get())
                batch_size = int(self.batch_size.get())
                dropout_rate = float(self.dropout_rate.get())
                use_augmentation = self.use_augmentation.get()
                noise_level = float(self.noise_level.get())
                mixup_prob = float(self.mixup_prob.get())
                patience = int(self.patience.get())
                scheduler_type = self.scheduler_type.get()
                min_lr = self.min_lr.get()
                use_gpu = torch.cuda.is_available() and self.use_gpu.get()
                use_mixed_precision = use_gpu and self.use_mixed_precision.get()
                use_sigmoid = self.use_sigmoid.get()
                weight_decay = float(self.weight_decay.get())
                learning_rate = float(self.learning_rate.get())

                column_mapping = map_columns_to_horizons(y_df.columns)
                target_tasks = []

                def build_rolling_folds(indices, num_splits=10):
                    segments = np.array_split(indices, num_splits)
                    folds = []
                    for fold_idx in range(num_splits - 1):
                        train_idx = np.concatenate(segments[:fold_idx + 1]) if segments[:fold_idx + 1] else np.array([], dtype=int)
                        val_idx = segments[fold_idx + 1]
                        if len(train_idx) == 0 or len(val_idx) == 0:
                            continue
                        meta_idx = segments[fold_idx + 2] if (fold_idx + 2) < num_splits else np.array([], dtype=int)
                        folds.append((fold_idx + 1, train_idx, val_idx, meta_idx))
                    return folds

                for horizon in HORIZON_ORDER:
                    column = column_mapping.get(horizon)
                    if column is None:
                        continue
                    series = y_df[column]
                    mask = series.notna()
                    if mask.sum() < 2:
                        continue

                    horizon_indices = np.nonzero(mask.values)[0]
                    y_values = series.values.reshape(-1, 1)
                    folds = build_rolling_folds(horizon_indices, num_splits=10)
                    if not folds:
                        self.output_text.insert(tk.END, f"Varování: Pro {horizon} není možné vytvořit rolling split.\n")
                        continue

                    fold_tasks = []
                    for fold_id, train_idx, val_idx, meta_idx in folds:
                        meta_train_idx = np.array([], dtype=int)
                        meta_val_idx = np.array([], dtype=int)
                        if len(meta_idx) >= 2:
                            split_point = int(len(meta_idx) * 0.8)
                            split_point = max(1, min(split_point, len(meta_idx) - 1))
                            meta_train_idx = meta_idx[:split_point]
                            meta_val_idx = meta_idx[split_point:]

                        X_train = X_scaled[train_idx]
                        X_val = X_scaled[val_idx]
                        y_train = y_values[train_idx]
                        y_val = y_values[val_idx]

                        meta_train = X_scaled[meta_train_idx] if len(meta_train_idx) > 0 else np.empty((0, X_scaled.shape[1]))
                        meta_val = X_scaled[meta_val_idx] if len(meta_val_idx) > 0 else np.empty((0, X_scaled.shape[1]))
                        meta_train_targets = y_values[meta_train_idx] if len(meta_train_idx) > 0 else np.empty((0, 1))
                        meta_val_targets = y_values[meta_val_idx] if len(meta_val_idx) > 0 else np.empty((0, 1))

                        fold_tasks.append(
                            {
                                'fold_id': fold_id,
                                'X_train': torch.FloatTensor(X_train),
                                'y_train': torch.FloatTensor(y_train),
                                'X_val': torch.FloatTensor(X_val),
                                'y_val': torch.FloatTensor(y_val),
                                'meta_train_X': torch.FloatTensor(meta_train),
                                'meta_val_X': torch.FloatTensor(meta_val),
                                'meta_train_y': torch.FloatTensor(meta_train_targets),
                                'meta_val_y': torch.FloatTensor(meta_val_targets),
                            }
                        )

                    self.output_text.insert(tk.END, f"{horizon}: rolling split vytvořen s {len(fold_tasks)} foldy.\n")
                    target_tasks.append(
                        {
                            'horizon': horizon,
                            'folds': fold_tasks,
                        }
                    )

                if not target_tasks:
                    messagebox.showerror(
                        "Chyba",
                        "Výstupní CSV neobsahuje žádné použitelné cíle (očekávám sloupce target_1w/4w/13w/26w/52w).",
                    )
                    self.enable_buttons()
                    return

                self.output_text.insert(
                    tk.END,
                    f"K tréninku připraveno {len(target_tasks)} horizontů: {', '.join(task['horizon'] for task in target_tasks)}\n",
                )

                network_ids = [task['horizon'] for task in target_tasks]
                self.network_selector['values'] = network_ids
                if network_ids:
                    self.network_selector.current(0)

                params = {
                    'seed': seed,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate,
                    'use_augmentation': use_augmentation,
                    'noise_level': noise_level,
                    'mixup_prob': mixup_prob,
                    'patience': patience,
                    'scheduler_type': scheduler_type,
                    'min_lr': min_lr,
                    'use_gpu': use_gpu,
                    'use_mixed_precision': use_mixed_precision,
                    'use_sigmoid': use_sigmoid,
                    'weight_decay': weight_decay,
                    'learning_rate': learning_rate,
                    'timestamp': timestamp,
                    'target_tasks': target_tasks,
                }

                self.training_thread = TrainingThread(self, params)
                self.training_thread.start()
                
                self.output_text.insert(tk.END, f"Trénink spuštěn v odděleném vláknu s ID: {timestamp}\n")
                self.output_text.see(tk.END)
                
            except Exception as e:
                import traceback
                error_msg = f"CHYBA při přípravě tréninku: {str(e)}\n{traceback.format_exc()}"
                self.output_text.insert(tk.END, error_msg)
                self.output_text.see(tk.END)
                self.enable_buttons()
                
        except Exception as e:
            import traceback
            messagebox.showerror("Chyba", f"Během přípravy tréninku nastala chyba: {str(e)}")
            self.output_text.insert(tk.END, f"\nCHYBA: {str(e)}\n{traceback.format_exc()}")
            self.output_text.see(tk.END)
            self.enable_buttons()
    
    def predict(self):
        try:
            use_gpu = torch.cuda.is_available() and self.use_gpu.get()
            use_mixed_precision = use_gpu and self.use_mixed_precision.get()
            device = torch.device("cuda:0" if use_gpu else "cpu")
            
            self.output_text.insert(tk.END, f"\nPredikce na: {device}\n")
            self.output_text.see(tk.END)
            
            if use_gpu:
                self.output_text.insert(tk.END, f"GPU: {torch.cuda.get_device_name(0)}\n")
                torch.cuda.synchronize()
            
            predict_path = self.predict_file.get()
            if not predict_path:
                messagebox.showerror("Chyba", "Vyberte CSV soubor pro predikci!")
                return
            
            specific_model_path = self.model_file.get()
            
            if self.loaded_models:
                models_to_use = self.loaded_models
                if len(models_to_use) > 1:
                    self.output_text.insert(tk.END, "Načteno více modelů, pro simulace bude použit první.\n")
                else:
                    self.output_text.insert(tk.END, "Používám načtený model.\n")
            elif specific_model_path and os.path.exists(specific_model_path):
                models_to_use = [specific_model_path]
                self.output_text.insert(tk.END, f"Používám konkrétní model: {specific_model_path}\n")
            else:
                messagebox.showerror("Chyba", "Nejsou načteny žádné modely ani vybrán konkrétní model!")
                return

            scaler_files = [f for f in os.listdir('.') if f.startswith('scaler_') and f.endswith('.pkl')]
            if scaler_files:
                scaler_file = scaler_files[-1]
                self.output_text.insert(tk.END, f"Používám scaler: {scaler_file}\n")
            else:
                messagebox.showerror("Chyba", "Nenalezen žádný scaler. Spusťte nejprve trénování!")
                return

            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            
            try:
                self.output_text.insert(tk.END, "Načítání dat pro predikci...\n")

                X_predict = pd.read_csv(predict_path, header=None)

                expected_cols = getattr(scaler, "n_features_in_", None) or (
                    len(self.training_columns) if self.training_columns else 108
                )
                if X_predict.shape[1] != expected_cols:
                    messagebox.showerror("Chyba", f"Vstupní data pro predikci musí mít {expected_cols} sloupců (má {X_predict.shape[1]})!")
                    return
                
                if self.training_columns:
                    X_predict.columns = self.training_columns
                else:
                    X_predict.columns = [f'col_{i}' for i in range(X_predict.shape[1])]
                
                self.output_text.insert(tk.END, f"Načteno {X_predict.shape[0]} řádků pro predikci.\n")
                
                for col in X_predict.columns:
                    if X_predict[col].isnull().any() or (X_predict[col] == 0).any():
                        non_zero_vals = X_predict[col][(X_predict[col] != 0) & ~X_predict[col].isnull()]
                        if len(non_zero_vals) > 0:
                            median_val = non_zero_vals.median()
                            X_predict[col] = X_predict[col].replace(0, median_val)
                            X_predict[col] = X_predict[col].fillna(median_val)
                
                X_predict = X_predict.fillna(X_predict.median())
                
                X_predict_values = X_predict.values
                X_predict_scaled = scaler.transform(X_predict_values)
                
                X_predict_tensor = torch.FloatTensor(X_predict_scaled)
                
                batch_size = int(self.pred_batch_size.get())
                
                self.output_text.insert(tk.END, "Provádění predikcí...\n")

                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                horizon_models = {}
                meta_model_states = {}

                def load_bundle(path: str):
                    try:
                        obj = torch.load(path, map_location='cpu')
                    except Exception:
                        return None
                    if isinstance(obj, dict) and obj.get('bundle_type') == 'training_bundle':
                        return obj
                    return None

                bundle_timestamp = None
                effective_use_sigmoid = self.use_sigmoid.get()
                bundle = None
                if len(models_to_use) == 1:
                    bundle = load_bundle(models_to_use[0])

                if bundle:
                    bundle_timestamp = bundle.get('timestamp') or infer_timestamp_from_model(models_to_use[0])
                    effective_use_sigmoid = bundle.get('use_sigmoid', effective_use_sigmoid)
                    for horizon, payload in bundle.get('models', {}).items():
                        horizon_models[horizon] = payload.get('model_state') or payload
                        if payload.get('meta_state') is not None:
                            meta_model_states[horizon] = payload['meta_state']
                    self.output_text.insert(tk.END, f"Načten bundle modelů s {len(horizon_models)} sítěmi.\n")
                else:
                    for path in models_to_use:
                        horizon = infer_horizon_from_filename(path)
                        if horizon:
                            horizon_models[horizon] = path
                    effective_use_sigmoid = self.use_sigmoid.get()

                num_rows = X_predict_tensor.shape[0]

                self.output_text.insert(
                    tk.END,
                    "Dropout během predikce je vypnutý, používám deterministické výstupy a metasíť pro odhad chyby.\n",
                )

                def predict_with_model(model_path, output_dim: int) -> np.ndarray:
                    state_dict = torch.load(model_path, map_location='cpu') if isinstance(model_path, str) else model_path
                    inferred_dim = infer_output_dim_from_state_dict(state_dict)
                    if inferred_dim != output_dim:
                        output_dim = inferred_dim

                    model = EfficientResNet(
                        X_predict_scaled.shape[1],
                        hidden_dims=scaled_hidden_dims(X_predict_scaled.shape[1]),
                        use_sigmoid=effective_use_sigmoid,
                        output_dim=output_dim
                    )

                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()

                    pass_predictions = []
                    with torch.no_grad():
                        for start in range(0, num_rows, batch_size):
                            end = min(start + batch_size, num_rows)
                            batch_data = X_predict_tensor[start:end].to(device, non_blocking=True)

                            if use_mixed_precision:
                                with torch.cuda.amp.autocast():
                                    batch_predictions = model(batch_data)
                            else:
                                batch_predictions = model(batch_data)

                            pass_predictions.append(batch_predictions.cpu())

                    stacked = torch.cat(pass_predictions)
                    return inverse_target_transform(stacked).cpu().numpy()

                def find_meta_model_path(base_model_path, horizon: str | None, timestamp_hint: str | None = None):
                    timestamp = infer_timestamp_from_model(base_model_path) if isinstance(base_model_path, str) else timestamp_hint
                    if horizon is None or timestamp is None:
                        return None

                    candidate = f"meta_resnet_{horizon}_{timestamp}.pt"
                    base_dir = (os.path.dirname(base_model_path) or ".") if isinstance(base_model_path, str) else "."
                    full_candidate = os.path.join(base_dir, candidate)

                    if os.path.exists(full_candidate):
                        return full_candidate
                    if os.path.exists(candidate):
                        return candidate
                    return None

                def predict_meta_errors(meta_model_path, base_predictions: np.ndarray) -> np.ndarray:
                    state_dict = torch.load(meta_model_path, map_location='cpu') if isinstance(meta_model_path, str) else meta_model_path
                    meta_model = ErrorEstimator(
                        X_predict_scaled.shape[1] + 1,
                        hidden_dims=scaled_hidden_dims(X_predict_scaled.shape[1] + 1),
                        dropout_rate=float(self.dropout_rate.get()) * 0.5,
                    )
                    meta_model.load_state_dict(state_dict)
                    meta_model.to(device)
                    meta_model.eval()

                    meta_inputs = torch.cat(
                        [
                            X_predict_tensor,
                            torch.as_tensor(base_predictions, dtype=torch.float32).reshape(-1, 1),
                        ],
                        dim=1,
                    )

                    meta_preds = []
                    with torch.no_grad():
                        for start in range(0, num_rows, batch_size):
                            end = min(start + batch_size, num_rows)
                            batch_data = meta_inputs[start:end].to(device, non_blocking=True)

                            if use_mixed_precision:
                                with torch.cuda.amp.autocast():
                                    batch_predictions = meta_model(batch_data)
                            else:
                                batch_predictions = meta_model(batch_data)

                            meta_preds.append(batch_predictions.cpu())

                    return torch.cat(meta_preds).numpy().reshape(-1)

                results = {}

                if horizon_models:
                    used_horizons = [h for h in HORIZON_ORDER if h in horizon_models]
                    for horizon in used_horizons:
                        mean_preds = predict_with_model(horizon_models[horizon], 1).reshape(-1)
                        results[f"mean_{horizon}"] = mean_preds

                        meta_state = meta_model_states.get(horizon)
                        meta_model_path = find_meta_model_path(horizon_models[horizon], horizon, bundle_timestamp) if meta_state is None else None
                        if meta_state is not None:
                            results[f"error_{horizon}"] = predict_meta_errors(meta_state, mean_preds)
                        elif meta_model_path:
                            results[f"error_{horizon}"] = predict_meta_errors(meta_model_path, mean_preds)
                        else:
                            self.output_text.insert(
                                tk.END,
                                f"Nebyl nalezen meta-model pro horizont {horizon}, odhady chyby budou NaN.\n",
                            )
                            results[f"error_{horizon}"] = np.full_like(mean_preds, np.nan, dtype=float)
                else:
                    fallback_model = models_to_use[0]
                    mean_preds = predict_with_model(fallback_model, 1)
                    output_dim = mean_preds.shape[1] if mean_preds.ndim == 2 else 1
                    horizons = HORIZON_ORDER[:output_dim]

                    if mean_preds.ndim == 1:
                        mean_preds = mean_preds.reshape(-1, 1)

                    for j, horizon in enumerate(horizons):
                        horizon_preds = mean_preds[:, j]
                        results[f"mean_{horizon}"] = horizon_preds

                        meta_state = meta_model_states.get(horizon)
                        base_meta_path = find_meta_model_path(fallback_model, horizon, bundle_timestamp) if meta_state is None else None
                        if meta_state is not None:
                            results[f"error_{horizon}"] = predict_meta_errors(meta_state, horizon_preds)
                        elif base_meta_path:
                            results[f"error_{horizon}"] = predict_meta_errors(base_meta_path, horizon_preds)
                        else:
                            self.output_text.insert(
                                tk.END,
                                f"Nebyl nalezen meta-model pro horizont {horizon}, odhady chyby budou NaN.\n",
                            )
                            results[f"error_{horizon}"] = np.full_like(horizon_preds, np.nan, dtype=float)

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                output_file = f"predictions_{timestamp}.csv"
                output_df = pd.DataFrame(results)
                output_df.to_csv(output_file, index=False)
                
                self.output_text.insert(tk.END, f"\nPredikce dokončeny. Výsledky uloženy do '{output_file}'\n")
                self.output_text.insert(tk.END, f"Počet řádků: {output_df.shape[0]}, počet sloupců: {output_df.shape[1]}\n")
                
                abs_path = os.path.abspath(output_file)
                self.output_text.insert(tk.END, f"Plná cesta k souboru: {abs_path}\n")
                self.output_text.see(tk.END)
                
                messagebox.showinfo("Predikce dokončena", f"Predikce byly úspěšně uloženy do souboru:\n{output_file}")
                
            except Exception as e:
                import traceback
                messagebox.showerror("Chyba", f"Při predikci nastala chyba: {str(e)}")
                self.output_text.insert(tk.END, f"\nCHYBA: {str(e)}\n{traceback.format_exc()}\n")
                self.output_text.see(tk.END)
            
        except Exception as e:
            import traceback
            messagebox.showerror("Chyba", f"Při predikci nastala chyba: {str(e)}")
            self.output_text.insert(tk.END, f"\nCHYBA: {str(e)}\n{traceback.format_exc()}\n")
            self.output_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetApp(root)
    root.mainloop()
