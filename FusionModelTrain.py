import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
import argparse
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import wandb  # Optional: for experiment tracking

# Import your models
from FusionModel import SimpleFusionModel
from FusionModelDataset import FusionModelDataset
from ADE20KUtils import ComboLoss  
from FusionModelUtils import compute_results
from sklearn.metrics import confusion_matrix

# ---- EMA (Exponential Moving Average) ----
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # unwrap DP
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}
    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state

# ---- Lovasz + Dice + CE (label smoothing) ----
def lovasz_softmax(logits, labels, classes='present', eps=1e-6):
    # logits: [B,C,H,W] (ham), labels: [B,H,W]
    probs = torch.softmax(logits.float(), dim=1)
    B, C, H, W = probs.shape
    losses = []
    for c in range(C):
        if classes == 'present' and (labels == c).sum() == 0:
            continue
        fg = (labels == c).float()
        pc = probs[:, c, ...]
        errors = (fg - pc).abs().reshape(-1)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = fg.reshape(-1)[perm]
        grad = torch.cumsum(gt_sorted, 0) / (gt_sorted.sum() + eps)
        grad = 1.0 - grad
        losses.append(torch.dot(errors_sorted, grad) / (H * W))
    return torch.stack(losses).mean() if len(losses) else probs.new_tensor(0.)

class ComboLoss3(nn.Module):
    def __init__(self, ce_w=0.6, dice_w=0.2, lovasz_w=0.2,
                 class_weights=None, ignore_index=None, label_smoothing=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.ce_w, self.dice_w, self.lovasz_w = ce_w, dice_w, lovasz_w

    def dice(self, logits, y, eps=1e-6):
        logits = logits.float()
        C = logits.shape[1]
        y1h = F.one_hot(y.clamp_min(0), C).permute(0,3,1,2).float()
        p = torch.softmax(logits, dim=1)
        inter = (p * y1h).sum(dim=(0,2,3))
        union = p.sum(dim=(0,2,3)) + y1h.sum(dim=(0,2,3))
        return 1 - ((2*inter + eps) / (union + eps)).mean()

    def forward(self, logits, y):
        # CE + Dice + Lovasz (kaybı fp32'de hesaplayacağız)
        return (self.ce_w * self.ce(logits, y)
              + self.dice_w * self.dice(logits, y)
              + self.lovasz_w * lovasz_softmax(logits, y))
    

# =============================================================================
# Training Components
# =============================================================================

class FusionTrainer:
    """
    Trainer class for RGB-IR Fusion model.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_seed(config.seed)
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.setup_model()
        
        # Setup data
        self.train_loader, self.val_loader = self.setup_data()
        
        # Setup training components
        self.criterion_main, self.criterion_aux = self.setup_loss()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = GradScaler('cuda')
        self.ema = ModelEMA(self.model, decay=self.config.ema_decay) if self.config.ema else None       
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.start_epoch = 1
        self.best_miou = 0.0
        self.best_loss = float('inf')
        
        # Load checkpoint if resuming
        if config.resume == True and config.resume_with_reset == False:
            self.load_checkpoint()

        elif config.resume and config.resume_with_reset == True:
            self.resume_from_best_with_reset()
    
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_directories(self):
        """Create necessary directories"""
        self.exp_dir = Path(self.config.exp_dir) / self.config.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=4)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def setup_model(self):
        """Initialize the fusion model"""
        model = SimpleFusionModel(
            rgb_model_path=self.config.rgb_model_path,
            ir_model_path=self.config.ir_model_path,
            num_classes=self.config.num_classes,
            fusion_strategy=self.config.fusion_strategy,
            freeze_backbones=self.config.freeze_backbones,
            distill_layers_enabled = self.config.distill_layers_enabled,
            fuse_type=self.config.fuse_type,
            distill_type=self.config.distill_type,
            context_dim=self.config.context_dim,
            input_resolution=(480, 640),
            rgb_backbone_resolution=(480,640),
            ir_backbone_resolution=(480, 640),
            output_resolution=(480, 640)
        )
        
        model = model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and self.config.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def setup_data(self):
        """Setup data loaders"""
        # Create datasets
        train_dataset = FusionModelDataset(self.config.data_root, 'train', have_label=True)
        val_dataset  = FusionModelDataset(self.config.data_root, 'test', have_label=True)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Train iterations per epoch: {len(train_loader)}")
        
        return train_loader, val_loader
    
    def setup_loss(self):
        """Setup loss functions"""
        # Main loss
        if self.config.class_weights:
            # Calculate or load class weights
            class_weights = self.calculate_class_weights()
        else:
            class_weights = None
        
        ignore_idx = 0 if self.config.ignore_unlabeled else -100

        if self.config.loss_type == 'combo3':
            criterion_main = ComboLoss3(
                ce_w=self.config.ce_weight,
                dice_w=self.config.dice_weight,
                lovasz_w=self.config.lovasz_weight,
                class_weights=class_weights,
                ignore_index=ignore_idx,
                label_smoothing=self.config.label_smoothing
            ).to(self.device)
        else:
            # Eski ADE20K ComboLoss
            criterion_main = ComboLoss(
                ce_weight=self.config.ce_weight,
                dice_weight=self.config.dice_weight,
                class_weights=class_weights,
                ignore_index=ignore_idx
            ).to(self.device)
        
        # Auxiliary loss
        criterion_aux = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index =ignore_idx
        ).to(self.device)
        
        return criterion_main, criterion_aux
    
    def calculate_class_weights(self):
        #Calculate class weights from training data
        print("Calculating class weights...")
        class_counts = torch.zeros(self.config.num_classes)
        
        for _, _, mask,_ in tqdm(self.train_loader, desc="Computing class weights"):
            # Ensure mask is integer type for counting (CrossEntropy and bincount expect Long)
            if torch.is_tensor(mask):
                m = mask.long()
            else:
                m = torch.tensor(mask, dtype=torch.long)

            for cls in range(self.config.num_classes):
                class_counts[cls] += (m == cls).sum()
        
        # Inverse frequency weighting
        total = class_counts.sum()
        class_weights = total / (self.config.num_classes * class_counts)
        class_weights = class_weights / class_weights.mean()  # Normalize
        
        # Clip extreme values
        class_weights = torch.clamp(class_weights, 0.1, 10.0)
        
        print(f"Class weights: {class_weights.tolist()}")
        class_weights = torch.tensor(class_weights)
        return class_weights.to(self.device)
    
    def setup_optimizer(self):
        """Setup optimizer with different learning rates"""
        # Separate parameter groups
        params = []
        assigned_params = set()  # Track already-assigned parameters
        
        # Get model (handle DataParallel)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Backbone parameters (if not frozen)
        if not self.config.freeze_backbones:
            backbone_params = []
            for name, param in model.named_parameters():
                if 'backbone' in name and param.requires_grad and param not in assigned_params:
                    backbone_params.append(param)
                    assigned_params.add(param)
            
            if backbone_params:
                params.append({
                    'params': backbone_params,
                    'lr': self.config.lr_backbone,
                    'name': 'backbone'
                })
        
        # Fusion module parameters
        fusion_params = []
        for name, param in model.named_parameters():
            if ('fusion' in name or 'aligner' in name or 'attention' in name) and param.requires_grad and param not in assigned_params:
                fusion_params.append(param)
                assigned_params.add(param)
        
        if fusion_params:
            params.append({
                'params': fusion_params,
                'lr': self.config.lr_fusion,
                'name': 'fusion'
            })
        
        # Decoder parameters
        decoder_params = []
        for name, param in model.named_parameters():
            if 'decoder' in name and param.requires_grad and param not in assigned_params:
                decoder_params.append(param)
                assigned_params.add(param)
        
        if decoder_params:
            params.append({
                'params': decoder_params,
                'lr': self.config.lr_decoder,
                'name': 'decoder'
            })

        head_params = []
        for name, param in model.named_parameters():
            if ('head.' in name) and param.requires_grad and param not in assigned_params:
                head_params.append(param); assigned_params.add(param)
        if head_params:
            params.append({'params': head_params, 'lr': self.config.lr_decoder, 'name': 'head'})

        others = [p for p in model.parameters() if p.requires_grad and p not in assigned_params]
        if others:
            params.append({'params': others, 'lr': self.config.lr_decoder, 'name': 'others'})
        # Create optimizer
        if self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def setup_scheduler(self):
        """Setup learning rate scheduler - MORE STABLE"""
        if self.config.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-7)
        elif self.config.scheduler == 'cosine_restart':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=25, T_mult=1, eta_min=1e-7)    
        elif self.config.scheduler == 'poly':
            from torch.optim.lr_scheduler import PolynomialLR
            scheduler = PolynomialLR(self.optimizer, total_iters=self.config.epochs, power=0.9)
        elif self.config.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  
                factor=0.7,
                patience=20,
                min_lr=1e-7,
                verbose=True
            )
        elif self.config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def setup_logging(self):
        """Setup tensorboard and wandb logging"""
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        # Plain text logging
        self.log_path = self.exp_dir / 'log.txt'
        # Open in append mode so resume runs don't overwrite
        self._log_file = open(self.log_path, 'a', buffering=1)
        # Weights & Biases (optional)
        if self.config.use_wandb:
            wandb.init(
                project="rgb-ir-fusion",
                name=self.config.exp_name,
                config=self.config
            )
            wandb.watch(self.model)

        # initial log
        # use helper below (safe because it's a class method)
        # but call after writer/file are set
        try:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._log_file.write(f'[{ts}] Experiment started: {self.exp_dir}\n')
        except Exception:
            pass

    def log(self, msg, print_console=True):
        """Helper to write a timestamped message to log.txt and optionally print."""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}\n'
        try:
            self._log_file.write(line)
        except Exception:
            # fallback: print to console if file write fails
            print('Log write failed; message:', line)
        if print_console:
            print(line, end='')
    
    def get_aux_weight(self, epoch):
        """Calculate auxiliary loss weight based on epoch"""
        if epoch < self.config.aux_weight_decay_epoch:
            alpha = epoch / self.config.aux_weight_decay_epoch
            return (1 - alpha) * self.config.aux_weight_start + alpha * self.config.aux_weight_end
        return self.config.aux_weight_end


    def train_epoch(self, epoch):
        """Train with gradient accumulation for stability"""
        self.model.train()
        accumulation_steps = max(1, int(self.config.grad_accum_steps))
        
        # Metrics tracking
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        num_batches = 0
        
        # Get aux weight
        aux_weight = self.get_aux_weight(epoch)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (rgb, ir, masks,_) in enumerate(pbar):
            # Forward pass
            if masks.max() >= self.config.num_classes:
                print(f"\nWarning: Found invalid labels in batch {batch_idx}")
                print(f"Label range before clamping: min={masks.min()}, max={masks.max()}")
                # Print samples with invalid labels
                invalid_mask = masks >= self.config.num_classes

                max_label = self.config.num_classes - 1
                masks = masks.clamp_(0, max_label)
                print(f"Labels clamped to range [0, {masks-1}]")

            rgb = rgb.to(device=self.device)
            ir = ir.to(device=self.device)
            masks = masks.to(device=self.device)
            
            with autocast('cuda', enabled=True): #with autocast('cuda',dtype=torch.bfloat16):#
                main_out, aux_out = self.model(rgb, ir)
                
                # Calculate losses
                loss_main = self.criterion_main(main_out, masks)
                loss_aux = self.criterion_aux(aux_out, masks) if aux_out is not None else 0
                
                # Combined loss
                loss = loss_main + aux_weight * loss_aux
                loss = loss / accumulation_steps  # Scale loss
            
            # Track metrics
            running_loss += loss.item() * accumulation_steps  # Unscale for tracking
            running_main_loss += loss_main.item()
            if aux_out is not None and loss_aux != 0:
                running_aux_loss += loss_aux.item()
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                
                # Check for NaN gradients
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        param.grad.zero_()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # EMA güncelle
                if self.ema is not None:
                    self.ema.update(self.model)
            
            # Update progress bar
            num_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'main': f'{loss_main.item():.4f}',
                'aux': f'{loss_aux.item() if loss_aux != 0 else 0:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_main_loss = running_main_loss / num_batches
        epoch_aux_loss = running_aux_loss / num_batches
        
        return epoch_loss, epoch_main_loss, epoch_aux_loss
    
    def validate(self, epoch):
        """Optimized validation"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_masks = []
        aux_weight = self.get_aux_weight(epoch)
        conf_total = np.zeros((self.config.num_classes, self.config.num_classes))
        if self.ema is not None and self.config.eval_use_ema:
            self.ema.apply_shadow(self.model)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for rgb, ir, masks,_ in pbar:
                rgb = rgb.to(self.device)
                ir = ir.to(self.device) 
                masks = masks.to(self.device)
                
                with autocast(device_type='cuda', enabled=True):
                    main_out, aux_out = self.model(rgb, ir)
                
                loss_main = self.criterion_main(main_out, masks)
                loss_aux = self.criterion_aux(aux_out, masks) if aux_out is not None else 0
                loss = loss_main + aux_weight * loss_aux
                running_loss += loss.item()
                
                preds = main_out.argmax(dim=1).cpu().numpy()
                labels = masks.cpu().numpy()

                for pred, label in zip(preds, labels):
                    label = label.flatten()
                    pred = pred.flatten()
                    conf = confusion_matrix(y_true=label, y_pred=pred, labels=list(range(self.config.num_classes)))
                    conf_total += conf
                    
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # EMA restore
        if self.ema is not None and self.config.eval_use_ema:
            self.ema.restore(self.model)

        precision_per_class, recall_per_class, iou_per_class, f1score = compute_results(conf_total)
        miou = np.mean(iou_per_class)
        pixel_acc = np.trace(conf_total) / np.sum(conf_total)


        torch.cuda.empty_cache()
        return running_loss / len(self.val_loader), miou, pixel_acc, iou_per_class

    def _unwrap_model(self):
        m = self.model
        return m.module if hasattr(m, "module") else m

    def save_checkpoint(self, epoch, is_best=False):
        model_to_save = self._unwrap_model().state_dict()  
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_to_save,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if getattr(self, "scheduler", None) else None,
            "scaler_state_dict": self.scaler.state_dict() if getattr(self, "scaler", None) else None,
            "ema_state_dict": (self.ema.state_dict() if hasattr(self, "ema") and self.ema is not None else None),
            "best_miou": float(getattr(self, "best_miou", 0.0)),
            "config": vars(self.config) if hasattr(self.config, "__dict__") else self.config,
        }

        latest_path = self.checkpoint_dir / "latest.pth"
        tmp_path = self.checkpoint_dir / "latest.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, latest_path)

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best.pth")
            # --- Save EMA full model separately ---
            if self.ema is not None:
                # 1) Apply EMA weights to model 
                self.ema.apply_shadow(self.model)
                # 2) Get full state dict
                full_state = self._unwrap_model().state_dict()
                # 3) Save
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": full_state,
                    "config": vars(self.config),
                    "best_miou": float(self.best_miou)
                }, self.checkpoint_dir / "best_model.ema.pth")
                print("[INFO] EMA FULL model saved as best_model.ema.pth")
                # 4) Restore original weights
                self.ema.restore(self.model)

        if getattr(self.config, "save_interval", 0) and epoch % self.config.save_interval == 0:
            torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch}.pth")  

    def resume_from_best_with_reset(self):
        """Resume training from best model weights but reset optimizer/scheduler/scaler states."""
        checkpoint_path = self.checkpoint_dir / "best.pth"
        if not checkpoint_path.exists():
            print(f"[WARN] No best checkpoint found at: {checkpoint_path}")
            return
        
        print(f"[INFO] Resuming from best model with LR reset: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        model_state = checkpoint.get("model_state_dict")
        if model_state:
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys: {unexpected_keys}")
            print("[INFO] Model weights loaded.")

            self.ema.load_state_dict(checkpoint['model_state_dict'])
            self.ema.apply_shadow(self.model) 

        # Load EMA if available
        if self.ema and checkpoint.get('ema_state_dict'):
            try:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                self.ema.apply_shadow(self.model)  # Apply EMA weights
                print("[INFO] EMA state loaded.")
            except Exception as e:
                print(f"[WARN] EMA state not loaded: {e}")

        # Reset optimizer & scheduler manually
        for param_group in self.optimizer.param_groups:
            name = param_group.get('name', 'default')
            if name == 'backbone':
                param_group['lr'] = self.config.lr_backbone
            elif name == 'fusion':
                param_group['lr'] = self.config.lr_fusion
            elif name in ['decoder', 'head', 'others']:
                param_group['lr'] = self.config.lr_decoder

        # Reset scheduler
        self.scheduler = self.setup_scheduler()
        self.scaler = GradScaler('cuda')

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint['best_miou']
        
        print(f"Resumed from epoch {self.start_epoch} with best mIoU: {self.best_miou:.4f}")

    def load_checkpoint(self):
        """Load checkpoint for resuming training"""
        checkpoint_path = os.path.join('FusionModel/experiments', self.config.exp_name, 'checkpoints', 'best.pth')

        if not os.path.exists(checkpoint_path):
            print(f"[WARN] No checkpoint found at: {checkpoint_path}")
            return
        
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

        # Always load model weights
        model_state = checkpoint.get('model_state_dict')
        if model_state:
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys in model state: {missing_keys}")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys in model state: {unexpected_keys}")
            print("[INFO] Model weights loaded.")

        # Attempt to load optimizer
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[INFO] Optimizer state loaded.")

        except Exception as e:
            print(f"[WARN] Optimizer state not loaded: {e}")

        # Attempt to load scheduler
        try:
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("[INFO] Scheduler state loaded.")
        except Exception as e:
            print(f"[WARN] Scheduler state not loaded: {e}")
        
        # Attempt to load scaler
        try:
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("[INFO] Scaler state loaded.")
        except Exception as e:
            print(f"[WARN] Scaler state not loaded: {e}")

        # Load EMA if available
        if self.ema and checkpoint.get('ema_state_dict'):
            try:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                print("[INFO] EMA state loaded.")
            except Exception as e:
                print(f"[WARN] EMA state not loaded: {e}")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint['best_miou']
        
        print(f"Resumed from epoch {self.start_epoch} with best mIoU: {self.best_miou:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        for epoch in range(self.start_epoch, self.config.epochs):
            torch.cuda.empty_cache()
            # Training
            train_loss, train_main_loss, train_aux_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, miou, pixel_acc, class_ious = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()
            
            # Check if best model
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, Aux: {train_aux_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"mIoU: {miou:.4f} (Best: {self.best_miou:.4f})")
            print(f"Pixel Acc: {pixel_acc:.4f}")
            
            # Log class-wise IoU
            if self.config.verbose:
                class_names = ['unlabeled', 'car', 'person', 'bike', 'curve', 
                              'car_stop', 'guardrail', 'color_cone', 'bump']
                for i, iou in enumerate(class_ious):
                    if i < len(class_names):
                        print(f"  {class_names[i]}: {iou:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/mIoU', miou, epoch)
            self.writer.add_scalar('Val/PixelAcc', pixel_acc, epoch)
            
            # Log learning rates
            for param_group in self.optimizer.param_groups:
                name = param_group.get('name', 'default')
                self.writer.add_scalar(f'LR/{name}', param_group['lr'], epoch)
            try:
                lrs = ','.join([f"{pg.get('name','default')}:{pg['lr']:.2e}" for pg in self.optimizer.param_groups])
            except Exception:
                lrs = ','.join([f"{pg.get('lr',0):.2e}" for pg in self.optimizer.param_groups])

            self.log(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"TrainLoss={train_loss:.4f} (Main={train_main_loss:.4f}, Aux={train_aux_loss:.4f}) | "
                f"ValLoss={val_loss:.4f} | mIoU={miou:.4f} | PixelAcc={pixel_acc:.4f} | LR={lrs}"
            )
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'miou': miou,
                    'pixel_acc': pixel_acc
                })
            
            print("-" * 50)
        
        print("\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train RGB-IR Fusion Model')
    
    # Model arguments
    parser.add_argument('--rgb_model_path', type=str, default='',
                        help='Path to pretrained RGB model')
    parser.add_argument('--ir_model_path', type=str, default='',
                        help='Path to pretrained IR model')
    parser.add_argument('--fusion_strategy', type=str, default='average_summation',
                        choices=['simple', 'middle_attention','average_summation'],
                        help='Fusion strategy')
    parser.add_argument('--fuse_type', type=str, default='scalar',#pixel',scalar
                        choices=['scalar', 'channel','pixel'])
    parser.add_argument('--distill_type', type=str, default='mul',
                        choices=['mul', 'sum'])
    parser.add_argument('--context_dim', type=int, default=[392, 784, 1568, 1568],
                        choices=['mul', 'sum']),
    parser.add_argument('--freeze_backbones', action='store_true', default = True,
                        help='Freeze pretrained backbones')
    parser.add_argument('--distill_layers_enabled', action='store_true', default = True,
                        help='Freeze pretrained backbones')
    parser.add_argument('--resume', action='store_true',default=False,
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_with_reset', action='store_true',default=False,
                        help='Resume training from checkpoint with reset parameters')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default = '/datavolume/data/emrecanitez/Datasets/MFNet',
                        help='Root directory of MFNet dataset')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of segmentation classes')
    parser.add_argument('--ignore_unlabeled', action='store_true', default=False,
                        help='Ignore unlabeled class in loss')
    parser.add_argument('--class_weights', action='store_true',default=True,
                        help='Use class weights in loss')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lr_backbone', type=float, default=1e-6,
                        help='Learning rate for backbone')
    parser.add_argument('--lr_fusion', type=float, default=6e-5,
                        help='Learning rate for fusion modules')
    parser.add_argument('--lr_decoder', type=float, default=1e-6,
                        help='Learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=0.1,
                        help='Gradient clipping value')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='plateau',# 'cosine', #plateau , cosine_restart
                        choices=['cosine', 'poly', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--t0', type=int, default=15,
                        help='T0 for cosine annealing warm restarts')
    parser.add_argument('--step_size', type=int, default=45,
                        help='Step size for step scheduler')
    
    # Loss arguments
    parser.add_argument('--ce_weight', type=float, default=0.7,
                        help='Cross entropy weight in combo loss')
    parser.add_argument('--dice_weight', type=float, default=0.2,
                        help='Dice weight in combo loss')
    parser.add_argument('--lovasz_weight', type=float, default=0.1,
                        help='Lovasz weight in combo loss')
    parser.add_argument('--aux_weight_start', type=float, default=0.3,
                        help='Initial auxiliary loss weight')
    parser.add_argument('--aux_weight_end', type=float, default=0.02,
                        help='Final auxiliary loss weight')
    parser.add_argument('--aux_weight_decay_epoch', type=int, default=30,
                        help='Epochs to decay auxiliary weight')
    
    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default='Experiment',
                        help='Experiment name')
    parser.add_argument('--exp_dir', type=str, default='datavolume/data/emrecanitez/experiments',
                        help='Experiment directory')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device ID')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    parser.add_argument('--ema', action='store_true', default=True, help='Track EMA of weights.')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay.')
    parser.add_argument('--eval_use_ema', action='store_true', default=True, help='Use EMA weights in validation.')
    parser.add_argument('--eval_tta', action='store_true', default=False, help='Use simple TTA in validation (slow).')
    parser.add_argument('--tta_scales', type=float, nargs='+', default=[1.0], help='TTA scales, e.g., 0.75 1.0 1.25')

    parser.add_argument('--loss_type', type=str, default='combo3', choices=['combo', 'combo3'],
                    help='combo: ADE20K ComboLoss; combo3: CE+Dice+Lovasz (önerilir)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='CE label smoothing.')

    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps.')

    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"fusion_{args.fusion_strategy}_{timestamp}"
    
    # Create trainer and start training
    trainer = FusionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()