import os
import sys

import torchinfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from ADE20KModel import ADE20KModel
from IRModel import IRModel
from thop import profile
from FasterViT.fastervit.models.faster_vit_any_res import faster_vit_1_any_res,faster_vit_2_any_res,faster_vit_4_21k_512_any_res

class SimpleFusionModel(nn.Module):    
    def __init__(
        self,
        rgb_model_path: str,
        ir_model_path: str,
        num_classes: int = 9,
        fusion_strategy: str = "middle_attention",
        freeze_backbones: bool = False,
        distill_layers_enabled: bool = True,
        fuse_type: str = "scalar", # "scalar", "channel", "pixel"
        distill_type: str = "mul", # "sum", "mul"
        context_dim: int = [192,384,768,768],
        input_resolution: Tuple[int, int] = (480, 640),
        rgb_backbone_resolution: Tuple[int, int] = (480, 640),  # For RGB backbone
        ir_backbone_resolution: Tuple[int, int] = (480, 640),   # For IR backbone
        output_resolution: Tuple[int, int] = (480, 640)  # Final output resolution
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.rgb_backbone_resolution = rgb_backbone_resolution
        self.ir_backbone_resolution = ir_backbone_resolution
        self.output_resolution = output_resolution
        self.num_classes = num_classes
        self.context_dims = context_dim
        self.distill_layers_enabled = distill_layers_enabled
        self.fuse_type = fuse_type
        self.distill_type = distill_type
        
        print(f"Fusion Model Configuration:")
        print(f"  Input Resolution: {input_resolution} (HxW)")
        print(f"  RGB Backbone Resolution: {rgb_backbone_resolution}")
        print(f"  IR Backbone Resolution: {ir_backbone_resolution}")
        print(f"  Output Resolution: {output_resolution} (HxW)")
        print(f"  Number of Classes: {num_classes}")
        
        # Load backbones
        self.rgb_backbone = self._load_rgb_backbone(rgb_model_path)
        self.ir_backbone = self._load_ir_backbone(ir_model_path)
        
        # Freeze or partial unfreeze
        if freeze_backbones:
            self._freeze_backbone(self.rgb_backbone)
            self._freeze_backbone(self.ir_backbone)
            print("Backbones frozen")
        else:
            # Freeze everything first
            self._freeze_backbone(self.rgb_backbone)
            self._freeze_backbone(self.ir_backbone)

            # Open early layers
            EARLY_PREFIXES = ['levels.1', 'levels.2']
            rgb_open = " "
            ir_open = " "
            rgb_open = self.set_trainable_by_prefix(self.rgb_backbone, EARLY_PREFIXES)
            ir_open = self.set_trainable_by_prefix(self.ir_backbone, EARLY_PREFIXES)

            self.print_trainable_stats(self.rgb_backbone, "RGB Backbone")
            self.print_trainable_stats(self.ir_backbone, "IR Backbone")

            print(f"Backbones partially unfrozen: RGB ({rgb_open} layers), IR ({ir_open} layers)")    

        self.feature_dims = [392, 784, 1568, 1568]
        
        # Initialize fusion modules
        self._init_fusion_modules(self.context_dims, fusion_strategy)

        if distill_layers_enabled:
            print("Cross-distillation enabled")
            self.distill_layers = [-1,-2] 
            self.cross_distill_blocks = nn.ModuleList([
                CrossDistillBlock(self.feature_dims[i],self.distill_type) for i in self.distill_layers
            ])

    def _load_rgb_backbone(self, model_path):
        wrapper = ADE20KModel()
        backbone = wrapper.backbone 

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            state_dict = checkpoint.get('state_dict', checkpoint)
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded RGB backbone from {model_path}")
            if missing:
                print(f"  ⚠️ Missing keys: {len(missing)}")
            if unexpected:
                print(f"  ⚠️ Unexpected keys: {len(unexpected)}")

        return backbone
    
    def _load_ir_backbone(self, model_path):
        wrapper = IRModel()
        backbone = wrapper.backbone

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded IR backbone from {model_path}")

        return backbone
    
    def _freeze_backbone(self, backbone):
        for param in backbone.parameters():
            param.requires_grad = False

    def set_trainable_by_prefix(self,module: nn.Module, prefixes):
        hit = 0
        for name, p in module.named_parameters():
            if any(name.startswith(pref) for pref in prefixes):
                p.requires_grad = True
                hit += 1
        # Train mode for matched modules
        for n, m in module.named_modules():
            if any(n.startswith(pref) for pref in prefixes):
                m.train()
        return hit

    def print_trainable_stats(self,module: nn.Module, title="module"):
        tot = sum(p.numel() for p in module.parameters())
        trn = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[{title}] trainable params: {trn:,} / {tot:,} ({100*trn/max(1,tot):.2f}%)")

    def _init_fusion_modules(self, context_dims, fusion_strategy):
        """Initialize fusion modules with stable configuration"""
        
        # Adapters with careful initialization
        self.rgb_adapters = nn.ModuleList()
        self.ir_adapters = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        
        for i, dim in enumerate(self.feature_dims):
            out_channels = context_dims[i]

            rgb_adapter = nn.Sequential(
                nn.Conv2d(dim, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            nn.init.normal_(rgb_adapter[0].weight, mean=0, std=0.02)
            nn.init.zeros_(rgb_adapter[0].bias)
            self.rgb_adapters.append(rgb_adapter)
            
            # IR adapter
            ir_adapter = nn.Sequential(
                        nn.Conv2d(dim, out_channels, 1, bias=True),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                        
            nn.init.normal_(ir_adapter[0].weight, mean=0, std=0.02)
            nn.init.zeros_(ir_adapter[0].bias)
            self.ir_adapters.append(ir_adapter)
        
            # Fusion module
            if fusion_strategy == 'simple':
                self.fusion_modules.append(BetterFusion(out_channels))
            elif fusion_strategy == 'average_summation':
                self.fusion_modules.append(AverageSumFusion(out_channels, fuse_type=self.fuse_type))
            else:
                self.fusion_modules.append(CrossModalFusion(out_channels, fusion_type=fusion_strategy))
        
        self.decoder = NativeResolutionDecoder(
            in_channels=context_dims,
            output_resolution=self.output_resolution,
            num_classes=self.num_classes
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    def extract_features(self, x, backbone):
        """Extract multi-scale features from backbone"""
        # Prefer a wrapper-provided extractor if available (ADE20KModel / IRModel implement this)
        if hasattr(backbone, 'extract_multi_scale_features'):
            return backbone.extract_multi_scale_features(x)

        # Otherwise try to use the inner raw backbone (FasterViT-like object)
        b = backbone.backbone if hasattr(backbone, 'backbone') else backbone

        features = []
        if hasattr(b, 'patch_embed') and hasattr(b, 'levels'):
            x = b.patch_embed(x)
            for level in b.levels:
                x = level(x)
                features.append(x)
            return features

        raise RuntimeError('Provided backbone does not implement extract_multi_scale_features and is not a recognized FasterViT-like object')
        
    def forward(self, rgb, ir):
        # Extract features (with gradients now)
        rgb_features = self.extract_features(rgb, self.rgb_backbone)
        ir_features = self.extract_features(ir, self.ir_backbone)

        if self.distill_layers_enabled:
            # Apply cross-distillation to selected deep layers
            for j, layer_idx in enumerate(self.distill_layers):
                rgb_features[layer_idx], ir_features[layer_idx] = self.cross_distill_blocks[j](rgb_features[layer_idx], ir_features[layer_idx])

        # Process all levels
        fused_features = []
        for i in range(len(rgb_features)):
            # Adapt
            rgb_feat = self.rgb_adapters[i](rgb_features[i])
            ir_feat = self.ir_adapters[i](ir_features[i])      
            
            # Fuse
            fused = self.fusion_modules[i](rgb_feat, ir_feat)
            fused_features.append(fused)
        
        # Decode
        main_logits, aux_logits = self.decoder(fused_features)
    
        return main_logits, aux_logits
    
class BetterFusion(nn.Module):
    """Improved fusion without attention complexity"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Careful initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
    
    def forward(self, rgb_feat, ir_feat):
        # Ensure same size
        if rgb_feat.shape[2:] != ir_feat.shape[2:]:
            ir_feat = F.interpolate(ir_feat, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        concat = torch.cat([rgb_feat, ir_feat], dim=1)
        fused = self.conv1(concat)
        fused = self.bn1(fused)
        fused = self.relu(fused)
        
        return fused

class CrossDistillBlock(nn.Module):
    """
    Lightweight cross-feature distillation (inspired by PADNet & MTI-Net)
    Each modality guides the other through learned projections and shared fusion.
    """
    def __init__(self, in_channels, distill_type='mul'):
        super().__init__()
        self.distill_type = distill_type
        mid = in_channels // 2

        # Project each modality to mid-dim
        self.rgb_to_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.ir_to_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # Fuse both into shared representation
        self.shared_fusion = nn.Sequential(
            nn.Conv2d(mid * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1), #eklendi
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, ir_feat):
        # Match spatial size
        if rgb_feat.shape[2:] != ir_feat.shape[2:]:
            ir_feat = F.interpolate(ir_feat, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)

        rgb_proj = self.rgb_to_shared(rgb_feat)
        ir_proj  = self.ir_to_shared(ir_feat)
        shared_gate = self.shared_fusion(torch.cat([rgb_proj, ir_proj], dim=1))

        if self.distill_type == 'sum':
            rgb_refined = rgb_feat + shared_gate
            ir_refined  = ir_feat  + shared_gate
        elif self.distill_type == 'mul':
            rgb_refined = rgb_feat + shared_gate * rgb_feat
            ir_refined  = ir_feat + (1 - shared_gate) * ir_feat

        return rgb_refined, ir_refined

class AverageSumFusion(nn.Module):
    def __init__(self, channels: int = 256, fuse_type = "scalar"):
        super().__init__()
        self.fuse_type = fuse_type
        if fuse_type=="scalar":
            self.weight_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),             # [B, 2C, 1, 1]
                nn.Conv2d(channels * 2, 1, kernel_size=1),  # [B, 1, 1, 1]
                nn.Sigmoid()                         # [B, 1, 1, 1] ∈ [0,1]
            )
        elif fuse_type=="channel":
            self.weight_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channels, 1),  # each channel gets its own weight
                nn.Sigmoid()
            )   
        elif fuse_type=="pixel":  
            self.weight_predictor = nn.Sequential(
                nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, 1),
                nn.Sigmoid()
            )  

    """Fusion by averaging and summing features"""
    def forward(self, rgb_feat, ir_feat):
        if rgb_feat.shape != ir_feat.shape:
            ir_feat = F.interpolate(ir_feat, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)

        # B, C, H, W → concat → [B, 2C, H, W]
        fused_input = torch.cat([rgb_feat, ir_feat], dim=1)
        weight = self.weight_predictor(fused_input)  # [B,1,1,1]

        # Fusion: w * RGB + (1 - w) * IR
        fused = weight * rgb_feat + (1 - weight) * ir_feat

        if torch.isnan(fused).any():
            print("NaN detected in AverageSumFusion output!")
            fused = torch.nan_to_num(fused, 0.0)

        return fused


class CrossModalFusion(nn.Module):
    """Fuse RGB and IR features with attention mechanism"""
    
    def __init__(self, channels, fusion_type='middle_attention'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Channel attention
        reduction = max(8, channels // 16)  # Minimum 8
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),  # Düzeltildi
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, ir_feat):
        eps = 1e-8
        
        # Channel attention
        concat = torch.cat([rgb_feat, ir_feat], dim=1)
        channel_att = self.channel_gate(concat)
        channel_att = torch.clamp(channel_att, min=eps, max=1.0-eps)
        
        rgb_ch_att, ir_ch_att = channel_att.chunk(2, dim=1)
        
        rgb_refined = rgb_feat * (1 + rgb_ch_att)  # Residual connection
        ir_refined = ir_feat * (1 + ir_ch_att)
        
        # Spatial attention
        concat_refined = torch.cat([rgb_refined, ir_refined], dim=1)
        spatial_max, _ = torch.max(concat_refined, dim=1, keepdim=True)
        spatial_mean = torch.mean(concat_refined, dim=1, keepdim=True)
        
        # Prevent extreme values
        spatial_max = torch.clamp(spatial_max, min=-10, max=10)
        spatial_mean = torch.clamp(spatial_mean, min=-10, max=10)
        
        spatial_concat = torch.cat([spatial_max, spatial_mean], dim=1)
        spatial_att = self.spatial_gate(spatial_concat)
        spatial_att = torch.clamp(spatial_att, min=eps, max=1.0-eps)
        
        rgb_final = rgb_refined * (1 + spatial_att)  # Residual
        ir_final = ir_refined * (1 + spatial_att)
        
        # Final fusion
        fused = self.fusion_conv(torch.cat([rgb_final, ir_final], dim=1))
        
        # NaN check
        if torch.isnan(fused).any():
            print("NaN detected in fusion output!")
            fused = torch.nan_to_num(fused, 0.0)
        
        return fused

class NativeResolutionDecoder(nn.Module):
    def __init__(self,
                 in_channels: List[int],     
                 num_classes: int,
                 output_resolution: Tuple[int,int] = (480, 640)):
        super().__init__()
        assert len(in_channels) >= 2, "At least two feature levels are required"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_resolution = output_resolution

        L = len(in_channels)

        # Top-down projections
        # Ex: 512 -> 256, 256 -> 256, 256 -> 128
        self.td_projs = nn.ModuleList([
            nn.Conv2d(in_channels[i+1], in_channels[i], kernel_size=1, bias=False)
            for i in range(L - 1)
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(L - 1)
        ])

        # heads: main ve auxiliary
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0] // 2, num_classes, kernel_size=1)
        )

        mid = L // 2
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels[mid], in_channels[mid] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[mid] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[mid] // 2, num_classes, kernel_size=1)
        )

    def forward(self, features: List[torch.Tensor]):
        L = len(features)
        assert L == len(self.in_channels)

        fpn_features = []

        prev = features[-1]
        fpn_features.insert(0, prev)

        for i in range(L - 2, -1, -1):
            lateral = features[i] 

            up = F.interpolate(prev, size=lateral.shape[2:], mode='bilinear', align_corners=False)
            td = self.td_projs[i](up) 

            merged = td + lateral  
            refined = self.fpn_convs[i](merged)

            prev = refined
            fpn_features.insert(0, prev)

        main_feat = fpn_features[0]
        main_logits = self.seg_head(main_feat)
        main_logits = F.interpolate(main_logits, size=self.output_resolution, mode='bilinear', align_corners=False)

        mid = L // 2
        aux_feat = fpn_features[mid]
        aux_logits = self.aux_head(aux_feat)
        aux_logits = F.interpolate(aux_logits, size=self.output_resolution, mode='bilinear', align_corners=False)

        return main_logits, aux_logits
def test_native_resolution_model():
    """Test the model with native MFNet resolution"""
    import os
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SimpleFusionModel(
        rgb_model_path=None,  
        ir_model_path=None,
        num_classes=9,
        fusion_strategy='average_summation',
        freeze_backbones=False,
        input_resolution=(480, 640),     
        context_dim=[392, 784, 1568, 1568],
        rgb_backbone_resolution=(480, 640),  # RGB backbone expects
        ir_backbone_resolution=(480, 640),   # IR backbone expects
        output_resolution=(480, 640)      # Output at native resolution
    ).to(device)
    
    print("\n" + "="*50)
    print("Testing Native Resolution Model")
    print("="*50)
    
    print("\nTest 1: Native MFNet Resolution")
    rgb = torch.randn(2, 3, 480, 640).to(device)
    ir = torch.randn(2, 1, 480, 640).to(device)

    
    with torch.no_grad():
        main_out, aux_out = model(rgb, ir)
        print(f"  RGB input: {rgb.shape}")
        print(f"  IR input: {ir.shape}")
        print(f"  Main output: {main_out.shape}")
        print(f"  Aux output: {aux_out.shape}")
        assert main_out.shape == (2, 9, 480, 640), "Output shape mismatch!"
        print("  ✅ Output matches MFNet native resolution!")
    
    # Memory and parameter stats
    print("\n" + "="*50)
    print("Model Statistics")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(rgb, ir))    
    
    print(f"Flops (G): {flops / 1e9:.4f}")
    print(f"Params (M): {params / 1e6:.4f}")

    torchinfo.summary(model) 
    
    return model


if __name__ == '__main__':
    model = test_native_resolution_model()