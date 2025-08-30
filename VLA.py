# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

# Standard library imports
import os
import glob
import math
import time
import copy
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

# Environment configuration
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
mp.set_start_method('spawn', force=True)

# Third-party imports
import numpy as np
import h5py

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lion_pytorch import Lion

# Additional ML libraries
import kornia.augmentation as K
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch configuration
torch.set_num_threads(16)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import os
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # GPU settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch to device with appropriate dtype conversion."""
    result = {}
    bf16_keys = {'images', 'actions', 'depth', 'contact_forces', 'proprioception'}
    
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype == torch.float32 and key in bf16_keys:
                result[key] = value.to(device, dtype=torch.bfloat16, non_blocking=True)
            else:
                result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def worker_init_fn(worker_id: int):
    """Initialize worker random seeds for DataLoader."""
    initial_seed = torch.initial_seed()
    seed = int(initial_seed % 2**30) + worker_id
    np.random.seed(seed)


def safe_int_convert(value, max_val: int = 1000) -> int:
    """Safely convert various types to int with bounds checking."""
    if torch.is_tensor(value):
        val = int(value.item() if value.numel() == 1 else value.flatten()[0].item())
    elif isinstance(value, np.ndarray):
        val = int(value.item() if value.size == 1 else value.flatten()[0])
    else:
        val = int(value)
    
    return max(1, min(val, max_val))



# =============================================================================
# DATA STRUCTURES AND CONFIGURATIONS
# =============================================================================

@dataclass
class ActionStats:
    """Statistics for action normalization."""
    mean: np.ndarray
    std: np.ndarray
    amin: np.ndarray
    amax: np.ndarray

    @staticmethod
    def zeros(dim: int) -> "ActionStats":
        return ActionStats(
            mean=np.zeros(dim, dtype=np.float32),
            std=np.ones(dim, dtype=np.float32),
            amin=np.full(dim, np.inf, dtype=np.float32),
            amax=np.full(dim, -np.inf, dtype=np.float32),
        )


@dataclass
class ModelConfig:
    """Configuration for the VLA model."""
    action_dim: int
    vision_dim: int = 1536
    language_dim: int = 768
    hidden_dim: int = 2048
    num_mamba_layers: int = 6
    mamba_d_state: int = 128
    mamba_expand: int = 2
    qlora_r: int = 96
    qlora_alpha: int = 192
    use_proprio: bool = False
    chunk_size: int = 4
    use_precision_focus: bool = True


# =============================================================================
# MIXINS AND HELPER CLASSES
# =============================================================================

class NormalizeMixin:
    """Mixin class for action normalization functionality."""
    def set_action_stats(self, stats: ActionStats):
        self.action_stats = stats

    def normalize_action(self, action_tensor: torch.Tensor) -> torch.Tensor:
        # Cache GPU tensors for efficiency
        if action_tensor.is_cuda and not hasattr(self, '_gpu_mean'):
            self._gpu_mean = torch.from_numpy(self.action_stats.mean).to(action_tensor.device, action_tensor.dtype)
            self._gpu_std = torch.from_numpy(self.action_stats.std).to(action_tensor.device, action_tensor.dtype)
            # Expand dimensions to match action tensor
            for _ in range(action_tensor.dim() - self._gpu_mean.dim()):
                self._gpu_mean = self._gpu_mean.unsqueeze(0)
                self._gpu_std = self._gpu_std.unsqueeze(0)
        
        # Use cached GPU tensors or create CPU tensors
        if action_tensor.is_cuda and hasattr(self, '_gpu_mean'):
            mean, std = self._gpu_mean, self._gpu_std
        else:
            mean = torch.from_numpy(self.action_stats.mean).to(action_tensor.dtype)
            std = torch.from_numpy(self.action_stats.std).to(action_tensor.dtype)
            # Expand dimensions to match action tensor
            for _ in range(action_tensor.dim() - mean.dim()):
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
        
        normalized = (action_tensor - mean) / (std + 1e-6)
        return torch.clamp(normalized, -5.0, 5.0)

    def denormalize_action(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(x, torch.Tensor):
            mean = torch.from_numpy(self.action_stats.mean).to(x.device, dtype=x.dtype)
            std = torch.from_numpy(self.action_stats.std).to(x.device, dtype=x.dtype)
            return x * std + mean
        return x * self.action_stats.std + self.action_stats.mean


# =============================================================================
# DATASET IMPLEMENTATION
# =============================================================================

class TriangleDataset(Dataset, NormalizeMixin):
    """Optimized dataset for loading H5 triangle manipulation data."""
    def __init__(self, data_path: str, sequence_length: int = 50, mode: str = "train",
                 train_split: float = 0.9, compute_stats_from: str = "train", stats_max_files: int = 1000,
                 use_augmentation: bool = True):
        assert mode in {"train", "val"}
        self.mode = mode
        self.data_path = data_path
        self.sequence_length = int(sequence_length)
        self.use_augmentation = use_augmentation and mode == "train"
        self.use_contact_data = True  # Enable contact forces for multimodal learning

        # Load and split files
        all_files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
        if not all_files:
            raise FileNotFoundError(f"No .h5 files found under {data_path}")
        
        split_point = int(len(all_files) * train_split)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]
        self.files = train_files if mode == "train" else val_files

        # Compute action statistics
        stats_files = train_files if compute_stats_from == "train" else all_files
        self.action_stats = self._compute_action_stats(stats_files, stats_max_files)
        
        # Setup GPU resources
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._setup_augmentation()
        self._setup_cache()
        
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
        
    def _setup_augmentation(self):
        if not torch.cuda.is_available() or not self.use_augmentation:
            self.augmentation = None
            return
        # Visual augmentation
        self.augmentation = nn.Sequential(
            # Color and intensity
            K.ColorJiggle(brightness=0.8, contrast=0.7, saturation=0.6, hue=0.2, p=0.9).to(dtype=torch.float32),
            K.RandomGamma(gamma=(0.5, 1.5), gain=(0.8, 1.2), p=0.7).to(dtype=torch.float32),
            K.RandomHue(hue=0.15, p=0.8).to(dtype=torch.float32),
            K.RandomBrightness(brightness=0.3, p=0.6).to(dtype=torch.float32),
            K.RandomContrast(contrast=0.4, p=0.6).to(dtype=torch.float32),
            K.RandomSaturation(saturation=0.5, p=0.6).to(dtype=torch.float32),
            
            # Noise and blur
            K.RandomGaussianNoise(mean=0.0, std=0.08, p=0.8).to(dtype=torch.float32),
            K.RandomGaussianBlur(kernel_size=(3, 9), sigma=(0.1, 4.0), p=0.6).to(dtype=torch.float32),
            K.RandomMotionBlur(kernel_size=7, angle=(-60, 60), direction=(-0.9, 0.9), p=0.5).to(dtype=torch.float32),
            K.RandomMedianBlur(kernel_size=(3, 5), p=0.3).to(dtype=torch.float32),
            
            # Geometric transformations
            K.RandomHorizontalFlip(p=0.7).to(dtype=torch.float32),
            K.RandomVerticalFlip(p=0.5).to(dtype=torch.float32),
            K.RandomRotation(degrees=30.0, p=0.6).to(dtype=torch.float32),
            K.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10, p=0.6).to(dtype=torch.float32),
            K.RandomElasticTransform(alpha=(75.0, 75.0), sigma=(7.0, 7.0), p=0.4).to(dtype=torch.float32),
            K.RandomPerspective(distortion_scale=0.3, p=0.4).to(dtype=torch.float32),
            
            # Advanced augmentations
            K.RandomPosterize(bits=2, p=0.4).to(dtype=torch.float32),
            K.RandomSolarize(thresholds=0.4, additions=0.2, p=0.4).to(dtype=torch.float32),
            K.RandomChannelShuffle(p=0.5).to(dtype=torch.float32),
            K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.5).to(dtype=torch.float32),
            K.RandomAutoContrast(p=0.2).to(dtype=torch.float32),
        ).to(self.device)
    
    def _setup_cache(self):
        self.cpu_cache = {}
        self.cache_size = min(20, len(self.files))
        
    def _apply_augmentation(self, img_data: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_augmentation:
            return img_data, actions
        
        T, _, _, _ = img_data.shape
        if T > self.sequence_length:
            max_start = T - self.sequence_length
            start_idx = np.random.randint(0, max_start + 1)
            img_data = img_data[start_idx:start_idx + self.sequence_length]
            actions = actions[start_idx:start_idx + self.sequence_length]
        
        if np.random.rand() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img_data = np.clip(img_data * brightness_factor, 0, 1)
        
        if np.random.rand() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = img_data.mean()
            img_data = np.clip((img_data - mean) * contrast_factor + mean, 0, 1)
        
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, 0.02, img_data.shape).astype(np.float32)
            img_data = np.clip(img_data + noise, 0, 1)
        
        if np.random.rand() < 0.3:
            action_noise = np.random.normal(0, 0.02, actions.shape).astype(np.float32)
            actions = actions + action_noise
        
        return img_data, actions
        
    def _preallocate_buffers(self):
        if torch.cuda.is_available():
            seq_len = self.sequence_length
            self._img_buffer = torch.empty((seq_len, 3, 224, 224), dtype=torch.float32, device=self.device)
            self._action_buffer = torch.empty((seq_len, 4), dtype=torch.float32, device=self.device)
            torch.cuda.empty_cache()
    def _compute_action_stats(self, files: List[str], max_files: int) -> ActionStats:
        all_actions = []
        files_to_process = files[:min(max_files, len(files))]
        
        for file_path in files_to_process:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'action' not in f:
                        continue
                    actions = np.array(f['action'][:], dtype=np.float32)
                    if actions.ndim == 1:
                        actions = actions[None, :]
                    all_actions.append(actions.reshape(-1, actions.shape[-1]))
            except Exception:
                continue
        
        if not all_actions:
            return ActionStats.zeros(dim=7)
        
        # Combine all actions and compute statistics
        combined_actions = np.concatenate(all_actions, axis=0)
        action_mean = combined_actions.mean(axis=0).astype(np.float32)
        action_std = combined_actions.std(axis=0).astype(np.float32) + 1e-8
        action_min = combined_actions.min(axis=0).astype(np.float32)
        action_max = combined_actions.max(axis=0).astype(np.float32)
        
        return ActionStats(mean=action_mean, std=action_std, amin=action_min, amax=action_max)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range for {len(self.files)} files")
        
        if idx in self.cpu_cache:
            return self.cpu_cache[idx]
        
        file_path = self.files[idx]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sample = self._load_sample_optimized(file_path)
        
        if len(self.cpu_cache) < self.cache_size:
            self.cpu_cache[idx] = sample
        
        return sample
    
    def _create_fallback_sample(self, T: int) -> Dict[str, Any]:
        return {
            'images': torch.zeros(T, 3, 128, 128, dtype=torch.float32),
            'depth': None,
            'contact_forces': None,
            'actions': torch.zeros(T, 4, dtype=torch.float32),
            'actions_raw': torch.zeros(T, 4, dtype=torch.float32),
            'proprioception': None,
            'seq_len': min(T, 50),
            'task_description': ""
        }
    
    def _load_sample_optimized(self, file_path: str) -> Dict[str, Any]:
        max_seq_len = self.sequence_length
        
        with h5py.File(file_path, 'r') as f:
            # Load raw data with multimodal support based on Stanford dataset structure  
            images = f['image'][:]
            actions = f['action'][:]
            
            # Load contact forces for enhanced multimodal representation (Stanford ICRA 2019)
            contact_forces = f['contact'][:] if 'contact' in f else None
            
            # Determine actual sequence length
            actual_len = min(len(images), len(actions), max_seq_len)
            images = images[:actual_len]
            actions = actions[:actual_len]
            
            # Convert images to float and normalize if needed
            if images.dtype == np.uint8:
                images = images.astype(np.float32) / 255.0
            else:
                images = images.astype(np.float32)
            
            # Convert to tensors and move to device
            img_tensor = torch.from_numpy(images).to(self.device, dtype=torch.float32)
            action_tensor = torch.from_numpy(actions.astype(np.float32)).to(self.device, dtype=torch.float32)
            
            # Pad sequences if needed
            if actual_len < max_seq_len:
                img_shape = (max_seq_len,) + img_tensor.shape[1:]
                action_shape = (max_seq_len, action_tensor.shape[-1])
                
                padded_images = torch.zeros(img_shape, device=self.device, dtype=torch.float32)
                padded_actions = torch.zeros(action_shape, device=self.device, dtype=torch.float32)
                
                padded_images[:actual_len] = img_tensor
                padded_actions[:actual_len] = action_tensor
                
                img_tensor, action_tensor = padded_images, padded_actions
            
            # Ensure correct image format (batch, channels, height, width)
            if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(0, 3, 1, 2).contiguous()
            
            # Apply ultra-aggressive augmentation with action noise for equal weighting breakthrough
            if self.augmentation is not None and self.use_augmentation:
                img_tensor = self.augmentation(img_tensor)
                # Aggressive action augmentation for robustness
                if torch.rand(1, device=self.device).item() < 0.5:  # 50% chance
                    action_tensor += torch.randn_like(action_tensor) * 0.03  # Tripled noise
                # Additional random scaling for actions
                if torch.rand(1, device=self.device).item() < 0.4:  # 40% chance
                    scale = torch.rand_like(action_tensor) * 0.2 + 0.9  # 90% to 110% scaling
                    action_tensor *= scale
            
            # Convert to bfloat16 for memory efficiency
            img_tensor = img_tensor.to(dtype=torch.bfloat16)
            action_tensor = action_tensor.to(dtype=torch.bfloat16)
            
            # Process contact forces if available for multimodal learning
            contact_tensor = None
            if contact_forces is not None and self.use_contact_data:
                contact_forces = contact_forces[:actual_len]
                if contact_forces.shape[-1] > 4:  # Reshape if needed (50,50) -> (50,4) 
                    contact_forces = contact_forces[:, :4]
                contact_tensor = torch.from_numpy(contact_forces.astype(np.float32)).to(self.device, dtype=torch.bfloat16)
                
                # Pad contact forces if needed
                if actual_len < max_seq_len:
                    padded_contact = torch.zeros((max_seq_len, contact_tensor.shape[-1]), device=self.device, dtype=torch.bfloat16)
                    padded_contact[:actual_len] = contact_tensor
                    contact_tensor = padded_contact
            
            return {
                'images': img_tensor,
                'depth': None,
                'contact_forces': contact_tensor,  # Now includes real contact data
                'actions': action_tensor,
                'actions_raw': action_tensor.clone(),
                'proprioception': None,
                'seq_len': actual_len,
                'task_description': ""
            }


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class MambaBlock(nn.Module):
    """Optimized Mamba2 block for sequence modeling."""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, 
                               groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ssm = self._parallel_ssm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        x_n = self.norm(x)
        x_and_res = self.in_proj(x_n)
        x_proj, res = x_and_res.split(self.d_inner, dim=-1)
        x_proj = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_proj)[..., :T].transpose(1, 2)
        x_conv = self.act(x_conv)
        x_conv = self.dropout(x_conv)
        
        y = self.ssm(x_conv)
        y = y * self.act(res)
        out = self.out_proj(y)
        return out + x

    def _parallel_ssm(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        device, dtype = x.device, x.dtype
        
        # Convert parameters to appropriate device and dtype
        A = -torch.exp(self.A_log.to(dtype).to(device))
        D = self.D.to(dtype).to(device)
        
        # Project input to state space dimensions
        x_proj = self.x_proj(x)
        B_state, C_state = x_proj.split(self.d_state, dim=-1)
        delta = F.softplus(self.dt_proj(x))
        
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.d_state, device=device, dtype=dtype)
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            # Update hidden state
            delta_t = delta[:, t].mean(dim=-1, keepdim=True)
            A_discrete = torch.exp(delta_t * A)
            hidden = hidden * A_discrete + B_state[:, t]
            
            # Compute output
            output_t = torch.sum(hidden * C_state[:, t], dim=-1)
            outputs.append(output_t)
        
        # Combine outputs and apply skip connection
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.unsqueeze(-1).expand(batch_size, seq_len, dim)
        skip_connection = D[:dim].unsqueeze(0).unsqueeze(0) * x
        
        return outputs + skip_connection


class QLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / max(1, r) if r > 0 else 1.0
        
        # Base linear layer
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        
        # Low-rank adaptation layers
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def forward(self, x):
        base_output = self.base_linear(x)
        
        if self.r > 0:
            lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
            return base_output + lora_output
        
        return base_output

    def merge_weights(self):
        """Merge LoRA weights into base layer for inference efficiency."""
        if self.r > 0 and self.lora_A is not None:
            merged_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling
            self.base_linear.weight.data += merged_weight
            self.lora_A = self.lora_B = None




class Tokenizer:
    def __init__(self, max_seq_len=128, model_name: str = "distilbert-base-uncased"):
        if AutoTokenizer is None:
            raise RuntimeError("Please 'pip install transformers' to use text tokens.")
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size

    def encode_language(self, text: List[str] | str):
        if isinstance(text, str):
            text = [text]
        tok = self.tokenizer(text, max_length=self.max_seq_len, padding=True, truncation=True, return_tensors="pt")
        return tok['input_ids'], tok['attention_mask']


class ActionDecoder(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, chunk_size: int = 8, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        mid_dim = hidden_dim // 2
        out_dim = hidden_dim // 4
        
        # Create separate encoders for each action dimension
        self.dimension_encoders = nn.ModuleList([
            self._create_encoder(feature_dim, mid_dim, dropout * 0.3)
            for _ in range(action_dim)
        ])
        
        # Create action prediction heads
        self.action_heads = nn.ModuleList([
            self._create_action_head(mid_dim, out_dim, chunk_size, dropout * 0.2)
            for _ in range(action_dim)
        ])
        
        # Temporal smoothing for multi-step predictions
        if chunk_size > 1:
            self.temporal_smoother = nn.Sequential(
                nn.Conv1d(action_dim, action_dim * 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(action_dim * 2, action_dim, kernel_size=1),
            )
            
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(feature_dim, out_dim),
                nn.SiLU(), 
                nn.Linear(out_dim, action_dim),
                nn.Softplus()
            )
        
        self._initialize_weights()
    
    def _create_encoder(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
    
    def _create_action_head(self, input_dim: int, mid_dim: int, output_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
        )
    
    def _initialize_weights(self):
        # Use more conservative initialization to prevent rapid overfitting
        precision_stds = [0.01, 0.01, 0.01, 0.01]  # Increased from ultra-low values
        precision_biases = [0.0, 0.0, 0.0, 0.0]
        
        for i, head in enumerate(self.action_heads):
            final_layer = head[-1]
            std = precision_stds[i] if i < len(precision_stds) else 0.01
            bias = precision_biases[i] if i < len(precision_biases) else 0.0
            
            nn.init.normal_(final_layer.weight, mean=0, std=std)
            nn.init.constant_(final_layer.bias, bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Encode features for each action dimension
        encoded_features = [encoder(features) for encoder in self.dimension_encoders]
        
        # Generate action predictions
        action_predictions = [
            head(encoded_features[i]) for i, head in enumerate(self.action_heads)
        ]
        actions = torch.stack(action_predictions, dim=-1)
        
        # Apply temporal smoothing for multi-step predictions
        if self.chunk_size > 1:
            smoothed_actions = self.temporal_smoother(actions.transpose(1, 2)).transpose(1, 2)
            uncertainty_weights = torch.sigmoid(self.uncertainty_estimator(features).unsqueeze(1))
            actions = uncertainty_weights * actions + (1 - uncertainty_weights) * smoothed_actions
        
        # Return first timestep without aggressive precision scaling
        current_actions = actions[:, 0]
        # Gentler scaling to prevent overfitting
        precision_factors = torch.tensor([0.9, 0.9, 0.9, 0.8], 
                                       device=current_actions.device, 
                                       dtype=current_actions.dtype)
        
        return current_actions * precision_factors


class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 1536):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.layers = nn.ModuleList([
            self._make_enhanced_block(64, 128, 2),
            self._make_enhanced_block(128, 256, 2),
            self._make_enhanced_block(256, 512, 2),
            self._make_enhanced_block(512, 768, 1),
        ])
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.global_max_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        
        self.feature_proj = nn.Sequential(
            nn.Linear(768 * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _make_enhanced_block(self, in_ch: int, out_ch: int, stride: int):
        expand_ch = in_ch * 3
        
        main_path = nn.Sequential(
            nn.Conv2d(in_ch, expand_ch, 1, bias=False),
            nn.BatchNorm2d(expand_ch),
            nn.SiLU(),
            nn.Conv2d(expand_ch, expand_ch, 5, stride, 2, groups=expand_ch, bias=False),
            nn.BatchNorm2d(expand_ch),
            nn.SiLU(),
            self._make_se_block(expand_ch),
            nn.Conv2d(expand_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        
        if stride == 1 and in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        return main_path
        
    def _make_se_block(self, channels: int, reduction: int = 16):
        reduced_ch = max(1, channels // reduction)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(reduced_ch, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.stem(x)
        
        feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feature_maps.append(x)
            if i > 0 and x.shape == feature_maps[i-1].shape:
                x = x + feature_maps[i-1] * 0.1
        
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights
        
        avg_pooled = self.global_pool(x)
        max_pooled = self.global_max_pool(x)
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        features = self.feature_proj(pooled_features)
        
        return features


# =============================================================================
# MAIN VLA MODEL
# =============================================================================

class VLAModel(nn.Module):
    """State-of-the-art tiny Vision-Language-Action model."""
    def __init__(self, cfg: ModelConfig, tokenizer_vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vision_backbone = VisionEncoder(cfg.vision_dim)

        self.lang_embed = nn.Embedding(int(tokenizer_vocab_size), cfg.language_dim)
        self.lang_dropout = nn.Dropout(0.7)  # Reverted to breakthrough configuration
        self.lang_encoder = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=cfg.language_dim, nhead=8,
                                     dim_feedforward=cfg.language_dim * 2, 
                                     dropout=0.18, batch_first=True, norm_first=True),
            nn.TransformerEncoderLayer(d_model=cfg.language_dim, nhead=8,
                                     dim_feedforward=cfg.language_dim * 2,
                                     dropout=0.18, batch_first=True, norm_first=True)
        )
        self.lang_out = nn.Linear(cfg.language_dim, cfg.language_dim)

        fusion_in = cfg.language_dim + cfg.vision_dim
        self.fusion_proj = QLoRALinear(fusion_in, cfg.hidden_dim, r=32, alpha=64)

        self.mamba = nn.ModuleList([
            MambaBlock(d_model=cfg.hidden_dim, d_state=cfg.mamba_d_state, d_conv=4, expand=cfg.mamba_expand, dropout=0.25)
            for _ in range(cfg.num_mamba_layers)
        ])
        
        # Add dropout layers for regularization
        self.dropout = nn.Dropout(0.8)  # Reverted to breakthrough configuration

        self.decoder = ActionDecoder(feature_dim=cfg.hidden_dim, action_dim=cfg.action_dim, 
                                          chunk_size=cfg.chunk_size, hidden_dim=cfg.hidden_dim // 2)

    def forward(self,
                images: torch.Tensor,
                language_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                seq_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        batch_size, seq_len = images.shape[:2]
        
        # Process vision input
        vision_features = self._process_vision(images, batch_size, seq_len)
        
        # Process language input
        language_features = self._process_language(language_ids, attention_mask, batch_size, seq_len)
        
        # Fuse vision and language features
        fused_features = self._fuse_modalities(vision_features, language_features)
        
        # Apply Mamba layers for temporal modeling
        temporal_features = self._apply_temporal_modeling(fused_features)
        
        # Apply dropout for regularization
        temporal_features = self.dropout(temporal_features)
        
        # Apply attention mechanism
        attended_features = self._apply_attention(temporal_features, seq_mask, batch_size, seq_len)
        
        # Pool features across time dimension
        pooled_features = self._pool_features(attended_features, seq_mask, seq_len)
        
        # Decode to actions
        actions = self.decoder(pooled_features)
        return actions
    
    def _process_vision(self, images: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Process visual input through vision backbone."""
        # Reshape for batch processing: (B*T, C, H, W)
        images_flat = images.view(batch_size * seq_len, *images.shape[2:])
        vision_features = self.vision_backbone(images_flat)
        # Reshape back to sequence: (B, T, D)
        return vision_features.view(batch_size, seq_len, -1)
    
    def _process_language(self, language_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], 
                         batch_size: int, seq_len: int) -> torch.Tensor:
        """Process language input and create sequence-level features."""
        # Embed and encode language
        lang_embeddings = self.lang_embed(language_ids)
        lang_embeddings = self.lang_dropout(lang_embeddings)
        lang_encoded = self.lang_encoder(lang_embeddings)
        
        # Pool language features
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=lang_encoded.dtype)
            lang_pooled = (lang_encoded * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-6)
        else:
            lang_pooled = lang_encoded.mean(1)
        
        # Project and expand to match sequence length
        lang_features = self.lang_out(lang_pooled)
        return lang_features.unsqueeze(1).expand(batch_size, seq_len, -1)
    
    def _fuse_modalities(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        """Fuse vision and language features."""
        combined_features = torch.cat([vision_features, language_features], dim=-1)
        return self.fusion_proj(combined_features)
    
    def _apply_temporal_modeling(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Mamba layers for temporal modeling."""
        x = features
        for mamba_block in self.mamba:
            x = mamba_block(x)
        return x
    
    def _apply_attention(self, features: torch.Tensor, seq_mask: Optional[torch.Tensor], 
                       batch_size: int, seq_len: int) -> torch.Tensor:
        """Apply multi-head attention with causal masking."""
        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=features.device, dtype=features.dtype)
        else:
            seq_mask = seq_mask.to(features.device, dtype=features.dtype)
        
        # Multi-head attention setup
        feature_dim = features.size(-1)
        num_heads = 12
        head_dim = feature_dim // num_heads
        
        # Create Q, K, V
        q = features.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = features.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = features.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = math.sqrt(head_dim) * 0.8
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Apply causal and sequence masks
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=features.device))
        mask_4d = (seq_mask.unsqueeze(1).unsqueeze(1) * causal_mask.unsqueeze(0).unsqueeze(0))
        mask_4d = mask_4d.expand(batch_size, num_heads, seq_len, seq_len)
        
        attention_scores = attention_scores.masked_fill(mask_4d == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=0.15, training=self.training)
        
        # Apply attention and reshape
        attended_output = torch.matmul(attention_weights, v)
        attended_output = attended_output.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
        
        # Residual connection with learnable gating
        gate = torch.sigmoid(features.mean(dim=-1, keepdim=True))
        return gate * attended_output + (1 - gate) * features
    
    def _pool_features(self, features: torch.Tensor, seq_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Pool features across time dimension using multiple strategies."""
        seq_lengths = seq_mask.sum(dim=1).long()
        
        # Recency-weighted pooling
        positions = torch.arange(seq_len, device=features.device, dtype=torch.float)
        recency_decay = torch.exp(-0.1 * (seq_lengths.unsqueeze(1).float() - positions - 1).clamp(min=0))
        recency_weights = F.softmax(recency_decay * seq_mask, dim=1)
        recency_pooled = (features * recency_weights.unsqueeze(-1)).sum(dim=1)
        
        # Content-based pooling
        content_scores = torch.tanh((features * features.mean(dim=1, keepdim=True)).sum(dim=-1))
        content_weights = F.softmax(content_scores * seq_mask, dim=1)
        content_pooled = (features * content_weights.unsqueeze(-1)).sum(dim=1)
        
        # Uniform pooling
        uniform_weights = seq_mask / seq_mask.sum(dim=1, keepdim=True)
        uniform_pooled = (features * uniform_weights.unsqueeze(-1)).sum(dim=1)
        
        # Adaptive combination based on sequence length
        seq_length_ratio = (seq_lengths.float() / seq_len).unsqueeze(1)
        recency_weight = (1 - seq_length_ratio) * 0.5
        content_weight = seq_length_ratio * 0.4
        uniform_weight = 0.1
        
        # Normalize weights
        total_weight = recency_weight + content_weight + uniform_weight
        recency_weight /= total_weight
        content_weight /= total_weight
        uniform_weight /= total_weight
        
        # Combine pooled features
        final_pooled = (recency_weight * recency_pooled + 
                       content_weight * content_pooled + 
                       uniform_weight * uniform_pooled)
        
        return final_pooled

    def get_model_size_gb(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buf_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buf_size) / (1024 ** 3)


# =============================================================================
# TRAINING INFRASTRUCTURE
# =============================================================================

class VLATrainer:
    """Trainer class for the VLA model with advanced curriculum learning and optimization."""
    def __init__(self, model: VLAModel, train_set: Dataset, val_set: Dataset,
                 batch_size: int = 2, lr: float = 2e-4, use_amp: bool = True,
                 accum_steps: int = 8, language_text: str = "perform task",
                 supervise: str = "last", abs_min_tol: float = 1e-2, label_smoothing: float = 0.1):
        self.model = model
        self.device = torch.device('cuda')
        
        self.model.to(self.device)
        
        self.adaptive_training = False  # Curriculum learning disabled - consistently harmful
        self.training_phase = 0  # Fixed at optimal phase
        self.phase_switch_tolerance = 20
        self.phase_wait = 0
        self.loss_thresholds = [0.190, 0.185, 0.180, 0.175]
        self.accuracy_gates = [8.0, 12.0, 18.0, 25.0]
        
        # Ultra-gentle phase transitions for maximum stability
        self.phase_params = {
            0: {'error_scale': 1.0, 'precision_weight': 0.20},  # More conservative start
            1: {'error_scale': 1.0, 'precision_weight': 0.22},  # Minimal increase
            2: {'error_scale': 1.0, 'precision_weight': 0.24},  # Gradual progression
            3: {'error_scale': 1.0, 'precision_weight': 0.26},  # Conservative final
        }
        
        torch.backends.cudnn.benchmark = True
        
        # Model saving for best epoch
        self.best_model_path = './best_model.pth'
        self.best_accuracy = 0.0
        
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        self.transfer_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        
        torch.cuda.empty_cache()
        
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.supervise = supervise  
        self.abs_min_tol = float(abs_min_tol)

        self.train_set = train_set
        self.val_set = val_set

        optimal_batch_size = ((batch_size + 7) // 8) * 8
        
        self.train_loader = DataLoader(
            self.train_set, batch_size=optimal_batch_size, shuffle=True, 
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=None,
            collate_fn=self._collate_fn_optimized
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=optimal_batch_size, shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=None,
            collate_fn=self._collate_fn_optimized
        )

        self._compile_model()
        
        self.original_lr = lr
        self.min_lr = lr / 100  # Use the learning rate directly without scaling
        self.optimizer = Lion(
            model.parameters(), 
            lr=self.original_lr * 2.2,  # Confirmed optimal Lion LR multiplier
            weight_decay=3e-2,  # Reverted to breakthrough optimal value
            betas=(0.95, 0.98),  # Lion's default betas
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=5, min_lr=lr*0.001)
        
        self.stability_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=6, min_lr=lr/50
        )

        self.use_amp = use_amp and self.device.type == 'cuda'
        self.accum_steps = accum_steps
        self.crit_mse = nn.L1Loss()
        self.current_epoch = 0
        self.best_val = float('inf')
        self.patience = 3
        self.wait = 0
        self.best_state = None
        
        self.best_val_accuracy = 0.0
        self.accuracy_patience = 12
        self.accuracy_wait = 0
        
        self.loss_history = []
        self.accuracy_history = []
        self.instability_counter = 0
        self.recovery_mode = False
        
        self.gradient_clip_value = 0.05
        self.weight_decay_increase = 1.0  # Disabled - causes phase transition harm
        
        # Regularization
        self.label_smoothing = label_smoothing
        self.noise_factor = 0.02
        self.mixup_alpha = 0.2
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        torch.cuda.empty_cache()
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self._setup_tokenizer()
        ids, mask = self.tokenizer.encode_language([language_text])
        self.lang_ids = ids.to(self.device)
        self.lang_mask = mask.to(self.device)

        val_stats = self.val_set.action_stats
        tolerance_range = 3.0 * val_stats.std
        self.action_range = torch.from_numpy(tolerance_range).to(self.device)

    def _compile_model(self):
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
        
        compile_options = {
            'backend': 'inductor', 
            'mode': 'default',
        }
        
        self.model.vision_backbone = torch.compile(self.model.vision_backbone, **compile_options)
        self.model.decoder = torch.compile(self.model.decoder, **compile_options)

    def _collate_fn_optimized(self, batch):
        collated = {}
        
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            
            if all(v is None for v in values):
                collated[key] = None
            elif key in ['images', 'actions', 'actions_raw']:
                collated[key] = torch.stack(values, dim=0)
            elif key == 'seq_len':
                collated[key] = torch.tensor(values, device=self.device)
            else:
                if isinstance(values[0], str):
                    collated[key] = values
                else:
                    # Use .detach().clone() for proper tensor copying
                    tensor_values = [v.detach().clone() if torch.is_tensor(v) else torch.tensor(v) for v in values]
                    collated[key] = torch.stack(tensor_values)
        
        return collated

    def _setup_tokenizer(self):
        self.tokenizer = Tokenizer()

    def _build_seq_mask(self, seq_len: torch.Tensor, T: int) -> torch.Tensor:
        safe_T = min(T, 2**30)
        idx = torch.arange(safe_T, device=self.device, dtype=torch.long)
        return (idx < seq_len.unsqueeze(1)).float()

    def _pick_targets(self, actions: torch.Tensor, seq_len: torch.Tensor, T: int) -> torch.Tensor:
        B = actions.size(0)
        
        safe_T = min(T, 2**30)
        safe_B = min(B, 2**30)
        
        if self.supervise == "rand":
            valid_T = seq_len
            rand_t = torch.randint(low=1, high=safe_T, size=(safe_B,), device=self.device, dtype=torch.long)
            rand_t = torch.minimum(rand_t, valid_T - 1)
            return actions[torch.arange(safe_B, device=self.device, dtype=torch.long), rand_t]
        last_idx = torch.clamp(seq_len - 1, min=0, max=safe_T-1)
        return actions[torch.arange(safe_B, device=self.device, dtype=torch.long), last_idx]

    def save_checkpoint(self, path: str):
        stats = getattr(self.train_set, 'action_stats', None)
        ckpt = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict(),
            'best_val': self.best_val,
            'action_mean': stats.mean if stats else None,
            'action_std': stats.std if stats else None,
            'action_amin': stats.amin if stats else None,
            'action_amax': stats.amax if stats else None,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.scheduler.load_state_dict(ckpt['sched'])
        self.current_epoch = ckpt.get('epoch', 0)
        self.best_val = ckpt.get('best_val', float('inf'))
        if 'action_mean' in ckpt:
            stats = ActionStats(
                mean=ckpt['action_mean'], std=ckpt['action_std'],
                amin=ckpt['action_amin'], amax=ckpt['action_amax']
            )
            self.train_set.set_action_stats(stats)
            self.val_set.set_action_stats(stats)
            self.action_range = torch.from_numpy(stats.amax - stats.amin).to(self.device)

    def detect_training_instability(self, loss: float, accuracy: float = 0.0) -> bool:
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
            self.accuracy_history.pop(0)
        
        if len(self.loss_history) < 3:
            return False
            
        recent_losses = self.loss_history[-2:]
        if len(recent_losses) >= 2 and recent_losses[-1] > recent_losses[-2] * 1.5:
            return True
            
        if len(self.accuracy_history) >= 3:
            early_acc = max(self.accuracy_history[:2])
            recent_acc = self.accuracy_history[-1]
            if early_acc > 30.0 and recent_acc < 5.0:
                return True
                
        if all(l > 3.0 for l in recent_losses):
            return True
            
        return False
        
    def recover_from_instability(self):
        if self.instability_counter >= 2:
            return
        
        self.recovery_mode = True
        self.instability_counter += 1
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
            
        self.loss_history = self.loss_history[-1:] if len(self.loss_history) > 1 else []
        self.accuracy_history = self.accuracy_history[-1:] if len(self.accuracy_history) > 1 else []
        
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
    def update_training_phase(self, val_loss: float, accuracy_metrics: dict = None) -> bool:
        if not self.adaptive_training:
            return False
            
        phase_advanced = False
        current_loss_threshold = self.loss_thresholds[self.training_phase]
        
        loss_ready = val_loss <= current_loss_threshold
        
        accuracy_ready = True
        if accuracy_metrics and 'accuracy_all_dims_%' in accuracy_metrics:
            target_acc = self.accuracy_gates[self.training_phase]
            current_2pct_acc = accuracy_metrics['accuracy_all_dims_%'].get(0.02, 0.0)
            accuracy_ready = current_2pct_acc >= target_acc
        
        # Only advance phases if we're making steady progress
        if loss_ready and accuracy_ready:
            self.phase_wait += 1
            if self.phase_wait >= self.phase_switch_tolerance:
                if self.training_phase < len(self.loss_thresholds) - 1:
                    # Ensure accuracy is consistently improving before advancing
                    recent_acc_trend = len(self.accuracy_history) >= 3
                    if recent_acc_trend:
                        recent_accs = self.accuracy_history[-3:]
                        acc_improving = all(recent_accs[i] <= recent_accs[i+1] for i in range(len(recent_accs)-1))
                    else:
                        acc_improving = True  # Allow early phases to advance
                    
                    if acc_improving or self.training_phase == 0:  # Always allow phase 0 -> 1 transition
                        self.training_phase += 1
                        self.phase_wait = 0
                        phase_advanced = True
                        
                        # Ultra-conservative learning rate adjustment for maximum stability
                        phase_lr_multipliers = [1.0, 0.98, 0.96, 0.94]  # More gradual LR reduction
                        new_lr = self.original_lr * phase_lr_multipliers[self.training_phase]
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        
                        # Reset early stopping counters for new phase exploration
                        self.best_val = float('inf')
                        self.wait = 0
                        
                        print(f"[INFO] Advanced to training phase {self.training_phase} with LR {new_lr:.2e}")
                    else:
                        # Reset wait if not improving steadily
                        self.phase_wait = max(0, self.phase_wait - 2)
        else:
            self.phase_wait = max(0, self.phase_wait - 1)  # Gradual decay instead of reset
            
        return phase_advanced
        
    def get_training_info(self) -> str:
        if not self.adaptive_training:
            return "Adaptive: Disabled"
        
        phase_config = self.phase_params[self.training_phase]
        return (f"Phase {self.training_phase}/3 | "
                f"Scale×{phase_config['error_scale']:.1f} | "
                f"Weight×{phase_config['precision_weight']:.2f} | "
                f"Wait {self.phase_wait}/{self.phase_switch_tolerance}")
    def early_stopping_check(self, val_loss: float, accuracy_metrics: dict = None) -> bool:
        phase_advanced = self.update_training_phase(val_loss, accuracy_metrics)
        
        if phase_advanced:
            self.wait = 0
            self.best_val = float('inf')
            self.best_state = copy.deepcopy(self.model.state_dict())
            return False
        
        self.stability_scheduler.step(val_loss)
        
        # Check for loss improvement
        loss_improved = False
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.wait = 0
            loss_improved = True
        else:
            self.wait += 1
        
        # Check for accuracy improvement
        accuracy_improved = False
        if accuracy_metrics:
            current_accuracy = accuracy_metrics.get('accuracy_all_dims_%', {}).get(0.02, 0.0)
            if current_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_accuracy
                self.accuracy_wait = 0
                accuracy_improved = True
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': current_accuracy,
                    'epoch': getattr(self, 'current_epoch', 0)
                }, self.best_model_path)
                print(f"[INFO] Saved best model with accuracy {current_accuracy:.1f}%")
            else:
                self.accuracy_wait += 1
        
        # Save best state if either metric improved
        if loss_improved or accuracy_improved:
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        # Early stop if both loss and accuracy have stagnated
        loss_stagnated = self.wait >= self.patience
        accuracy_stagnated = self.accuracy_wait >= self.accuracy_patience
        
        # Detect overfitting: if loss increases while accuracy drops
        if len(self.loss_history) >= 3 and len(self.accuracy_history) >= 3:
            recent_loss_trend = sum(self.loss_history[-3:]) / 3.0
            early_loss_trend = sum(self.loss_history[:3]) / 3.0
            recent_acc_trend = sum(self.accuracy_history[-3:]) / 3.0
            early_acc_trend = sum(self.accuracy_history[:3]) / 3.0
            
            if (recent_loss_trend > early_loss_trend * 1.2 and 
                recent_acc_trend < early_acc_trend * 0.8):
                print("[WARNING] Overfitting detected - loss increasing while accuracy decreasing")
                return True
        
        return loss_stagnated and accuracy_stagnated
    
        
    def load_best_model_for_final_validation(self):
        """Load best model for final validation."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded best model from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 0):.1f}%")
            return True
        return False

    def train_epoch(self, max_batches: Optional[int] = None) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        self.optimizer.zero_grad(set_to_none=True)
        T = self.train_set.sequence_length
        
        if not hasattr(self, '_gpu_tensors_allocated'):
            self._preallocate_training_tensors()
            self._gpu_tensors_allocated = True

        for b_idx, batch in enumerate(self.train_loader):
            if max_batches and b_idx >= max_batches:
                break
            
            loss = torch.tensor(0.0, device=self.device)
            
            with torch.cuda.stream(self.transfer_stream) if hasattr(self, 'transfer_stream') else torch.no_grad():
                batch_gpu = self._transfer_batch_to_gpu_optimized(batch)
                B = batch_gpu['images'].size(0)
                
                seq_len_tensor = self._process_seq_len(batch_gpu['seq_len'], T, B)
                mask = self._build_seq_mask(seq_len_tensor, T)

            if hasattr(self, 'transfer_stream'):
                torch.cuda.synchronize()
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = self.model(
                    images=batch_gpu['images'],
                    language_ids=self.lang_ids.expand(B, -1),
                    attention_mask=self.lang_mask.expand(B, -1),
                    seq_mask=mask,
                )

                gt = self._pick_targets(batch_gpu['actions'], seq_len_tensor, T)
                loss = self._compute_optimized_loss(pred, gt)
                
                if loss is None:
                    continue
                    
                loss = loss / self.accum_steps

            if torch.isfinite(loss).all() and not torch.isnan(loss).any() and loss.item() < 25.0:
                loss.backward()
                
                total_loss += loss.item() * self.accum_steps
                n += 1
            else:
                self.optimizer.zero_grad(set_to_none=True)
                if torch.isnan(loss).any() or loss.item() > 30.0:
                    self.recover_from_instability()
                continue

            if (b_idx + 1) % self.accum_steps == 0:
                self._gradient_step()

            del batch_gpu, pred, gt, loss
            
            if (b_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

        if n > 0 and (n % self.accum_steps != 0):
            pass
            
        # Scheduler step moved to after validation in main loop

        self.current_epoch += 1
        return total_loss / max(1, n)

    def _gradient_step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value, 
                                                  norm_type=2.0, error_if_nonfinite=False)
        
        if torch.isfinite(grad_norm) and grad_norm < 10.0:
            self.optimizer.step()
        else:
            if grad_norm > 5.0:
                self.recover_from_instability()
                # Increase weight decay to combat overfitting
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] *= self.weight_decay_increase
        
        self.optimizer.zero_grad(set_to_none=True)
    
    def _transfer_batch_to_gpu_optimized(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch_gpu = {}
        for k, v in batch.items():
            batch_gpu[k] = self.train_set.normalize_action(v) if k == 'actions' else v
        
        return batch_gpu
    
    def _process_seq_len(self, seq_len_raw, T: int, B: int) -> torch.Tensor:
        if torch.is_tensor(seq_len_raw):
            return torch.clamp(seq_len_raw, min=1, max=T).long()
        seq_len_vals = [max(1, min(int(x), T)) for x in seq_len_raw] if isinstance(seq_len_raw, (list, tuple)) else [max(1, min(int(seq_len_raw), T))] * B
        return torch.tensor(seq_len_vals, device=self.device, dtype=torch.long)
        
    def _preallocate_training_tensors(self):
        dummy = torch.zeros(1, device=self.device)
        del dummy
        torch.cuda.empty_cache()
            
    def _compute_optimized_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Simplified loss function to reduce overfitting."""
        if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)
            
        # Basic L1 and L2 losses
        l1_loss = F.l1_loss(pred, gt)
        l2_loss = F.mse_loss(pred, gt)
        
        # Smooth L1 loss for robustness
        smooth_l1 = F.smooth_l1_loss(pred, gt, beta=0.1)
        
        # Dimension-specific balancing with X-axis focus
        diff = pred - gt
        abs_diff = diff.abs()
        
        # Enhanced dimension-specific weighting to fix X,Y axis issues
        if abs_diff.size(-1) >= 4:
            x_error = abs_diff[..., 0]  # X-axis is dimension 0
            y_error = abs_diff[..., 1]  # Y-axis 
            z_error = abs_diff[..., 2]  # Z-axis
            g_error = abs_diff[..., 3]  # Gripper
            
            # Equal weighting across all dimensions
            dim_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred.device, dtype=pred.dtype)
            weighted_error = abs_diff * dim_weights.unsqueeze(0).unsqueeze(0)
            balance_penalty = weighted_error.mean()
        else:
            # Fallback for different dimensions
            max_error_per_sample = abs_diff.max(dim=-1)[0]
            mean_error_per_sample = abs_diff.mean(dim=-1)
            balance_penalty = F.relu(max_error_per_sample - mean_error_per_sample * 2.0).mean()
        
        # Progressive loss weighting based on phase for gradual refinement
        phase_config = self.phase_params.get(self.training_phase, self.phase_params[0])
        precision_weight = phase_config['precision_weight']
        
        # Gradually increase balance penalty weight as training progresses
        balance_weight = 0.05 + (0.1 * self.training_phase / 3.0)  # 0.05 -> 0.15
        
        # Maintain consistent base loss across phases
        base_weight = 0.6 - balance_weight  # Reduce base as balance increases
        
        combined_loss = (base_weight * (0.6 * l1_loss + 0.4 * smooth_l1) + 
                        0.2 * l2_loss + 
                        balance_weight * balance_penalty) * precision_weight
        
        return torch.clamp(combined_loss, max=10.0)

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        T = self.val_set.sequence_length
        
        batch_gpu = {}
        
        for b_idx, batch in enumerate(self.val_loader):
            if max_batches and b_idx >= max_batches:
                break
                
            batch_gpu = self._transfer_batch_to_gpu_optimized(batch)
            B = batch_gpu['images'].size(0)
            
            seq_len_tensor = self._process_seq_len(batch_gpu['seq_len'], T, B)
            mask = self._build_seq_mask(seq_len_tensor, T)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = self.model(
                    images=batch_gpu['images'],
                    language_ids=self.lang_ids.expand(B, -1),
                    attention_mask=self.lang_mask.expand(B, -1),
                    seq_mask=mask,
                )
                gt = self._pick_targets(batch_gpu['actions'], seq_len_tensor, T)
                
                loss = self._compute_optimized_loss(pred, gt)
                
            if torch.isfinite(loss).all() and not torch.isnan(loss).any() and loss.item() < 25.0:
                total += loss.item()
                n += 1
            else:
                print(f"[eval][batch={b_idx}] Skipping batch - invalid loss: {loss.item():.2e}")
            
            if (b_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        return total / max(1, n)

    def _compute_val_action_range(self) -> torch.Tensor:
        """Compute action range from validation data to avoid data leakage."""
        all_actions = []
        for batch in self.val_loader:
            actions = batch['actions']
            if torch.is_tensor(actions):
                # Convert to float32 first to handle bfloat16
                actions = actions.detach().cpu().float().numpy()
            all_actions.append(actions.reshape(-1, actions.shape[-1]))
        
        if not all_actions:
            return self.action_range  # fallback
        
        combined = np.concatenate(all_actions, axis=0)
        val_std = combined.std(axis=0).astype(np.float32) + 1e-8
        return torch.from_numpy(3.0 * val_std).to(self.device)

    @torch.no_grad()
    def test_accuracy(self, num_val_batches: Optional[int] = None) -> Dict[str, Any]:
        self.model.eval()
        T = self.val_set.sequence_length
        total_samples = 0
        mse_sum = 0.0
        mae_sum = 0.0

        tol_list = [0.01, 0.02, 0.05, 0.1]
        all_dim_acc = {}
        any_dim_acc = {}
        per_dim_acc = {}
        individual_dim_acc = {}

        # Use validation-based action range to avoid data leakage
        val_action_range = self._compute_val_action_range()
        A = val_action_range.numel()
        batches_processed = 0
        batches_skipped = 0
        for b_idx, batch in enumerate(self.val_loader):
            if num_val_batches and b_idx >= num_val_batches:
                break
                
            batch = to_device(batch, self.device)
            B = batch['images'].size(0)
            
            seq_len_raw = batch['seq_len']
            if torch.is_tensor(seq_len_raw):
                seq_len_tensor = torch.clamp(seq_len_raw, min=1, max=T).long().to(self.device)
            else:
                if isinstance(seq_len_raw, (list, tuple)):
                    seq_len_vals = [max(1, min(int(x), T)) for x in seq_len_raw]
                else:
                    seq_len_vals = [max(1, min(int(seq_len_raw), T))] * B
                seq_len_tensor = torch.tensor(seq_len_vals, device=self.device, dtype=torch.long)
            
            mask = self._build_seq_mask(seq_len_tensor, T)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = self.model(
                    images=batch['images'],
                    language_ids=self.lang_ids.expand(B, -1),
                    attention_mask=self.lang_mask.expand(B, -1),
                    seq_mask=mask,
                )
            gt = self._pick_targets(batch['actions'], seq_len_tensor, T)
            
            # Check if this batch would pass loss validation
            batch_loss = self._compute_optimized_loss(pred, gt)
            if not (torch.isfinite(batch_loss).all() and not torch.isnan(batch_loss).any() and batch_loss.item() < 50.0):
                batches_skipped += 1
                continue  # Skip this batch for accuracy calculation too
            
            batches_processed += 1

            pred_np = self.val_set.denormalize_action(pred.detach().cpu())
            gt_np = self.val_set.denormalize_action(gt.detach().cpu())
            pred_t = pred_np.to(self.device)
            gt_t = gt_np.to(self.device)

            mse_sum += F.mse_loss(pred_t, gt_t, reduction='sum').item()
            mae_sum += F.l1_loss(pred_t, gt_t, reduction='sum').item()
            total_samples += B

            ranges = val_action_range.unsqueeze(0).expand(B, -1)
            abs_err = (pred_t - gt_t).abs()
            for frac in tol_list:
                tol = ranges * frac
                within = (abs_err <= tol)
                all_hit = within.all(dim=1).float().mean().item() * 100.0
                any_hit = within.any(dim=1).float().mean().item() * 100.0
                per_dim = within.float().mean().item() * 100.0
                all_dim_acc[frac] = all_dim_acc.get(frac, 0.0) + all_hit
                any_dim_acc[frac] = any_dim_acc.get(frac, 0.0) + any_hit
                per_dim_acc[frac] = per_dim_acc.get(frac, 0.0) + per_dim
                
                if frac not in individual_dim_acc:
                    individual_dim_acc[frac] = {}
                dim_names = ['x', 'y', 'z', 'gripper']
                for dim_idx in range(min(A, len(dim_names))):
                    dim_name = dim_names[dim_idx]
                    dim_accuracy = within[:, dim_idx].float().mean().item() * 100.0
                    if dim_name not in individual_dim_acc[frac]:
                        individual_dim_acc[frac][dim_name] = []
                    individual_dim_acc[frac][dim_name].append(dim_accuracy)

        # Debug info available: {batches_processed} processed, {batches_skipped} skipped
        
        if total_samples == 0:
            print(f"[ERROR] No valid samples found in accuracy calculation! Processed {batches_processed} batches, skipped {batches_skipped}")
            return {}
        num_elems = total_samples * A
        rmse = float(torch.sqrt(torch.tensor(mse_sum / num_elems)).item())
        n_batches = max(1, (b_idx + 1 if num_val_batches is None else min(b_idx + 1, num_val_batches)))
        all_dim_acc = {t: v / n_batches for t, v in all_dim_acc.items()}
        any_dim_acc = {t: v / n_batches for t, v in any_dim_acc.items()}
        per_dim_acc = {t: v / n_batches for t, v in per_dim_acc.items()}
        
        individual_dim_final = {}
        for frac in individual_dim_acc:
            individual_dim_final[frac] = {}
            for dim_name in individual_dim_acc[frac]:
                individual_dim_final[frac][dim_name] = sum(individual_dim_acc[frac][dim_name]) / len(individual_dim_acc[frac][dim_name])

        main_acc = all_dim_acc.get(0.02, 0.0)
        rating = (
            "EXCELLENT (SOTA Level)" if main_acc >= 75 else
            "GOOD (Competitive)" if main_acc >= 50 else
            "FAIR (Baseline)" if main_acc >= 25 else
            "NEEDS IMPROVEMENT"
        )
        return {
            'action_mse': mse_sum / num_elems,
            'action_mae': mae_sum / num_elems,
            'action_rmse': rmse,
            'accuracy_all_dims_%': all_dim_acc,
            'accuracy_any_dim_%': any_dim_acc,
            'accuracy_per_dim_mean_%': per_dim_acc,
            'accuracy_individual_dims_%': individual_dim_final,
            'total_samples': total_samples,
            'performance_rating': rating,
            'debug_action_range': val_action_range.cpu().numpy().tolist(),
            'debug_avg_abs_error': (abs_err.mean().item() if 'abs_err' in locals() else 0.0),
        }

    def format_comprehensive_metrics(self, accuracy_metrics: dict, prefix: str = "") -> str:
        """Format comprehensive accuracy metrics for display during training."""
        if not accuracy_metrics:
            return f"{prefix}No metrics available"
        
        # Extract key metrics
        all_dim_acc = accuracy_metrics.get('accuracy_all_dims_%', {})
        individual_acc = accuracy_metrics.get('accuracy_individual_dims_%', {})
        
        lines = []
        lines.append(f"{prefix}=== Comprehensive Validation Results ===")
        
        # Overall accuracy summary
        acc_1pct = all_dim_acc.get(0.01, 0.0)
        acc_2pct = all_dim_acc.get(0.02, 0.0) 
        acc_5pct = all_dim_acc.get(0.05, 0.0)
        acc_10pct = all_dim_acc.get(0.1, 0.0)
        
        lines.append(f"{prefix}All-Dims Accuracy: 1%={acc_1pct:.1f}% | 2%={acc_2pct:.1f}% | 5%={acc_5pct:.1f}% | 10%={acc_10pct:.1f}%")
        
        # Per-dimension breakdown
        if individual_acc:
            lines.append(f"{prefix}Individual Dimension Accuracy:")
            for tolerance in [0.01, 0.02, 0.05, 0.1]:
                if tolerance in individual_acc:
                    dims = individual_acc[tolerance]
                    x_acc = dims.get('x', 0.0)
                    y_acc = dims.get('y', 0.0)
                    z_acc = dims.get('z', 0.0)
                    g_acc = dims.get('gripper', 0.0)
                    lines.append(f"{prefix}  {tolerance*100:3.0f}%: X={x_acc:4.1f}% | Y={y_acc:4.1f}% | Z={z_acc:4.1f}% | Gripper={g_acc:4.1f}%")
        
        # Error metrics if available
        if 'action_mse' in accuracy_metrics:
            mse = accuracy_metrics['action_mse']
            mae = accuracy_metrics['action_mae'] 
            rmse = accuracy_metrics['action_rmse']
            lines.append(f"{prefix}Error Metrics: MSE={mse:.2e} | MAE={mae:.4f} | RMSE={rmse:.4f}")
        
        # Performance rating
        if 'performance_rating' in accuracy_metrics:
            rating = accuracy_metrics['performance_rating']
            lines.append(f"{prefix}Performance: {rating}")
            
        lines.append(f"{prefix}=====================================\n")
        return "\n".join(lines)
    
    def print_metrics(self, metrics: dict, title: str = "METRICS"):
        """Print formatted metrics with a title."""
        print(f"\n[INFO] {title}")
        formatted_output = self.format_comprehensive_metrics(metrics, prefix="")
        print(formatted_output)


# =============================================================================
# MODEL FACTORY AND DATASET BUILDER
# =============================================================================

def create_model(action_dim: int, tokenizer_vocab_size: int) -> Tuple[VLAModel, ModelConfig]:
    """Create a state-of-the-art VLA model with optimized configuration."""
    cfg = ModelConfig(
        action_dim=action_dim,
        vision_dim=2048,
        language_dim=1536,
        hidden_dim=3072,
        num_mamba_layers=8,
        mamba_d_state=128,
        mamba_expand=2,
        qlora_r=96,
        qlora_alpha=192,
        use_proprio=False,
        chunk_size=1,
        use_precision_focus=True,
    )
    model = VLAModel(cfg, tokenizer_vocab_size=tokenizer_vocab_size)
    return model, cfg


def build_datasets(args):
    """Build training and validation datasets from H5 files."""
    if not os.path.exists(args.data_dir) or not any(p.endswith('.h5') for p in os.listdir(args.data_dir)):
        raise FileNotFoundError(f"H5 data directory not found or empty: {args.data_dir}")
    
    train_set = TriangleDataset(
        data_path=args.data_dir,
        sequence_length=args.sequence_length,
        mode='train',
        train_split=args.train_split,
        compute_stats_from='train',
        stats_max_files=args.stats_max_files,
        use_augmentation=True
    )
    val_set = TriangleDataset(
        data_path=args.data_dir,
        sequence_length=args.sequence_length,
        mode='val',
        train_split=args.train_split,
        compute_stats_from='train',
        stats_max_files=args.stats_max_files,
        use_augmentation=False
    )
    return train_set, val_set


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training script for the VLA model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./triangle_real_data')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=120)  
    parser.add_argument('--lr', type=float, default=1e-6)  
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--stats_max_files', type=int, default=1000)
    parser.add_argument('--language_text', type=str, default='manipulate the object using vision')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--auto_download', action='store_true', help='Auto-download Stanford triangle_real_data.zip if data_dir missing/empty')
    parser.add_argument('--stanford_zip_url', type=str, default='http://downloads.cs.stanford.edu/juno/triangle_real_data.zip')
    parser.add_argument('--supervise', choices=['last','rand'], default='last', help='train/eval target frame selection')
    parser.add_argument('--abs_min_tol', type=float, default=2e-3, help='ultra-tight tolerance for precision training')

    args = parser.parse_args()

    set_seed(args.seed)

    train_set, val_set = build_datasets(args)
    action_dim = int(train_set.action_stats.mean.shape[0])

    tok = Tokenizer()
    model, _ = create_model(action_dim=action_dim, tokenizer_vocab_size=tok.vocab_size)
    
    model = model.to(dtype=torch.bfloat16)

    trainer = VLATrainer(
        model, train_set, val_set,
        batch_size=args.batch_size, lr=args.lr, use_amp=True,
        accum_steps=args.accum_steps,
        language_text=args.language_text,
        supervise=args.supervise, abs_min_tol=args.abs_min_tol,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, 'best.pt')

    best_accuracy = 0.0
    plateau_counter = 0
    
    for epoch in range(args.epochs):
        trainer.current_epoch = epoch + 1
        t0 = time.time()
        tr_loss = trainer.train_epoch(max_batches=args.max_train_batches)
        ev_loss = trainer.evaluate(max_batches=args.max_val_batches)
        dt = time.time() - t0
        
        training_info = trainer.get_training_info()
        
        accuracy_metrics = None
        current_accuracy = 0.0
        # Calculate accuracy every epoch to properly track performance
        accuracy_metrics = trainer.test_accuracy(num_val_batches=min(50, args.max_val_batches or 50))
        current_accuracy = accuracy_metrics.get('accuracy_all_dims_%', {}).get(0.02, 0.0)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            plateau_counter = 0
        else:
            plateau_counter += 1
        
        # Basic epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} | Train {tr_loss:.6f} | Val {ev_loss:.6f} | "
              f"Best {trainer.best_val:.6f} | Wait {trainer.wait}/{trainer.patience} | "
              f"Acc2% {current_accuracy:.1f}% | Best {best_accuracy:.1f}% | {training_info} | {dt:.1f}s")
        
        # Comprehensive validation results
        if accuracy_metrics:
            comprehensive_results = trainer.format_comprehensive_metrics(accuracy_metrics, prefix="  ")
            print(comprehensive_results)
        
        if trainer.detect_training_instability(ev_loss, current_accuracy):
            trainer.recover_from_instability()
            
        if trainer.recovery_mode and len(trainer.loss_history) >= 3:
            recent_stable = all(l < 1.0 for l in trainer.loss_history[-3:])
            if recent_stable and current_accuracy > 5.0:
                trainer.recovery_mode = False

        if hasattr(trainer.scheduler, 'step') and 'ReduceLR' in str(type(trainer.scheduler)):
            trainer.scheduler.step(ev_loss)
            
        # Also step scheduler based on accuracy to prevent overfitting
        if hasattr(trainer.stability_scheduler, 'step'):
            # Use negative accuracy so we can use 'min' mode scheduler
            trainer.stability_scheduler.step(-current_accuracy if current_accuracy > 0 else 0)
        
        if trainer.early_stopping_check(ev_loss, accuracy_metrics):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model for final validation and benchmarking
    print("\n[INFO] Loading best model for final validation...")
    if trainer.load_best_model_for_final_validation():
        # Run final validation with best model
        print("[INFO] Running final validation with best model...")
        final_loss, final_accuracy_metrics = trainer.test_accuracy(num_val_batches=args.max_val_batches), trainer.test_accuracy(num_val_batches=args.max_val_batches)
        trainer.print_metrics(final_accuracy_metrics, "FINAL VALIDATION (Best Model)")
        
        # Benchmark inference speed
        print("[INFO] Benchmarking inference speed...")

    if trainer.best_state is not None:
        model.load_state_dict(trainer.best_state)

    res = trainer.test_accuracy(num_val_batches=args.max_val_batches)
    for k, v in res.items():
        if k == 'accuracy_individual_dims_%':
            print(f"{k}:")
            for tolerance, dims in v.items():
                print(f"  {tolerance*100}%: x={dims.get('x', 0.0):.1f}% | y={dims.get('y', 0.0):.1f}% | z={dims.get('z', 0.0):.1f}% | gripper={dims.get('gripper', 0.0):.1f}%")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    mp.freeze_support()
    main()

