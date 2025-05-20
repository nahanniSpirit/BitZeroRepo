"""
BitZero Hybrid Precision Module

This module implements a simplified and more robust version of the dynamic precision
allocation system for BitZero, ensuring dimension compatibility across the model.
MODIFIED: Added Straight-Through Estimator (STE) to BitQuantizer for 1.58-bit path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any

# Attempt to import the new vocabulary size, handle if not found for standalone testing
try:
    from tokenizer_config import VOCAB_SIZE as DEFAULT_VOCAB_SIZE
except ImportError:
    print("Warning: tokenizer_config.py not found. Using default vocab_size=256 for HybridPrecisionTransformer.")
    DEFAULT_VOCAB_SIZE = 256


class BitQuantizer(nn.Module):
    """
    Simplified bit-level quantizer for BitZero.
    
    This quantizer supports 1.58-bit quantization (ternary + zero) as described
    in the BitNet paper, but with simplified implementation for robustness.
    Includes Straight-Through Estimator (STE) for the 1.58-bit path during training.
    Includes caching for quantized weights during inference (Rec 1.3).
    """
    
    def __init__(self, bit_width: float = 1.58):
        """
        Initialize bit quantizer.
        
        Args:
            bit_width: Bit width for quantization (1.58 for ternary + zero)
        """
        super().__init__()
        self.bit_width = bit_width
        self.cached_quantized_tensors = {} # Cache for inference (renamed for clarity)
        
    def forward(self, x: torch.Tensor, per_tensor_scale: Optional[torch.Tensor] = None, cache_key: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass for quantization.
        
        Args:
            x: Input tensor to quantize
            per_tensor_scale: Optional scaling factor
            cache_key: Optional key for caching the quantized tensor (e.g., for static weights during inference)
            
        Returns:
            Quantized tensor
        """
        # Use cache during inference if key is provided and tensor is found
        if not self.training and cache_key is not None and cache_key in self.cached_quantized_tensors:
            return self.cached_quantized_tensors[cache_key]

        result_tensor = x # Default to x if quantization is skipped

        if self.bit_width == 1.58:
            if per_tensor_scale is None:
                # Fallback if no specific scale is provided (should ideally always be provided for weights)
                # This could be a source of issues if not handled carefully.
                # For weights, HybridPrecisionLinear should compute and pass this.
                current_scale = torch.mean(torch.abs(x)).clamp(min=1e-5) # Use absolute mean as scale, clamp for stability
            else:
                current_scale = per_tensor_scale.clamp(min=1e-5) # Clamp provided scale

            # Quantization process based on BitNet (ternary {-1, 0, 1})
            # Step 1: Scale and center. The paper uses W_norm = W / EMA(|W|) + eps.
            # Here, x is analogous to W. current_scale is analogous to EMA(|W|).
            x_normalized = x / current_scale

            # Step 2: Quantize to {-1, 0, 1}. Thresholding based on the paper's idea.
            # A common approach for ternary is two thresholds.
            # For BitNet 1.58 (ternary with zero point), it's often simpler:
            # positive_threshold = some_value (e.g., 0.5, or learned/derived)
            # negative_threshold = -some_value
            # Here, we use the paper's simplified threshold idea:
            # The paper mentions quantizing W_q = Q(W/alpha) where Q rounds to nearest {-1, 0, 1} values.
            # Or using specific thresholds. Let's use a simplified round-based approach for ternary values.
            # round(x_normalized * c) / c where c makes it effectively ternary.
            # Or, direct thresholding:
            
            # Let's use a more direct ternary quantization:
            # scale values for quantization based on max value (as in BitNet-LLM for activations)
            # For weights, paper suggests scaling factor alpha derived from EMA of abs weights.
            # We use `current_scale` as this alpha.
            
            # x_normalized is W / alpha
            # Quantize x_normalized to values in {-1, 0, 1}
            # This is a hard quantization step.
            # A common way for ternary is to have two thresholds, e.g., +/- 0.5 after normalization.
            # Let's use a slightly more robust thresholding than the 0.7*mean(abs(x_scaled)) for stability.
            # If x_normalized values are typically around [-1, 1], then thresholds like +/- 0.5 make sense.
            
            # Let's try a fixed threshold on the normalized values
            threshold_val = 0.5 # Fixed threshold for {-1, 0, 1} on normalized input
            
            x_quantized_ternary_normalized = torch.zeros_like(x_normalized)
            x_quantized_ternary_normalized[x_normalized > threshold_val] = 1.0
            x_quantized_ternary_normalized[x_normalized < -threshold_val] = -1.0
            # Values between -threshold_val and threshold_val remain 0.

            # Step 3: Rescale: W_quantized = x_quantized_ternary_normalized * alpha
            output_quantized_rescaled = x_quantized_ternary_normalized * current_scale
            
            if self.training:
                # STE: output = W_original + (W_quantized - W_original).detach()
                result_tensor = x + (output_quantized_rescaled - x).detach()
            else:
                result_tensor = output_quantized_rescaled
        elif self.bit_width > 0 and self.bit_width != 32: # Assuming 32 means no quantization for this branch
            # Uniform quantizer (no STE applied here by default, typically used for non-critical higher precision)
            num_levels = 2 ** int(self.bit_width)
            if num_levels <= 1: result_tensor = x
            else:
                max_val = torch.max(torch.abs(x)).clamp(min=1e-5) # Clamp max_val
                # if max_val == 0: result_tensor = x # Covered by clamp
                # else:
                step = 2 * max_val / (num_levels - 1)
                if step == 0: result_tensor = x # Should be rare after clamping max_val
                else:
                    x_scaled_uniform = x / step
                    result_tensor = torch.round(x_scaled_uniform) * step
        else: # bit_width is 0 or 32 or some other non-quantizing value
             result_tensor = x


        # Cache result during inference if key is provided
        if not self.training and cache_key is not None and result_tensor is not None:
            self.cached_quantized_tensors[cache_key] = result_tensor
            
        return result_tensor


class HybridPrecisionLinear(nn.Module):
    """
    Hybrid precision linear layer for BitZero.
    
    This layer implements a simplified version of adaptive precision allocation
    that is more robust to dimension changes and initialization issues.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 critical_ratio: float = 0.1, # Default from plan, can be made adaptive
                 base_bits: float = 1.58):
        """
        Initialize hybrid precision linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias
            critical_ratio: Ratio of weights to keep at higher precision
            base_bits: Bit width for base quantization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.critical_ratio_config = critical_ratio # Store configured base ratio
        self.base_bits = base_bits
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Remove learnable weight_scaling. We will compute it dynamically.
        # self.weight_scaling = nn.Parameter(torch.ones(1), requires_grad=True) 
        
        self.quantizer = BitQuantizer(bit_width=base_bits)
        self.reset_parameters()
        self.weight_cache_key = f"weight_{id(self)}"

    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def get_adaptive_critical_ratio(self, layer_idx: Optional[int] = None, num_total_layers: Optional[int] = None) -> float:
        """
        Returns an adaptive critical ratio.
        If layer_idx or num_total_layers is None, returns the fixed critical_ratio.
        This method implements recommendation 1.1 (Quantization Ratio Tuning).
        """
        if layer_idx is None or num_total_layers is None:
            return self.critical_ratio_config # Fallback to fixed ratio

        # Adaptive logic from recommendation
        if layer_idx == 0 or layer_idx == num_total_layers - 1:
            return 0.15  # Higher precision for embedding-like and output-like layers
        elif layer_idx < num_total_layers // 3:
            return 0.12  # Higher precision for early layers
        else:
            return 0.08  # Lower precision for middle layers

    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None, num_total_layers: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with hybrid precision.
        
        Args:
            x: Input tensor
            layer_idx: Optional current layer index for adaptive critical ratio.
            num_total_layers: Optional total number of layers for adaptive critical ratio.
            
        Returns:
            Output tensor
        """
        # Ensure input has correct shape
        if x.dim() == 2:
            features = x.size(1)
        elif x.dim() == 3:
            _, _, features = x.size()
        else:
            raise ValueError(f"Expected input to be 2D or 3D, got {x.dim()}D")
        
        if features != self.in_features:
            raise ValueError(f"Expected input features {self.in_features}, got {features}")
        
        original_shape = x.shape
        if x.dim() == 3:
            x_reshaped = x.reshape(-1, features)
        else:
            x_reshaped = x
        
        current_critical_ratio = self.get_adaptive_critical_ratio(layer_idx, num_total_layers)

        # Compute per-tensor scale for weights (alpha in BitNet)
        # This is done on the full-precision weights `self.weight`
        # Clamp to avoid division by zero or extremely small scales.
        with torch.no_grad(): # Scale computation should not affect gradients of weights directly here
            weight_scale_factor = torch.mean(torch.abs(self.weight)).clamp(min=1e-5)
        
        # The quantizer will now use this weight_scale_factor
        quantized_part = self.quantizer(self.weight, 
                                        per_tensor_scale=weight_scale_factor, # Pass the computed scale
                                        cache_key=self.weight_cache_key if not self.training else None)

        if current_critical_ratio >= 1.0: # All critical, no quantization
            effective_weight = self.weight
        elif current_critical_ratio <= 0.0: # All quantized
            effective_weight = quantized_part
        else: # Hybrid
            weight_abs = torch.abs(self.weight)
            num_elements = self.weight.numel()
            k = int(current_critical_ratio * num_elements)
            
            if k >= num_elements: # Should be caught by cr >= 1.0 but good to have
                effective_weight = self.weight
            elif k <= 0: # Should be caught by cr <= 0.0
                 effective_weight = quantized_part
            else:
                threshold_val = torch.kthvalue(weight_abs.flatten(), num_elements - k).values
                critical_mask = (weight_abs >= threshold_val) # Boolean mask
                
                # Use .data for where to avoid in-place modification issues if self.weight was involved on both sides without .data
                # However, since quantized_part is a new tensor, this should be fine.
                effective_weight = torch.where(
                    critical_mask,
                    self.weight, 
                    quantized_part
                )
        
        output = F.linear(x_reshaped, effective_weight, self.bias)
        return output.view(*original_shape[:-1], -1) if x.dim() == 3 else output


class HybridPrecisionAttention(nn.Module):
    """
    Hybrid precision attention module for BitZero.
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 dropout_prob: float = 0.1,
                 critical_ratio: float = 0.1): # Accepts base critical_ratio
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.critical_ratio = critical_ratio # Store base critical_ratio
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
        self.head_size = hidden_size // num_heads
        
        # Pass critical_ratio to HybridPrecisionLinear layers
        self.q_proj = HybridPrecisionLinear(hidden_size, hidden_size, critical_ratio=critical_ratio)
        self.k_proj = HybridPrecisionLinear(hidden_size, hidden_size, critical_ratio=critical_ratio)
        self.v_proj = HybridPrecisionLinear(hidden_size, hidden_size, critical_ratio=critical_ratio)
        self.o_proj = HybridPrecisionLinear(hidden_size, hidden_size, critical_ratio=critical_ratio)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_scores = None
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None, # For adaptive ratio
                num_total_layers: Optional[int] = None) -> torch.Tensor: # For adaptive ratio
        batch_size, seq_len, _ = hidden_states.size()
        
        # Pass layer_idx and num_total_layers to linear layer's forward pass
        q = self.q_proj(hidden_states, layer_idx=layer_idx, num_total_layers=num_total_layers)
        k = self.k_proj(hidden_states, layer_idx=layer_idx, num_total_layers=num_total_layers)
        v = self.v_proj(hidden_states, layer_idx=layer_idx, num_total_layers=num_total_layers)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        self.attention_scores = attention_probs.detach()
        
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(context, layer_idx=layer_idx, num_total_layers=num_total_layers) # Pass adaptive params
        return output


class HybridPrecisionMLP(nn.Module):
    """
    Hybrid precision MLP module for BitZero.
    """
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 dropout_prob: float = 0.1,
                 critical_ratio: float = 0.1): # Accepts base critical_ratio
        super().__init__()
        # Pass critical_ratio to HybridPrecisionLinear layers
        self.fc1 = HybridPrecisionLinear(hidden_size, intermediate_size, critical_ratio=critical_ratio)
        self.fc2 = HybridPrecisionLinear(intermediate_size, hidden_size, critical_ratio=critical_ratio)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor, layer_idx: Optional[int] = None, num_total_layers: Optional[int] = None) -> torch.Tensor:
        # Pass layer_idx and num_total_layers to linear layer's forward pass
        hidden_states = self.fc1(hidden_states, layer_idx=layer_idx, num_total_layers=num_total_layers)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states, layer_idx=layer_idx, num_total_layers=num_total_layers)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class HybridPrecisionTransformerLayer(nn.Module):
    """
    Hybrid precision transformer layer for BitZero.
    """
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 intermediate_size: int,
                 dropout_prob: float = 0.1,
                 critical_ratio: float = 0.1): # Accepts base critical_ratio
        super().__init__()
        self.attention = HybridPrecisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            critical_ratio=critical_ratio # Pass base critical_ratio
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.mlp = HybridPrecisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob,
            critical_ratio=critical_ratio # Pass base critical_ratio
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None, # For adaptive ratio
                num_total_layers: Optional[int] = None) -> torch.Tensor: # For adaptive ratio
        residual = hidden_states
        hidden_states_norm = self.ln1(hidden_states)
        # Pass layer_idx and num_total_layers to attention and MLP
        attention_output = self.attention(hidden_states_norm, attention_mask, layer_idx=layer_idx, num_total_layers=num_total_layers)
        hidden_states = attention_output + residual
        
        residual = hidden_states
        hidden_states_norm = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states_norm, layer_idx=layer_idx, num_total_layers=num_total_layers)
        hidden_states = mlp_output + residual
        
        return hidden_states


class HybridPrecisionTransformer(nn.Module):
    """
    Hybrid precision transformer model for BitZero.
    """
    def __init__(self, 
                 vocab_size: int = DEFAULT_VOCAB_SIZE,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_seq_len: int = 768,
                 dropout_prob: float = 0.1,
                 critical_ratio: float = 0.1, # Base critical_ratio for layers
                 pad_token_id: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.critical_ratio = critical_ratio # Store base critical_ratio
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0

        # For embedding and output layers, we might use a specific critical_ratio or allow them to use adaptive
        # The adaptive logic can consider layer_idx=0 for embedding (conceptually) and layer_idx=num_layers-1 for output.
        # Here, token_embedding is effectively layer -1 or 0, output is layer num_layers.
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.layers = nn.ModuleList([
            HybridPrecisionTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout_prob=dropout_prob,
                critical_ratio=critical_ratio # Pass base critical_ratio to each layer
            ) for _ in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(hidden_size)
        # Output layer uses HybridPrecisionLinear, can also use adaptive ratio
        self.output_projection = HybridPrecisionLinear(hidden_size, vocab_size, critical_ratio=critical_ratio)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, HybridPrecisionLinear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        seq_len = min(seq_len, self.max_seq_len)
        input_ids_truncated = input_ids[:, :seq_len]
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Note: Embedding layers are not typically quantized in the same way as linear layers.
        # The critical_ratio concept applies to HybridPrecisionLinear.
        # For adaptive ratio, conceptual layer_idx for embedding could be 0.
        # We'll pass layer_idx to actual transformer layers.
        token_embeds = self.token_embedding(input_ids_truncated)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is None and self.pad_token_id is not None:
            extended_attention_mask = (input_ids_truncated != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        elif attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                extended_attention_mask = attention_mask.unsqueeze(1)
            else:
                extended_attention_mask = attention_mask
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            if not torch.is_floating_point(extended_attention_mask):
                 extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
        else:
            extended_attention_mask = None

        for i, layer_module in enumerate(self.layers):
            # Pass current layer_idx and total_layers for adaptive critical_ratio
            hidden_states = layer_module(hidden_states, extended_attention_mask, layer_idx=i, num_total_layers=self.num_layers)
        
        hidden_states = self.ln_out(hidden_states)
        # For output projection, conceptual layer_idx is num_layers - 1 (or num_layers for post-loop)
        # Let's use num_layers - 1 as if it's the last layer in the stack.
        logits = self.output_projection(hidden_states, layer_idx=self.num_layers -1 , num_total_layers=self.num_layers) 
        
        return logits
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        total_params = 0
        quantized_params = 0
        critical_params_count = 0 # Renamed for clarity
        
        # Iterate through HybridPrecisionLinear modules to get their effective critical ratios
        # This includes linear layers within attention, MLP, and the final output projection.
        
        effective_bits_numerator = 0.0
        
        # Stats for layers
        for layer_idx, layer_module_concrete in enumerate(self.layers):
            # Attention projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                module = getattr(layer_module_concrete.attention, proj_name)
                if isinstance(module, HybridPrecisionLinear):
                    layer_params = module.weight.numel()
                    total_params += layer_params
                    current_cr = module.get_adaptive_critical_ratio(layer_idx, self.num_layers)
                    layer_critical = int(layer_params * current_cr)
                    
                    critical_params_count += layer_critical
                    quantized_params += (layer_params - layer_critical)
                    effective_bits_numerator += (layer_critical * 32.0) + ((layer_params - layer_critical) * module.base_bits)

            # MLP projections
            for proj_name in ['fc1', 'fc2']:
                module = getattr(layer_module_concrete.mlp, proj_name)
                if isinstance(module, HybridPrecisionLinear):
                    layer_params = module.weight.numel()
                    total_params += layer_params
                    current_cr = module.get_adaptive_critical_ratio(layer_idx, self.num_layers)
                    layer_critical = int(layer_params * current_cr)

                    critical_params_count += layer_critical
                    quantized_params += (layer_params - layer_critical)
                    effective_bits_numerator += (layer_critical * 32.0) + ((layer_params - layer_critical) * module.base_bits)

        # Stats for output projection
        if isinstance(self.output_projection, HybridPrecisionLinear):
            module = self.output_projection
            layer_params = module.weight.numel()
            total_params += layer_params
            # Conceptual layer_idx for output_projection is self.num_layers - 1
            current_cr = module.get_adaptive_critical_ratio(self.num_layers - 1, self.num_layers)
            layer_critical = int(layer_params * current_cr)
            
            critical_params_count += layer_critical
            quantized_params += (layer_params - layer_critical)
            effective_bits_numerator += (layer_critical * 32.0) + ((layer_params - layer_critical) * module.base_bits)

        if total_params == 0:
            return {
                "total_params": 0, "quantized_params": 0, "critical_params": 0,
                "quantized_ratio": 0, "critical_ratio_actual_avg": 0, "effective_bits_avg": 32.0
            }

        quantized_ratio_actual = quantized_params / total_params
        critical_ratio_actual_avg = critical_params_count / total_params
        effective_bits_avg = effective_bits_numerator / total_params if total_params > 0 else 32.0
        
        return {
            "total_params_in_hybrid_layers": total_params,
            "quantized_params": quantized_params,
            "critical_params": critical_params_count,
            "quantized_ratio_in_hybrid_layers": quantized_ratio_actual,
            "critical_ratio_actual_avg_in_hybrid_layers": critical_ratio_actual_avg,
            "effective_bits_avg_in_hybrid_layers": effective_bits_avg
        }

def create_bitzero_nano_hybrid(critical_ratio: float = 0.1, vocab_size: int = DEFAULT_VOCAB_SIZE, pad_token_id: Optional[int] = 0, max_seq_len: int = 768) -> HybridPrecisionTransformer:
    """
    Create a nano-sized BitZero model with hybrid precision.
    """
    return HybridPrecisionTransformer(
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=max_seq_len, # Pass the parameter here
        dropout_prob=0.1,
        critical_ratio=critical_ratio, # Base critical_ratio
        pad_token_id=pad_token_id
    )


def create_bitzero_micro_hybrid(critical_ratio: float = 0.1, vocab_size: int = DEFAULT_VOCAB_SIZE, pad_token_id: Optional[int] = 0, max_seq_len: int = 768) -> HybridPrecisionTransformer:
    """
    Create a micro-sized BitZero model with hybrid precision.
    """
    return HybridPrecisionTransformer(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_seq_len=max_seq_len, # Pass the parameter here
        dropout_prob=0.1,
        critical_ratio=critical_ratio, # Base critical_ratio
        pad_token_id=pad_token_id
    )


if __name__ == "__main__":
    try:
        from tokenizer_config import VOCAB_SIZE as current_vocab_size, CHAR_TO_ID, PAD_TOKEN
        current_pad_token_id = CHAR_TO_ID[PAD_TOKEN]
    except ImportError:
        current_vocab_size, current_pad_token_id = 256, 0

    print(f"--- Testing Nano Model (Quantization Caching, Adaptive CR & STE) ---")
    model_nano = create_bitzero_nano_hybrid(critical_ratio=0.1, vocab_size=current_vocab_size, pad_token_id=current_pad_token_id, max_seq_len=768) # Added max_seq_len for test
    
    # Test caching during inference
    model_nano.eval() # Set to inference mode
    print(f"Quantization stats (Nano, eval): {model_nano.get_quantization_stats()}")
    
    batch_size, seq_len = 1, 5 # Small batch for easy inspection
    input_ids = torch.randint(0, current_vocab_size, (batch_size, seq_len))

    print(f"\nFirst inference pass (populates cache):")
    with torch.no_grad():
        logits_eval1 = model_nano(input_ids)
    
    # Inspect cache (example for the first linear layer in the first transformer block)
    first_transformer_layer_q_proj = model_nano.layers[0].attention.q_proj
    cache_key_q = first_transformer_layer_q_proj.weight_cache_key
    if cache_key_q in first_transformer_layer_q_proj.quantizer.cached_quantized_tensors:
        print(f"Cache HIT for key '{cache_key_q}' (q_proj weight) - Shape: {first_transformer_layer_q_proj.quantizer.cached_quantized_tensors[cache_key_q].shape}")
    else:
        print(f"Cache MISS for key '{cache_key_q}' - this is unexpected after first pass.")

    print(f"\nSecond inference pass (should use cache):")
    with torch.no_grad():
        logits_eval2 = model_nano(input_ids) # Should be faster if caching is effective (hard to measure here)

    # Verify outputs are the same (they should be, as weights are static in eval)
    if torch.allclose(logits_eval1, logits_eval2):
        print("\nLogits from first and second inference pass are identical (as expected with caching).")
    else:
        print("\nERROR: Logits from first and second pass differ!")

    # Clear cache for a specific layer and test again (optional detailed test)
    # print("\nClearing cache for one layer and re-testing...")
    # if cache_key_q in first_transformer_layer_q_proj.quantizer.cached_quantized_tensors:
    #     del first_transformer_layer_q_proj.quantizer.cached_quantized_tensors[cache_key_q]
    
    # with torch.no_grad():
    #     logits_eval3 = model_nano(input_ids)
    # if cache_key_q not in first_transformer_layer_q_proj.quantizer.cached_quantized_tensors:
    #      print(f"Cache for key '{cache_key_q}' correctly empty after manual del, then re-populated on next pass.")


    # Test training mode (cache should not be used, STE active)
    model_nano.train()
    print(f"\nQuantization stats (Nano, train): {model_nano.get_quantization_stats()}")
    # Clear all caches before training pass to ensure no interference from eval mode
    for module in model_nano.modules():
        if isinstance(module, BitQuantizer):
            module.cached_quantized_tensors.clear()
    print("Caches cleared for training mode test.")
            
    logits_train = model_nano(input_ids) # STE should be active
    print(f"Output shape (Nano, train): {logits_train.shape}")
    if cache_key_q in first_transformer_layer_q_proj.quantizer.cached_quantized_tensors:
        print(f"ERROR: Cache for key '{cache_key_q}' was populated during training mode!")
    else:
        print(f"Cache for key '{cache_key_q}' correctly NOT populated during training mode.")