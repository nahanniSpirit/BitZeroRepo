"""
BitZero Optimized Training Module

This module implements optimized training for BitZero with batch processing,
GPU acceleration, and other performance enhancements, including the ability
to load and train from a JSONL dataset.

MODIFIED:
    - Implemented a robust conversation packing and causal masking strategy for training
      on conversational data, ensuring proper handling of max_seq_len limits.
    - Added Learning Rate Scheduler (OneCycleLR) - Recommendation 4.2.
    - Added Progressive Batch Size Scaling (get_optimal_batch_size) - Recommendation 4.1.
    - FIXED: AttributeError when accessing self.verbose in threaded data preparation.
"""

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import time
import json
import random
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from hybrid_precision import create_bitzero_nano_hybrid, create_bitzero_micro_hybrid 
from task_generator import TaskGenerator, TaskVerifier
try:
    from tokenizer_config import (
        VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR,
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
    )
except ImportError:
    print("ERROR: tokenizer_config.py not found. This script requires it.")
    # Minimal fallback to prevent immediate crash, but training will be broken.
    VOCAB_SIZE = 150 
    CHAR_TO_ID = {chr(i): i for i in range(150)} 
    ID_TO_CHAR = {i: chr(i) for i in range(150)} 
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    CHAR_TO_ID[PAD_TOKEN], CHAR_TO_ID[UNK_TOKEN], CHAR_TO_ID[BOS_TOKEN], CHAR_TO_ID[EOS_TOKEN] = 0,1,2,3
    ID_TO_CHAR = {v:k for k,v in CHAR_TO_ID.items()}


class OptimizedTrainer:
    """Optimized trainer for BitZero with batch processing and GPU acceleration."""
    
    # Define a minimum number of tokens for the target/label part of a sequence
    # This prevents training on examples where the assistant's response is too short after truncation.
    MIN_TARGET_LEN_FOR_TRAINING = 3 

    def __init__(self, 
                 model_size: str = "nano",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 initial_batch_size: int = 8, 
                 learning_rate: float = 1e-5, 
                 max_seq_len: int = 768, 
                 use_mixed_precision: bool = True,
                 use_gradient_checkpointing: bool = True,
                 gradient_clip_val: float = 1.0, 
                 use_parallel_tasks: bool = True, 
                 dataset_path: Optional[str] = None, 
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs",
                 critical_ratio_for_model: float = 0.1,
                 num_epochs_for_scheduler: int = 5, 
                 total_steps_for_scheduler: Optional[int] = None,
                 weight_decay_for_adamw: float = 0.01,
                 vram_per_sample_gb_estimate: float = 0.2, 
                 max_dynamic_batch_size: int = 32,
                 verbose: bool = False # Make sure verbose is stored
                 ): 
        self.model_size = model_size
        self.batch_size = initial_batch_size 
        self.initial_batch_size = initial_batch_size 
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.use_mixed_precision = use_mixed_precision and device == "cuda" and torch.cuda.is_available()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_clip_val = gradient_clip_val
        self.use_parallel_tasks = use_parallel_tasks
        self.dataset_path = dataset_path
        self.full_dataset_cache = [] 
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.debug_batch_printed_this_epoch = False 
        self.critical_ratio_for_model = critical_ratio_for_model
        self.weight_decay_for_adamw = weight_decay_for_adamw
        self.vram_per_sample_gb_estimate = vram_per_sample_gb_estimate 
        self.max_dynamic_batch_size = max_dynamic_batch_size 
        self.verbose = verbose # Store verbose flag

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        if torch.cuda.is_available() and device == "cuda":
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            self.device = "cuda"
        else:
            print("CUDA not available or not selected, using CPU.")
            self.device = "cpu"
            if use_mixed_precision: self.use_mixed_precision = False
        
        print(f"Initializing BitZero-{model_size} on {self.device} with Vocabulary Size: {VOCAB_SIZE}...")
        pad_token_id_for_model = CHAR_TO_ID.get(PAD_TOKEN) 

        if model_size.lower() == "nano":
            self.model = create_bitzero_nano_hybrid(
                critical_ratio=self.critical_ratio_for_model, 
                vocab_size=VOCAB_SIZE, 
                pad_token_id=pad_token_id_for_model,
                max_seq_len=self.max_seq_len
            )
        elif model_size.lower() == "micro":
            self.model = create_bitzero_micro_hybrid(
                critical_ratio=self.critical_ratio_for_model, 
                vocab_size=VOCAB_SIZE, 
                pad_token_id=pad_token_id_for_model,
                max_seq_len=self.max_seq_len
            )
        else: raise ValueError(f"Unknown model size: {model_size}")
        
        self.model.to(self.device)
        
        if self.use_gradient_checkpointing: self.enable_gradient_checkpointing()
        
        self.task_generator = TaskGenerator()
        self.task_verifier = TaskVerifier()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=self.weight_decay_for_adamw
        )
        
        self.scheduler = None 
        self.num_epochs_for_scheduler = num_epochs_for_scheduler
        self.total_steps_for_scheduler = total_steps_for_scheduler

        self.scaler = torch.amp.GradScaler(enabled=self.use_mixed_precision) if self.device == 'cuda' else None
        if self.use_mixed_precision: print("Mixed precision training enabled")
        
        self.task_cache = [] 
        self.stats = {
            "episodes": 0, "correct_answers": 0, "total_answers": 0,
            "rewards": [], "difficulties": [], "batch_times": [], "memory_usage": [],
            "batch_losses": [], "learning_rates": [], "current_batch_sizes": [],
            "math_batch_losses": [], 
            "code_batch_losses": []  
        }
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"BitZero initialized with {num_params} parameters")
        print(f"Quantization stats: {self.model.get_quantization_stats()}")
        
        if self.device == "cuda": self.verify_cuda_usage()

    def get_optimal_batch_size(self) -> int:
        """Dynamically adjust batch size based on available VRAM (Rec 4.1)."""
        if self.device != "cuda":
            return self.initial_batch_size # No adjustment needed for CPU, use initial

        try:
            torch.cuda.empty_cache() # Attempt to clear cache before checking

            current_usage_gb = torch.cuda.memory_allocated() / 1e9
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            reserved_vram_gb = 1.0 # Increased fixed reservation for safety
            available_vram_gb = total_vram_gb - current_usage_gb - reserved_vram_gb
            
            if available_vram_gb <= 0.2: # Stricter threshold for reducing batch size
                new_batch_size = max(1, self.batch_size // 2) 
                return new_batch_size

            if self.vram_per_sample_gb_estimate <= 0:
                return self.initial_batch_size

            calculated_optimal_size = int(available_vram_gb / self.vram_per_sample_gb_estimate)
            
            # If this is an early estimate (low current usage), be more cautious
            if current_usage_gb < total_vram_gb * 0.4: # If less than 40% VRAM used
                calculated_optimal_size = int(calculated_optimal_size * 0.70) # Reduce by 30%

            optimal_batch_size = max(1, min(calculated_optimal_size, self.max_dynamic_batch_size))
            return optimal_batch_size

        except Exception as e:
            print(f"Warning: Error in get_optimal_batch_size: {e}. Falling back to current batch size: {self.batch_size}")
            return self.batch_size


    def _configure_scheduler_if_needed(self, num_epochs: int, estimated_batches_per_epoch: int):
        """Configures the OneCycleLR scheduler."""
        if self.scheduler is None: 
            if self.total_steps_for_scheduler is None:
                if self.dataset_path and self.full_dataset_cache and self.initial_batch_size > 0:
                    precise_batches_per_epoch = (len(self.full_dataset_cache) + self.initial_batch_size - 1) // self.initial_batch_size
                    self.total_steps_for_scheduler = num_epochs * precise_batches_per_epoch
                else: 
                    self.total_steps_for_scheduler = num_epochs * estimated_batches_per_epoch 
                
                if self.total_steps_for_scheduler > 0 :
                    self.total_steps_for_scheduler = int(self.total_steps_for_scheduler * 1.1) + 1 
                else: 
                    self.total_steps_for_scheduler = num_epochs * 100 

            if self.total_steps_for_scheduler <= 0:
                print("Warning: total_steps_for_scheduler is 0 or less. Scheduler will not be effective.")
                self.scheduler = None 
                return

            print(f"Configuring OneCycleLR scheduler with total_steps: {self.total_steps_for_scheduler}, max_lr: {self.learning_rate}")
            self.scheduler = lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.total_steps_for_scheduler,
                pct_start=0.1,      
                div_factor=25.0,    
                final_div_factor=10000.0 
            )

    def enable_gradient_checkpointing(self):
        if not (hasattr(self.model, 'layers') and isinstance(self.model.layers, torch.nn.ModuleList)):
            print("Warning: Model does not have 'layers' attribute or it's not a ModuleList. Gradient checkpointing cannot be applied.")
            self.use_gradient_checkpointing = False
            return
        try:
            from torch.utils.checkpoint import checkpoint as torch_checkpoint_function
            if not callable(torch_checkpoint_function):
                self.use_gradient_checkpointing = False; return 
        except ImportError:
            self.use_gradient_checkpointing = False; return 

        print("Applying gradient checkpointing to model layers...")
        applied_checkpointing_to_any_layer = False
        for i, layer in enumerate(self.model.layers):
            if not hasattr(layer, 'forward') or not callable(layer.forward): continue
            if not hasattr(layer, 'forward_orig'): layer.forward_orig = layer.forward  
            layer.use_reentrant = False 
            def create_checkpointed_forward(module_layer_local, chkpt_func_local):
                def custom_forward_for_checkpoint(*inputs_cfc, **kwargs_cfc):
                    return module_layer_local.forward_orig(*inputs_cfc, **kwargs_cfc)
                def new_forward_method(*args_nfm, **kwargs_nfm):
                    return chkpt_func_local(custom_forward_for_checkpoint, *args_nfm, **kwargs_nfm, use_reentrant=module_layer_local.use_reentrant)
                return new_forward_method
            layer.forward = create_checkpointed_forward(layer, torch_checkpoint_function)
            applied_checkpointing_to_any_layer = True
        
        if applied_checkpointing_to_any_layer: print("Gradient checkpointing enabled and applied to layers.")
        else: self.use_gradient_checkpointing = False; print("Gradient checkpointing was requested but could not be applied.")


    def verify_cuda_usage(self):
        if self.device != "cuda": return
        model_on_cuda = next(self.model.parameters()).is_cuda
        test_input = torch.randint(0, VOCAB_SIZE, (1, 10), device=self.device)
        output = self.model(test_input)
        output_on_cuda = output.is_cuda
        current_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Model on CUDA: {model_on_cuda}")
        print(f"Output on CUDA: {output_on_cuda}")
        print(f"Current VRAM usage: {current_memory:.2f} GB")
        print(f"Max VRAM usage: {max_memory:.2f} GB")
        if not model_on_cuda or not output_on_cuda: print("WARNING: Some operations are not on CUDA!")

    def tokenize_sequence(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        if not CHAR_TO_ID or (add_bos and BOS_TOKEN not in CHAR_TO_ID) or \
           (add_eos and EOS_TOKEN not in CHAR_TO_ID) or UNK_TOKEN not in CHAR_TO_ID or PAD_TOKEN not in CHAR_TO_ID:
            print("CRITICAL ERROR: Tokenizer vocabulary not properly initialized during tokenize_sequence.")
            return [] 
        token_ids = [CHAR_TO_ID[BOS_TOKEN]] if add_bos else []
        for char in text: token_ids.append(CHAR_TO_ID.get(char, CHAR_TO_ID[UNK_TOKEN]))
        if add_eos: token_ids.append(CHAR_TO_ID[EOS_TOKEN])
        return token_ids

    def pad_sequence(self, token_ids: List[int], for_labels: bool = False) -> List[int]:
        pad_value = -100 if for_labels else CHAR_TO_ID.get(PAD_TOKEN)
        current_len = len(token_ids)
        if current_len > self.max_seq_len:
            # If truncating, ensure EOS is at the end if it was intended to be
            # This logic should mostly be handled by _prepare_single_qa_pair_for_lm_training now
            if not for_labels and CHAR_TO_ID.get(EOS_TOKEN) is not None:
                if token_ids[-1] == CHAR_TO_ID.get(EOS_TOKEN) and token_ids[self.max_seq_len-1] != CHAR_TO_ID.get(EOS_TOKEN):
                    return token_ids[:self.max_seq_len-1] + [CHAR_TO_ID[EOS_TOKEN]]
            return token_ids[:self.max_seq_len]
        return token_ids + [pad_value] * (self.max_seq_len - current_len)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        if not ID_TO_CHAR or EOS_TOKEN not in CHAR_TO_ID or PAD_TOKEN not in CHAR_TO_ID or BOS_TOKEN not in CHAR_TO_ID:
            return "[Detokenization Error]"
        text_chars = []
        token_ids_list = token_ids.squeeze().tolist() if token_ids.dim() > 1 else token_ids.tolist()
        eos_id, pad_id, bos_id = CHAR_TO_ID.get(EOS_TOKEN), CHAR_TO_ID.get(PAD_TOKEN), CHAR_TO_ID.get(BOS_TOKEN)
        for token_id_val in token_ids_list:
            if token_id_val in (eos_id, pad_id, -100): break 
            if token_id_val != bos_id: text_chars.append(ID_TO_CHAR.get(token_id_val, UNK_TOKEN))
        return "".join(text_chars)

    def _prepare_single_qa_pair_for_lm_training(self, user_content: str, assistant_content: str, verbose_arg: bool) -> Optional[Tuple[List[int], List[int]]]:
        """
        Prepares a single user-assistant conversation pair into input_ids and labels
        for language model training, respecting max_seq_len and applying causal masking.
        verbose_arg: Explicitly passed verbose flag for use in threaded context.
        """
        
        # Tokenize parts
        user_part_tokens = self.tokenize_sequence(f"User: {user_content}", add_bos=False, add_eos=False)
        bitzero_prefix_tokens = self.tokenize_sequence(" BitZero: ", add_bos=False, add_eos=False) # Note the leading space
        assistant_part_tokens = self.tokenize_sequence(assistant_content, add_bos=False, add_eos=False)

        # Construct the full sequence of token IDs
        # Includes BOS, user_part, EOS (after user), BitZero_prefix, assistant_part, EOS (after assistant)
        input_ids_full_composition = [CHAR_TO_ID[BOS_TOKEN]] + user_part_tokens + [CHAR_TO_ID[EOS_TOKEN]] + \
                                     bitzero_prefix_tokens + assistant_part_tokens + [CHAR_TO_ID[EOS_TOKEN]]
        
        # Calculate the length of the prompt part for causal masking
        # This includes everything up to, and including, the " BitZero: " prefix
        prompt_len_for_masking = (1 + # BOS
                                  len(user_part_tokens) + 
                                  1 + # EOS after user
                                  len(bitzero_prefix_tokens))
        
        # Handle truncation if the full sequence exceeds max_seq_len
        # We truncate from the right (end) to keep the beginning of the prompt and as much of the answer as possible.
        if len(input_ids_full_composition) > self.max_seq_len:
            input_ids_final = input_ids_full_composition[:self.max_seq_len]
            
            # If truncation cut into the prompt part, adjust prompt_len_for_masking
            # It's crucial that the prompt part is never entirely removed if it's the start of the sequence.
            # We want to ensure at least MIN_TARGET_LEN_FOR_TRAINING tokens are left for the target.
            prompt_len_for_masking = min(prompt_len_for_masking, len(input_ids_final) - self.MIN_TARGET_LEN_FOR_TRAINING)
            prompt_len_for_masking = max(0, prompt_len_for_masking) # Cannot be negative
            
            # Ensure the truncated sequence still ends with EOS if it was supposed to
            if CHAR_TO_ID.get(EOS_TOKEN) is not None and input_ids_final[-1] != CHAR_TO_ID.get(EOS_TOKEN):
                if len(input_ids_final) > 1: # Only replace if there's space
                    input_ids_final[-1] = CHAR_TO_ID.get(EOS_TOKEN)
                elif self.max_seq_len >= 1: # If seq_len is 1, just be EOS
                    input_ids_final = [CHAR_TO_ID.get(EOS_TOKEN)]
                else: # Cannot fit even EOS
                    return None 
            
        else:
            input_ids_final = input_ids_full_composition # No truncation needed

        # Create labels with causal masking: -100 for prompt, actual tokens for target
        labels_final = [-100] * prompt_len_for_masking + input_ids_final[prompt_len_for_masking:]
        
        # Apply final padding to `max_seq_len` (only padding, no more truncation here)
        input_ids_padded = self.pad_sequence(input_ids_final, for_labels=False)
        labels_padded = self.pad_sequence(labels_final, for_labels=True)

        # Final sanity checks
        if len(input_ids_padded) != self.max_seq_len or len(labels_padded) != self.max_seq_len:
            if verbose_arg: # Use verbose_arg here
                print(f"Skipping pair: final length mismatch after padding. Input {len(input_ids_padded)}, Labels {len(labels_padded)}, MaxSeq {self.max_seq_len}")
            return None
        
        num_actual_labels = sum(1 for x in labels_padded if x != -100)
        if num_actual_labels < self.MIN_TARGET_LEN_FOR_TRAINING:
            if verbose_arg: # Use verbose_arg here
                print(f"Skipping pair: too few actual labels ({num_actual_labels} < {self.MIN_TARGET_LEN_FOR_TRAINING}) after processing. Likely prompt too long for meaningful answer.")
            return None

        return input_ids_padded, labels_padded


    def _load_full_dataset_once(self):
        if self.dataset_path and not self.full_dataset_cache:
            print(f"Loading dataset from {self.dataset_path} for the first time...")
            raw_dataset = []
            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try: raw_dataset.append(json.loads(line))
                        except json.JSONDecodeError as e: print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                print(f"Loaded {len(raw_dataset)} raw items from {self.dataset_path}")
                
                prepared_items = []
                with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                    # Pass self.verbose (the value from __init__) as an explicit argument
                    futures = [executor.submit(self.prepare_data_for_model, item, self.verbose) for item in raw_dataset]
                    for future in tqdm(as_completed(futures), total=len(raw_dataset), desc="Preparing dataset items"):
                        result = future.result()
                        if result: # Only add if preparation was successful
                            prepared_items.append(result)

                self.full_dataset_cache = prepared_items
                print(f"Successfully prepared {len(self.full_dataset_cache)} valid items for training.")
            except Exception as e: 
                print(f"Error loading dataset: {e}"); 
                self.full_dataset_cache = []
        elif not self.dataset_path: 
            print("No dataset_path provided. Dynamic task generation will be used if enabled.")


    def prepare_data_for_model(self, dataset_item: Dict[str, Any], verbose_arg: bool) -> Optional[Tuple[List[int], List[int], Any, str]]:
        """
        Prepares a single dataset item (conversation pair) for model training,
        including tokenization, padding/truncation, and label masking.
        Returns input_ids, labels, expected_answer_for_verifier, task_type.
        verbose_arg: Explicitly passed verbose flag for use in threaded context.
        """
        try:
            task_type = dataset_item['type']
            conversation = dataset_item.get('conversation')
            if not isinstance(conversation, list) or len(conversation) < 2:
                if verbose_arg: print(f"Skipping item: conversation not list or too short: {dataset_item.get('conversation')}")
                return None
            
            user_turn, assistant_turn = conversation[0], conversation[1]
            if user_turn.get('role') != 'user' or assistant_turn.get('role') != 'assistant':
                if verbose_arg: print(f"Skipping item: roles not user/assistant: {user_turn.get('role')}, {assistant_turn.get('role')}")
                return None

            user_content = user_turn.get('content')
            assistant_content = assistant_turn.get('content')
            if not user_content or not assistant_content: 
                if verbose_arg: print(f"Skipping item: empty user/assistant content: '{user_content}', '{assistant_content}'")
                return None
            
            # Call the dedicated preparation helper for LM training format
            prepared_tokens = self._prepare_single_qa_pair_for_lm_training(user_content, assistant_content, verbose_arg)
            if prepared_tokens is None:
                return None # Indicate that this pair cannot be prepared for training
            
            input_ids, labels = prepared_tokens

            # Expected answer for verifier (still needed for evaluation/difficulty adjustment with dynamic tasks)
            # For pre-loaded datasets, this might be redundant if the verification logic isn't used
            # but it's kept for consistency in the data structure.
            expected_answer_for_verifier = dataset_item.get('metadata', {}).get('expected_answer')
            if expected_answer_for_verifier is None: 
                if task_type == "math":
                    try: expected_answer_for_verifier = float(assistant_content)
                    except ValueError: expected_answer_for_verifier = assistant_content 
                elif task_type == "code":
                    expected_answer_for_verifier = dataset_item.get('metadata', {}).get('expected_answer', assistant_content)
            
            return input_ids, labels, expected_answer_for_verifier, task_type
        except Exception as e: 
            if verbose_arg: print(f"Error preparing data item: {e}. Item: {dataset_item.get('conversation', 'N/A')[:50]}...")
            return None


    def cache_tasks_for_epoch(self, num_dynamic_tasks_to_generate: int = 0):
        self.task_cache = [] 
        if self.dataset_path:
            if not self.full_dataset_cache: self._load_full_dataset_once()
            if self.full_dataset_cache:
                self.task_cache = list(self.full_dataset_cache) 
                random.shuffle(self.task_cache)
                print(f"Epoch task cache refreshed with {len(self.task_cache)} tasks from dataset.")
            else: print("Dataset is empty or failed to load.")
        elif num_dynamic_tasks_to_generate > 0 : 
            print(f"Pre-generating {num_dynamic_tasks_to_generate} dynamic tasks for epoch...")
            
            prepared_dynamic_tasks = []
            if self.use_parallel_tasks:
                with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                    # Submit raw task generation, then wrap the preparation of the returned tuple
                    future_raw_tasks = [executor.submit(self.task_generator.generate_task) for _ in range(num_dynamic_tasks_to_generate)]
                    for future in tqdm(as_completed(future_raw_tasks), total=num_dynamic_tasks_to_generate, desc="Caching dynamic tasks"):
                        task_desc, exp_ans_data, task_type_dyn = future.result()
                        
                        actual_assistant_content_for_lm = None
                        if task_type_dyn == "code" and isinstance(exp_ans_data, dict) and "solution" in exp_ans_data:
                            actual_assistant_content_for_lm = exp_ans_data["solution"]
                        elif task_type_dyn == "math":
                            actual_assistant_content_for_lm = str(exp_ans_data) # Convert math answer to string
                        else:
                            # Fallback if task_type_dyn isn't 'code' or 'math' or no solution/answer is extractable
                            actual_assistant_content_for_lm = "Okay." 
                        
                        if actual_assistant_content_for_lm:
                            # Pass self.verbose to the preparation function
                            prepared_item = self._prepare_single_qa_pair_for_lm_training(task_desc, actual_assistant_content_for_lm, self.verbose)
                            if prepared_item:
                                # Append input_ids, labels, and also verifier_data for stat tracking
                                prepared_dynamic_tasks.append((prepared_item[0], prepared_item[1], exp_ans_data, task_type_dyn))
            else: # Sequential task generation for non-parallel
                for _ in tqdm(range(num_dynamic_tasks_to_generate), desc="Caching dynamic tasks (seq)"):
                    task_desc, exp_ans_data, task_type_dyn = self.task_generator.generate_task()
                    actual_assistant_content_for_lm = None
                    if task_type_dyn == "code" and isinstance(exp_ans_data, dict) and "solution" in exp_ans_data:
                        actual_assistant_content_for_lm = exp_ans_data["solution"]
                    elif task_type_dyn == "math":
                        actual_assistant_content_for_lm = str(exp_ans_data)
                    else:
                        actual_assistant_content_for_lm = "Okay." # Fallback

                    if actual_assistant_content_for_lm:
                        # Pass self.verbose to the preparation function
                        prepared_item = self._prepare_single_qa_pair_for_lm_training(task_desc, actual_assistant_content_for_lm, self.verbose)
                        if prepared_item:
                            prepared_dynamic_tasks.append((prepared_item[0], prepared_item[1], exp_ans_data, task_type_dyn))
            
            self.task_cache = prepared_dynamic_tasks
            print(f"Dynamic task cache filled with {len(self.task_cache)} prepared tasks for epoch.")
        else: 
            print("No dataset path and no dynamic tasks requested for epoch.")
        self.debug_batch_printed_this_epoch = False

    def get_next_task_from_cache(self) -> Optional[Tuple[List[int], List[int], Any, str]]:
        """
        Retrieves the next prepared task (input_ids, labels, verifier_data, task_type) from cache.
        """
        return self.task_cache.pop(0) if self.task_cache else None

    def train_batch(self, batch_data: List[Tuple[List[int], List[int], Any, str]], current_epoch: int, current_batch_idx: int, verbose: bool) -> Dict[str, Any]:
        """
        Processes a batch of pre-tokenized, pre-labeled data for training.
        `batch_data` is expected to be a list of (input_ids, labels, verifier_data, task_type) tuples.
        """
        
        input_ids_list = [item[0] for item in batch_data]
        labels_list = [item[1] for item in batch_data]
        task_types = [item[3] for item in batch_data] # For potential type-specific loss tracking
        
        self.model.train() 
        self.optimizer.zero_grad(set_to_none=True)
        
        num_valid_items_for_loss = len(input_ids_list)

        if num_valid_items_for_loss == 0: 
            return {"batch_size": 0, "batch_loss": 0, "math_loss_in_batch": 0, "code_loss_in_batch": 0, "difficulty": self.task_generator.current_difficulty, "current_lr": self.optimizer.param_groups[0]['lr']}

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)
        
        if verbose and not self.debug_batch_printed_this_epoch and num_valid_items_for_loss > 0:
            print(f"\n--- Debug Batch (Epoch {current_epoch}, Batch {current_batch_idx+1}, Current BS: {self.batch_size}, Valid Items: {num_valid_items_for_loss}) ---")
            for i_debug in range(min(2, input_ids_tensor.size(0))): 
                # Find prompt_len for debug display based on -100 in labels
                prompt_len_for_debug = 0
                try:
                    # Find first non -100 in labels, which marks the start of the target
                    first_target_idx = (labels_tensor[i_debug] != -100).nonzero(as_tuple=True)[0][0].item()
                    prompt_len_for_debug = first_target_idx
                except IndexError: 
                    # If no non -100 labels (e.g., all -100), treat as full prompt (though should be caught earlier)
                    prompt_len_for_debug = input_ids_tensor.size(1) 
                
                print(f"Example {i_debug+1} Input Tokens (first 40): {input_ids_tensor[i_debug, :40].tolist()}")
                # Detokenize the prompt part
                print(f"Example {i_debug+1} Input Detokenized (prompt part, approx {prompt_len_for_debug} tokens): '{self.detokenize(input_ids_tensor[i_debug, :prompt_len_for_debug])}'")
                print(f"Example {i_debug+1} Label Tokens (first 40): {labels_tensor[i_debug, :40].tolist()}")
                
                # Detokenize the label content (assistant's part)
                label_content_tokens_for_display = []
                for tok_idx, tok_val in enumerate(labels_tensor[i_debug].tolist()):
                    if tok_idx >= prompt_len_for_debug and tok_val != -100: # Only include actual labels
                        label_content_tokens_for_display.append(tok_val)
                        if tok_val == CHAR_TO_ID.get(EOS_TOKEN): # Stop if EOS is the actual label
                            break
                print(f"Example {i_debug+1} Label Detokenized (target part): '{self.detokenize(torch.tensor(label_content_tokens_for_display))}'")

            print(f"  Tensor Shapes: input_ids: {input_ids_tensor.shape}, labels: {labels_tensor.shape}")
            print("--- End Debug Batch ---")
            self.debug_batch_printed_this_epoch = True


        with torch.amp.autocast(device_type=self.device, enabled=self.use_mixed_precision and self.device == 'cuda'):
            logits = self.model(input_ids_tensor) 
            if verbose and not hasattr(self, '_logits_shape_printed_this_epoch'): 
                print(f"  Logits Shape: {logits.shape}")
                self._logits_shape_printed_this_epoch = True
            
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels_tensor.view(-1), ignore_index=-100, reduction='mean')
        
        current_lr_for_tracking = self.optimizer.param_groups[0]['lr']

        if torch.isnan(loss) or torch.isinf(loss) or num_valid_items_for_loss == 0:
            if num_valid_items_for_loss > 0: 
                print(f"Warning: NaN/Inf loss ({loss.item() if 'loss' in locals() and hasattr(loss, 'item') else 'N/A'}) or no valid items. Skipping backward for this batch.")
            batch_total_loss = loss.item() if 'loss' in locals() and hasattr(loss, 'item') and num_valid_items_for_loss > 0 else 0.0
            avg_math_loss_this_batch = 0 
            avg_code_loss_this_batch = 0
        else:
            batch_total_loss = loss.item() 
            if self.use_mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val > 0: 
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else: 
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
            
            if self.scheduler: 
                if self.scheduler.last_epoch < self.scheduler.total_steps -1: 
                    self.scheduler.step()
                    current_lr_for_tracking = self.scheduler.get_last_lr()[0]
                else:
                    current_lr_for_tracking = self.optimizer.param_groups[0]['lr'] 
                    if verbose and (self.stats["episodes"] % (self.batch_size * 50) == 0 or not hasattr(self, '_scheduler_finished_warning_printed')): 
                        print(f"Scheduler cycle completed (total_steps={self.scheduler.total_steps}). Continuing with LR: {current_lr_for_tracking:.2e}")
                        self._scheduler_finished_warning_printed = True 
            else:
                current_lr_for_tracking = self.optimizer.param_groups[0]['lr']

            avg_math_loss_this_batch = 0 
            avg_code_loss_this_batch = 0
            # To track type-specific loss, you'd need to calculate per-item loss and then average based on task_types.
            # This is left as a future enhancement to avoid over-complicating this core change.

        self.stats["episodes"] += num_valid_items_for_loss 
        self.stats["total_answers"] += num_valid_items_for_loss 
        if num_valid_items_for_loss > 0 : 
            self.stats["batch_losses"].append(batch_total_loss)
            self.stats["learning_rates"].append(current_lr_for_tracking)
            self.stats["current_batch_sizes"].append(self.batch_size) 

        if not self.dataset_path and len(self.stats["batch_losses"]) > 0 : 
            avg_recent_loss = np.mean(self.stats["batch_losses"][-10:])
            if not np.isnan(avg_recent_loss):
                if avg_recent_loss < 0.5 and len(self.stats["batch_losses"]) >=5 : self.task_generator.adjust_difficulty(0.8) 
                elif avg_recent_loss > 2.0 and len(self.stats["batch_losses"]) >=5 : self.task_generator.adjust_difficulty(0.2)
        
        return {
            "batch_size": num_valid_items_for_loss, 
            "batch_loss": batch_total_loss if num_valid_items_for_loss > 0 else 0.0,
            "math_loss_in_batch": avg_math_loss_this_batch,
            "code_loss_in_batch": avg_code_loss_this_batch,
            "difficulty": self.task_generator.current_difficulty,
            "current_lr": current_lr_for_tracking
        }

    def train(self, 
              num_epochs: int = 1, 
              save_interval_batches: int = 20, 
              verbose: bool = True,
              dynamic_batch_adjustment_interval_batches: int = 10 
              ) -> Dict[str, Any]:
        
        if self.device == "cuda":
            torch.cuda.empty_cache() 
        
        if self.device == "cuda":
            self.batch_size = self.get_optimal_batch_size()
            print(f"Initial optimal batch size set to: {self.batch_size}")
        
        print(f"Starting optimized training for {num_epochs} epochs with current batch size {self.batch_size}...")
        if self.dataset_path:
            print(f"Training from dataset: {self.dataset_path}")
            self._load_full_dataset_once() 
            if not self.full_dataset_cache: print("Failed to load dataset. Aborting."); return {}
        else: print("Training with dynamically generated tasks.")
        
        start_time = time.time()
        total_batches_processed_effectively = 0 
        
        if self.dataset_path and self.full_dataset_cache and self.initial_batch_size > 0:
            estimated_batches_this_run_per_epoch = (len(self.full_dataset_cache) + self.initial_batch_size - 1) // self.initial_batch_size if self.initial_batch_size > 0 else 200
        elif not self.dataset_path and self.batch_size > 0: 
            estimated_batches_this_run_per_epoch = 200 
        else: 
            estimated_batches_this_run_per_epoch = 100 

        if self.scheduler is None and self.total_steps_for_scheduler is None:
             self._configure_scheduler_if_needed(num_epochs, estimated_batches_this_run_per_epoch)
        elif self.scheduler is None and self.total_steps_for_scheduler is not None: 
             self._configure_scheduler_if_needed(num_epochs, self.total_steps_for_scheduler // num_epochs if num_epochs > 0 else self.total_steps_for_scheduler)


        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            if self.device == "cuda" and epoch > 1 : 
                new_bs = self.get_optimal_batch_size()
                if new_bs != self.batch_size:
                    print(f"Epoch {epoch}: Optimal batch size adjusted from {self.batch_size} to {new_bs}")
                    self.batch_size = new_bs
            
            if self.dataset_path:
                 self.cache_tasks_for_epoch()
            else:
                 num_dynamic_tasks_for_epoch = self.batch_size * save_interval_batches * 5 
                 self.cache_tasks_for_epoch(num_dynamic_tasks_to_generate=num_dynamic_tasks_for_epoch)

            if not self.task_cache: print(f"No tasks for epoch {epoch}. Skipping."); continue

            num_batches_this_epoch = (len(self.task_cache) + self.batch_size - 1) // self.batch_size
            
            if self.scheduler is None and not self.dataset_path: 
                self._configure_scheduler_if_needed(num_epochs, num_batches_this_epoch)

            for batch_idx in range(num_batches_this_epoch):
                if self.device == "cuda" and (total_batches_processed_effectively + 1) % dynamic_batch_adjustment_interval_batches == 0 and total_batches_processed_effectively > 0:
                    new_bs = self.get_optimal_batch_size()
                    if new_bs != self.batch_size:
                        print(f"Batch {total_batches_processed_effectively + 1}: Optimal batch size adjusted from {self.batch_size} to {new_bs}")
                        self.batch_size = new_bs
                
                batch_start_time = time.time()
                current_batch_tasks_data = [self.get_next_task_from_cache() for _ in range(self.batch_size) if self.task_cache]
                if not current_batch_tasks_data: break 
                
                # Filter out any None values if _prepare_single_qa_pair_for_lm_training returned None for some items
                current_batch_tasks_data = [item for item in current_batch_tasks_data if item is not None]

                batch_metrics = self.train_batch(current_batch_tasks_data, epoch, batch_idx, verbose) 
                
                if batch_metrics["batch_size"] > 0: 
                    total_batches_processed_effectively += 1
                
                batch_time = time.time() - batch_start_time
                self.stats["batch_times"].append(batch_time)
                
                if self.device == "cuda":
                    current_memory_gb = torch.cuda.memory_allocated() / 1e9
                    max_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    self.stats["memory_usage"].append(max_memory_gb)

                if verbose:
                    elapsed_time = time.time() - start_time
                    avg_batch_loss_overall = np.mean(self.stats["batch_losses"]) if self.stats["batch_losses"] else 0
                    current_lr_display = batch_metrics.get('current_lr', self.optimizer.param_groups[0]['lr'])
                    
                    print(f"Ep {epoch}|B {batch_idx+1}/{num_batches_this_epoch}|BS {self.batch_size}|Items {self.stats['episodes']}|VldInB {batch_metrics['batch_size']}/{len(current_batch_tasks_data)}|T {elapsed_time:.1f}s|BT {batch_time:.2f}s|Loss(B) {batch_metrics['batch_loss']:.3f}|AvgL(A) {avg_batch_loss_overall:.3f}|LR {current_lr_display:.1e}|Diff {batch_metrics['difficulty']:.2f}")
                    if self.device == "cuda":
                        max_vram_overall = max(self.stats['memory_usage']) if self.stats['memory_usage'] else 0
                        print(f"VRAM: Cur {current_memory_gb:.2f}GB, MaxPeak {max_vram_overall:.2f}GB")
                
                if total_batches_processed_effectively > 0 and total_batches_processed_effectively % save_interval_batches == 0 :
                    checkpoint_path = f"{self.checkpoint_dir}/bitzero_optimized_{self.model_size}_epoch{epoch}_effbatch{total_batches_processed_effectively}.pt"
                    self.save_checkpoint(checkpoint_path)
            
            if self.dataset_path: print(f"Finished epoch {epoch} using data from {self.dataset_path}.")
            self.stats["current_epoch_for_plot"] = epoch 
            self._plot_training_progress()
        
        final_stats = {
            "total_items_processed": self.stats["episodes"],
            "final_difficulty_dynamic_gen": self.task_generator.current_difficulty,
            "training_time_seconds": time.time() - start_time,
            "avg_batch_time_seconds": np.mean(self.stats["batch_times"]) if self.stats["batch_times"] else 0,
            "avg_loss_overall": np.mean(self.stats["batch_losses"]) if self.stats["batch_losses"] else float('inf'),
            "avg_math_loss_overall": np.mean(self.stats["math_batch_losses"]) if self.stats["math_batch_losses"] else float('inf'),
            "avg_code_loss_overall": np.mean(self.stats["code_batch_losses"]) if self.stats["code_batch_losses"] else float('inf'),
            "final_learning_rate": self.optimizer.param_groups[0]['lr'],
            "final_batch_size_used": self.batch_size,
            "quantization_stats": self.model.get_quantization_stats()
        }
        if self.device == "cuda" and self.stats["memory_usage"]:
            final_stats["max_vram_usage_gb"] = max(self.stats["memory_usage"]) if self.stats["memory_usage"] else 0
        
        print(f"\nTraining completed in {final_stats['training_time_seconds']:.2f}s")
        print(f"Average training loss: {final_stats['avg_loss_overall']:.4f}")
        if final_stats['avg_batch_time_seconds'] > 0: print(f"Average batch time: {final_stats['avg_batch_time_seconds']:.2f}s")
        
        final_checkpoint_path = f"{self.checkpoint_dir}/bitzero_optimized_{self.model_size}_final_items{self.stats['episodes']}.pt"
        self.save_checkpoint(final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")
        return final_stats

    def save_checkpoint(self, checkpoint_path: str):
        model_device = next(self.model.parameters()).device
        self.model.cpu() 

        # --- START OF PROPOSED CHANGE: Limit stats history ---
        HISTORY_LENGTH = 500 # Keep data for the last 500 batches/items for plots/summary

        for key in ["rewards", "difficulties", "batch_times", "memory_usage", 
                     "batch_losses", "learning_rates", "current_batch_sizes",
                     "math_batch_losses", "code_batch_losses"]:
            if key in self.stats and isinstance(self.stats[key], list):
                self.stats[key] = self.stats[key][-HISTORY_LENGTH:]
        # --- END OF PROPOSED CHANGE ---

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None, 
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None, 
            "stats": self.stats, 
            "task_generator_difficulty": self.task_generator.current_difficulty,
            "initial_batch_size_config": self.initial_batch_size, 
            "tokenizer_config": { 
                "VOCAB_SIZE": VOCAB_SIZE, "CHAR_TO_ID_items": list(CHAR_TO_ID.items()), 
                "ID_TO_CHAR_items": list(ID_TO_CHAR.items()), "PAD_TOKEN": PAD_TOKEN, 
                "UNK_TOKEN": UNK_TOKEN, "BOS_TOKEN": BOS_TOKEN, "EOS_TOKEN": EOS_TOKEN 
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        self.model.to(model_device) 
        
        stats_summary_path = checkpoint_path.replace(".pt", "_stats_summary.json")
        avg_loss_overall = np.mean(self.stats["batch_losses"]) if self.stats["batch_losses"] else float('inf')
        summary_stats_to_save = {
            "items_processed": self.stats["episodes"], 
            "final_difficulty_dynamic_gen": self.task_generator.current_difficulty,
            "avg_loss_overall": avg_loss_overall,
            "rewards_last_100_items": self.stats["rewards"][-100:], 
            "losses_last_100_batches": self.stats["batch_losses"][-100:],
            "learning_rates_last_100_batches": self.stats["learning_rates"][-100:], 
            "batch_sizes_last_100_batches": self.stats["current_batch_sizes"][-100:], 
            "batch_times_last_100": self.stats["batch_times"][-100:],
            "quantization_stats": self.model.get_quantization_stats()
        }
        if self.device == "cuda" and self.stats["memory_usage"]:
            memory_usage_to_report = self.stats["memory_usage"][-100:] if len(self.stats["memory_usage"]) >=100 else self.stats["memory_usage"]
            summary_stats_to_save["max_vram_usage_gb_last_100_batches"] = max(memory_usage_to_report) if memory_usage_to_report else 0
        
        with open(stats_summary_path, "w") as f: json.dump(summary_stats_to_save, f, indent=2)
        print(f"Stats summary saved to {stats_summary_path}")

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "tokenizer_config" in checkpoint:
                loaded_vocab_size = checkpoint["tokenizer_config"].get("VOCAB_SIZE")
                if loaded_vocab_size and VOCAB_SIZE != loaded_vocab_size:
                    print(f"CRITICAL WARNING: Current VOCAB_SIZE ({VOCAB_SIZE}) differs from checkpoint's ({loaded_vocab_size}).")
            else: print("WARNING: Checkpoint missing tokenizer_config.")
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.initial_batch_size = checkpoint.get("initial_batch_size_config", self.initial_batch_size)
            self.batch_size = self.initial_batch_size 
            
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                if self.scheduler is None and self.total_steps_for_scheduler is not None and self.total_steps_for_scheduler > 0:
                     self._configure_scheduler_if_needed(self.num_epochs_for_scheduler, self.total_steps_for_scheduler // self.num_epochs_for_scheduler if self.num_epochs_for_scheduler > 0 else self.total_steps_for_scheduler)
                
                if self.scheduler: 
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("Scheduler state loaded.")
                else:
                    print("Scheduler state found in checkpoint, but scheduler not configured (likely due to missing total_steps). Will re-initialize at train start.")

            if self.scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            self.stats = checkpoint.get("stats", self.stats) 
            if "current_batch_sizes" not in self.stats: self.stats["current_batch_sizes"] = []

            self.task_generator.current_difficulty = checkpoint.get("task_generator_difficulty", self.task_generator.current_difficulty)
            print("Checkpoint loaded successfully")
        except FileNotFoundError: print(f"Error: Checkpoint file not found at {checkpoint_path}.")
        except Exception as e: print(f"Error loading checkpoint: {e}.");

    def _plot_training_progress(self):
        if not self.stats["batch_losses"]: 
            print("No batch loss data to plot.")
            return

        num_loss_points = len(self.stats["batch_losses"])
        if num_loss_points == 0:
            print("No batch loss data points to plot.")
            return

        num_subplots = 1 
        if self.stats["math_batch_losses"] and any(self.stats["math_batch_losses"]): num_subplots += 1
        if self.stats["code_batch_losses"] and any(self.stats["code_batch_losses"]): num_subplots += 1
        if self.stats["batch_times"] and any(self.stats["batch_times"]): num_subplots += 1
        if self.stats["learning_rates"] and any(self.stats["learning_rates"]): num_subplots +=1 
        if self.stats["current_batch_sizes"] and any(self.stats["current_batch_sizes"]): num_subplots +=1 


        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharex=False)
        if num_subplots == 1: axes = [axes] 
        
        ax_idx = 0

        # Plot Overall Loss
        current_ax = axes[ax_idx]; ax_idx +=1
        batch_indices_overall_loss = list(range(1, num_loss_points + 1))
        current_ax.plot(batch_indices_overall_loss, self.stats["batch_losses"], 'b-', alpha=0.7, label="Per-Batch Loss (Overall)")
        if num_loss_points >= 10:
            window_size_loss = min(10, num_loss_points)
            moving_avg_overall_loss = np.convolve(self.stats["batch_losses"], np.ones(window_size_loss)/window_size_loss, mode='valid')
            current_ax.plot(batch_indices_overall_loss[window_size_loss-1:], moving_avg_overall_loss, 'r-', linewidth=2, label=f"{window_size_loss}-batch Moving Avg (Overall)")
        current_ax.set_xlabel('Batch Index'); current_ax.set_ylabel('Loss')
        current_ax.set_title(f'Overall Training Batch Loss (Epoch {self.stats.get("current_epoch_for_plot", "N/A")})')
        current_ax.legend(); current_ax.grid(True)
        
        # Plot Math Loss
        if self.stats["math_batch_losses"] and any(self.stats["math_batch_losses"]) and ax_idx < len(axes):
            current_ax = axes[ax_idx]; ax_idx +=1
            num_math_loss_points = len(self.stats["math_batch_losses"])
            math_loss_indices = list(range(1, num_math_loss_points + 1))
            current_ax.plot(math_loss_indices, self.stats["math_batch_losses"], 'g-', alpha=0.7, label="Math Batch Loss")
            if num_math_loss_points >= 10:
                window_size_math_loss = min(10, num_math_loss_points)
                moving_avg_math_loss = np.convolve(self.stats["math_batch_losses"], np.ones(window_size_math_loss)/window_size_math_loss, mode='valid')
                current_ax.plot(math_loss_indices[window_size_math_loss-1:], moving_avg_math_loss, 'darkgreen', linewidth=2, label=f"{window_size_math_loss}-batch Moving Avg (Math)")
            current_ax.set_xlabel('Batch Index (math tasks)'); current_ax.set_ylabel('Loss')
            current_ax.set_title('Math Task Batch Loss'); current_ax.legend(); current_ax.grid(True)

        # Plot Code Loss
        if self.stats["code_batch_losses"] and any(self.stats["code_batch_losses"]) and ax_idx < len(axes):
            current_ax = axes[ax_idx]; ax_idx +=1
            num_code_loss_points = len(self.stats["code_batch_losses"])
            code_loss_indices = list(range(1, num_code_loss_points + 1))
            current_ax.plot(code_loss_indices, self.stats["code_batch_losses"], 'm-', alpha=0.7, label="Code Batch Loss")
            if num_code_loss_points >= 10:
                window_size_code_loss = min(10, num_code_loss_points)
                moving_avg_code_loss = np.convolve(self.stats["code_batch_losses"], np.ones(window_size_code_loss)/window_size_code_loss, mode='valid')
                current_ax.plot(code_loss_indices[window_size_code_loss-1:], moving_avg_code_loss, 'purple', linewidth=2, label=f"{window_size_code_loss}-batch Moving Avg (Code)")
            current_ax.set_xlabel('Batch Index (code tasks)'); current_ax.set_ylabel('Loss')
            current_ax.set_title('Code Task Batch Loss'); current_ax.legend(); current_ax.grid(True)
        
        # Plot Batch Times
        if self.stats["batch_times"] and any(self.stats["batch_times"]) and ax_idx < len(axes): 
            current_ax = axes[ax_idx]; ax_idx +=1
            batch_indices_times = list(range(1, len(self.stats["batch_times"]) + 1))
            current_ax.plot(batch_indices_times, self.stats["batch_times"], 'c-')
            current_ax.set_xlabel('Batch Index'); current_ax.set_ylabel('Time (s)')
            current_ax.set_title('Batch Processing Time'); current_ax.grid(True)
        
        # Plot Learning Rates
        if self.stats["learning_rates"] and any(self.stats["learning_rates"]) and ax_idx < len(axes):
            current_ax = axes[ax_idx]; ax_idx +=1
            lr_indices = list(range(1, len(self.stats["learning_rates"]) + 1))
            current_ax.plot(lr_indices, self.stats["learning_rates"], 'y-')
            current_ax.set_xlabel('Batch Index'); current_ax.set_ylabel('Learning Rate')
            current_ax.set_title('Learning Rate Schedule'); current_ax.grid(True)
            
        # Plot Dynamic Batch Sizes
        if self.stats["current_batch_sizes"] and any(self.stats["current_batch_sizes"]) and ax_idx < len(axes):
            current_ax = axes[ax_idx]; ax_idx +=1
            bs_indices = list(range(1, len(self.stats["current_batch_sizes"]) + 1))
            current_ax.plot(bs_indices, self.stats["current_batch_sizes"], 'p-', color='orange', label="Batch Size Used") 
            current_ax.set_xlabel('Batch Index')
            current_ax.set_ylabel('Batch Size')
            current_ax.set_title('Effective Batch Size Over Training')
            current_ax.legend(); current_ax.grid(True)


        plt.tight_layout()
        epoch_for_filename = self.stats.get("current_epoch_for_plot", "final") 
        plot_save_path = f"{self.log_dir}/training_progress_optimized_epoch{epoch_for_filename}.png"
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Training progress plot saved to {plot_save_path}")

def main():
    parser = argparse.ArgumentParser(description="BitZero Optimized Training with Optional Dataset")
    parser.add_argument("--model_size", type=str, default="nano", choices=["nano", "micro"], help="Model size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu). Auto-detects if None.")
    parser.add_argument("--initial_batch_size", type=int, default=8, help="Initial batch size (can be dynamic on CUDA)")
    parser.add_argument("--max_dynamic_batch_size", type=int, default=32, help="Max cap for dynamic batch sizing on CUDA")
    parser.add_argument("--vram_per_sample_gb", type=float, default=0.2, help="Estimated VRAM in GB per sample for dynamic batch sizing") 
    parser.add_argument("--dynamic_bs_adjust_interval", type=int, default=10, help="Adjust dynamic batch size every N batches")


    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Max learning rate for scheduler") 
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value (0 to disable)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to .jsonl dataset file.")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no_parallel_tasks", action="store_true", help="Disable parallel dynamic task generation")
    parser.add_argument("--save_interval_batches", type=int, default=20, help="Save checkpoint every N batches")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output per batch")
    parser.add_argument("--critical_ratio", type=float, default=0.1, help="Critical ratio for hybrid precision model (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    selected_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = OptimizedTrainer(
        model_size=args.model_size, device=selected_device, 
        initial_batch_size=args.initial_batch_size, 
        max_dynamic_batch_size=args.max_dynamic_batch_size, 
        vram_per_sample_gb_estimate=args.vram_per_sample_gb, 
        learning_rate=args.learning_rate, max_seq_len=args.max_seq_len, 
        use_mixed_precision=not args.no_mixed_precision, 
        use_gradient_checkpointing=not args.no_gradient_checkpointing, 
        gradient_clip_val=args.gradient_clip_val, 
        use_parallel_tasks=not args.no_parallel_tasks, dataset_path=args.dataset_path,
        critical_ratio_for_model=args.critical_ratio,
        num_epochs_for_scheduler=args.epochs, 
        weight_decay_for_adamw=args.weight_decay,
        verbose=args.verbose # Pass the verbose flag
    )
    
    if args.checkpoint: 
        trainer.load_checkpoint(args.checkpoint)
    
    trainer.train(
        num_epochs=args.epochs, 
        save_interval_batches=args.save_interval_batches, 
        verbose=args.verbose,
        dynamic_batch_adjustment_interval_batches=args.dynamic_bs_adjust_interval 
    )

if __name__ == "__main__":
    main()
