import torch
import argparse
import os
# import json # No longer needed for chat history directly if using MemoryManager
import time
from typing import Dict, List, Tuple, Optional, Any # Added Tuple and Optional
# Ensure this matches how models are created and configured in your training script
from hybrid_precision import create_bitzero_nano_hybrid, HybridPrecisionTransformer 
from memory_manager import MemoryManager # **** NEW IMPORT ****

try:
    from tokenizer_config import (
        VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR,
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
    )
except ImportError:
    print("ERROR: tokenizer_config.py not found. Chat script requires it.")
    print("Please ensure tokenizer_config.py is in the same directory or PYTHONPATH.")
    # Minimal fallback to prevent immediate crash, assuming chat won't work correctly
    VOCAB_SIZE = 99 # Default to new vocab size to match potential new checkpoints
    CHAR_TO_ID = {} # Needs proper definition
    ID_TO_CHAR = {} # Needs proper definition
    PAD_TOKEN = "<PAD>" ; CHAR_TO_ID[PAD_TOKEN]=0
    UNK_TOKEN = "<UNK>" ; CHAR_TO_ID[UNK_TOKEN]=1
    BOS_TOKEN = "<BOS>" ; CHAR_TO_ID[BOS_TOKEN]=2
    EOS_TOKEN = "<EOS>" ; CHAR_TO_ID[EOS_TOKEN]=3
    ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()} # Basic fallback
    # This fallback is very basic and primarily for structure.
    # For actual operation, tokenizer_config.py is essential.


# Define the path for storing conversation history
# HISTORY_FILE = "bitzero_chat_history.jsonl" # Superseded by MemoryManager
# MAX_HISTORY_TURNS_FOR_CONTEXT = 10 # Superseded by MemoryManager
MAX_HISTORY_TURNS_FOR_RAW_PROMPT_CONTEXT = 5 # Keep some raw recent turns for prompt

class BitZeroChat:
    def __init__(self, checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu", max_seq_len=768, db_path="bitzero_memory.db", critical_ratio_for_model: float = 1.0): # Add param with default
        self.device = device
        self.max_seq_len = max_seq_len # Store max_seq_len
        self.critical_ratio_for_model = critical_ratio_for_model # Store if needed
        print(f"Initializing BitZero Chat on {device} with critical_ratio: {self.critical_ratio_for_model}...") # Add print

        pad_token_id_for_model = CHAR_TO_ID.get(PAD_TOKEN)
        if pad_token_id_for_model is None: 
            print(f"Warning: PAD_TOKEN ID not found in CHAR_TO_ID. Defaulting to 0 for model padding_idx.")
            pad_token_id_for_model = 0

        print(f"Model Creation - VOCAB_SIZE: {VOCAB_SIZE}, PAD_TOKEN_ID: {pad_token_id_for_model}, Critical Ratio: {self.critical_ratio_for_model}")
        self.model = create_bitzero_nano_hybrid(
            critical_ratio=self.critical_ratio_for_model, # Use it here
            vocab_size=VOCAB_SIZE, 
            pad_token_id=pad_token_id_for_model,
            max_seq_len=self.max_seq_len
        )
        self.model.to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if "tokenizer_config" in checkpoint:
                    ckpt_vocab_size = checkpoint["tokenizer_config"].get("VOCAB_SIZE")
                    if ckpt_vocab_size and ckpt_vocab_size != VOCAB_SIZE:
                        print(f"CRITICAL WARNING: Checkpoint VOCAB_SIZE ({ckpt_vocab_size}) differs from current tokenizer VOCAB_SIZE ({VOCAB_SIZE}).")
                    else:
                        print("Checkpoint tokenizer VOCAB_SIZE matches current configuration.")
                else:
                    print("WARNING: Checkpoint does not contain tokenizer_config. Assuming compatibility.")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Using initialized (untrained) model.")
        else:
            print("No checkpoint path provided or checkpoint not found. Using initialized (untrained) model.")

        self.model.eval()
        
        self.memory_manager = MemoryManager(db_path=db_path)
        self.user_id = "default_user" # Can be made more dynamic, e.g., from args or login
        self.conversation_id = self.memory_manager.get_or_create_conversation_session(self.user_id)

        print(f"BitZero Chat initialized and ready! Session ID: {self.conversation_id}")

    # Old file-based history methods (_load_history, _save_turn) are removed.

    def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
        """Tokenize text based on the defined vocabulary from tokenizer_config."""
        if not CHAR_TO_ID or (add_bos and BOS_TOKEN not in CHAR_TO_ID) or \
           (add_eos and EOS_TOKEN not in CHAR_TO_ID) or UNK_TOKEN not in CHAR_TO_ID or \
           PAD_TOKEN not in CHAR_TO_ID:
            print("CRITICAL ERROR IN CHAT TOKENIZER: Vocabulary not properly initialized.")
            # Return a tensor of zeros of expected shape to prevent downstream errors, though chat will be broken.
            return torch.zeros((1, self.max_seq_len), dtype=torch.long, device=self.device)

        token_ids = [CHAR_TO_ID[BOS_TOKEN]] if add_bos else []
        for char in text:
            token_ids.append(CHAR_TO_ID.get(char, CHAR_TO_ID[UNK_TOKEN]))
        if add_eos:
            token_ids.append(CHAR_TO_ID[EOS_TOKEN])

        # Truncation and Padding
        pad_id = CHAR_TO_ID[PAD_TOKEN]
        eos_id = CHAR_TO_ID[EOS_TOKEN]

        if len(token_ids) > self.max_seq_len:
            # Truncate, ensuring EOS is the last token if add_eos was true
            token_ids = token_ids[:self.max_seq_len-1] + [eos_id] if add_eos else token_ids[:self.max_seq_len]
        else:
            token_ids += [pad_id] * (self.max_seq_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Detokenize tensor of token IDs based on the defined vocabulary."""
        if not ID_TO_CHAR or EOS_TOKEN not in CHAR_TO_ID or PAD_TOKEN not in CHAR_TO_ID or \
           BOS_TOKEN not in CHAR_TO_ID or UNK_TOKEN not in CHAR_TO_ID:
            print("CRITICAL ERROR IN CHAT DETOKENIZER: Vocabulary not properly initialized.")
            return "[Detokenization Error]"
            
        text_chars = []
        # Squeeze if token_ids is [1, N], then convert to list
        token_ids_list = token_ids.squeeze().tolist() if token_ids.dim() > 1 else token_ids.tolist()

        eos_id = CHAR_TO_ID[EOS_TOKEN]
        pad_id = CHAR_TO_ID[PAD_TOKEN]
        bos_id = CHAR_TO_ID[BOS_TOKEN]

        for token_id_val in token_ids_list:
            if token_id_val == eos_id or token_id_val == pad_id:
                break
            if token_id_val != bos_id: # Don't append BOS to the string
                text_chars.append(ID_TO_CHAR.get(token_id_val, UNK_TOKEN))
        return "".join(text_chars)

    def generate_response(self, user_input, max_new_tokens=150, temperature=1.0, top_k=80) -> Tuple[str, Optional[int]]:
        # 1. Store user input in memory manager
        user_turn_id = self.memory_manager.add_turn_to_conversation(
            conversation_session_id=self.conversation_id, 
            user_id=self.user_id, 
            role="user", 
            content=user_input
        )

        # 3. Get rich context from MemoryManager
        # generate_context_from_memory now internally calls get_context_summary
        memory_context_str = self.memory_manager.generate_context_from_memory(
            user_input, 
            for_user_id=self.user_id, 
            conversation_id=self.conversation_id
        )
        
        # 4. Construct prompt
        recent_turns_for_prompt = self.memory_manager.get_recent_turns_for_prompt(
            conversation_session_id=self.conversation_id, 
            limit=MAX_HISTORY_TURNS_FOR_RAW_PROMPT_CONTEXT 
        )
        
        prompt_text_for_model_input = ""
        if memory_context_str:
            prompt_text_for_model_input += f"[System Memory Context]:\n{memory_context_str}\n\n"
        
        prompt_text_for_model_input += "--- Recent Conversation ---\n"
        # Iterate through recent_turns_for_prompt (which is already in chronological order from DB)
        for turn in recent_turns_for_prompt:
            role_label = "User" if turn["role"] == "user" else "BitZero"
            prompt_text_for_model_input += f"{role_label}: {turn['content']}\n"
        # The current user_input is already part of recent_turns if limit is high enough
        # or it's the first turn. If not, ensure it's added. 
        # The current logic in get_recent_turns_for_prompt fetches based on what's in DB.
        # The user_input that triggered this call is the *very latest* user turn.
        # It was just added. So get_recent_turns should include it if limit allows.
        # To be absolutely sure the current user input is the last thing before "BitZero: ",
        # we can construct it carefully. The current get_recent_turns_for_prompt returns in chrono order.
        
        # Re-construct prompt ensuring current user input is last before BitZero's turn        # This might duplicate the last user turn if already fetched by get_recent_turns_for_prompt
        # A cleaner way is to ensure get_recent_turns_for_prompt doesn't include the *absolute* last user turn
        # if we are going to append it manually, or just rely on it being there.
        # For now, let's assume get_recent_turns includes the latest user turn if limit is sufficient.
        prompt_text_for_model_input += f"BitZero: I think that " # Prompt for model's response with a starter phrase
        
        # Debug: Print the full prompt text being fed to the model
        print(f"\n--- DEBUG: Full Prompt Text for Model ---")
        print(f"'{prompt_text_for_model_input}'")  # Add quotes to see leading/trailing whitespace
        print(f"--- END DEBUG: Full Prompt Text ---")
        print(f"DEBUG: Length of prompt text: {len(prompt_text_for_model_input)}")
        
        generation_prompt_tokens = [CHAR_TO_ID.get(BOS_TOKEN)]
        for char in prompt_text_for_model_input: 
            generation_prompt_tokens.append(CHAR_TO_ID.get(char, CHAR_TO_ID.get(UNK_TOKEN)))
        
        if len(generation_prompt_tokens) >= self.max_seq_len:
            print(f"Warning: Prompt context is too long ({len(generation_prompt_tokens)} tokens), truncating.")
            keep_tokens_count = self.max_seq_len - 1 
            generation_prompt_tokens = [CHAR_TO_ID.get(BOS_TOKEN)] + generation_prompt_tokens[-keep_tokens_count+1:]

        current_input_ids = torch.tensor([generation_prompt_tokens], dtype=torch.long, device=self.device)
        
        generated_ids_for_response = []
        eos_token_id = CHAR_TO_ID.get(EOS_TOKEN)
        pad_token_id = CHAR_TO_ID.get(PAD_TOKEN)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if current_input_ids.size(1) >= self.max_seq_len:
                    print("Max sequence length reached for model input during generation.")
                    break
                logits = self.model(current_input_ids) 
                next_token_logits = logits[0, -1, :] 
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0 and top_k < VOCAB_SIZE: 
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                    next_token_id_tensor = top_k_indices[torch.multinomial(probs, num_samples=1)]
                else: 
                    next_token_id_tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                next_token_id_item = next_token_id_tensor.item()
                if next_token_id_item == eos_token_id or next_token_id_item == pad_token_id:
                    break
                generated_ids_for_response.append(next_token_id_item)
                current_input_ids = torch.cat([current_input_ids, next_token_id_tensor.view(1,1)], dim=1) 
          print(f"DEBUG: Raw generated token IDs for response: {generated_ids_for_response}")
        response = self.detokenize(torch.tensor(generated_ids_for_response, dtype=torch.long, device="cpu"))
        
        # Check for empty/whitespace-only responses
        if not response.strip():  # If the response is empty or only whitespace
            print("INFO: Model generated an empty/whitespace response. Not saving to history. Trying a fallback.")
            response = "I'm not sure how to respond to that. Can you try rephrasing?"
            # Or you could have it try to regenerate, but that might loop.
            # For now, a fixed fallback is safer for testing.
            # assistant_turn_id will still be generated for this fallback.
        
        # 5. Store model response
        assistant_turn_id = self.memory_manager.add_turn_to_conversation(
            conversation_session_id=self.conversation_id, 
            user_id=self.user_id, # Or a system user_id for assistant
            role="assistant", 
            content=response
        )

        # 6. Extract insights from model response (and potentially user input again)
        self.memory_manager.extract_and_store_memories_from_interaction(
            user_id=self.user_id, 
            conversation_id=self.conversation_id,
            user_text=user_input, 
            assistant_text=response
        )
        
        return response, assistant_turn_id # Return assistant_turn_id for feedback

    def chat(self):
        print("Welcome to BitZero Chat! Type 'exit' to end the conversation.")
        print(f"Current Session ID: {self.conversation_id}")
        print("=" * 50)
        
        initial_context_summary = self.memory_manager.get_context_summary(self.user_id, self.conversation_id)
        if initial_context_summary:
            print("\n--- Initial Context from Memory ---")
            print(initial_context_summary)
            print("--- End Context ---\n")
        else: 
            recent_history_display = self.memory_manager.get_recent_turns_for_prompt(self.conversation_id, limit=3)
            if recent_history_display:
                print("\n--- Recent History (from current session) ---")
                for turn in recent_history_display:
                     role_label = "You" if turn["role"] == "user" else "BitZero"
                     print(f"{role_label}: {turn['content']}")
                print("--- End History ---\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                self.memory_manager.end_conversation_session(self.conversation_id)
                print("BitZero: Goodbye!")
                break
            if not user_input.strip(): continue

            print("Thinking...")
            start_time = time.time()
            response, assistant_turn_id = self.generate_response(user_input)
            elapsed_time = time.time() - start_time
            
            print(f"BitZero ({elapsed_time:.2f}s): {response}")
            
            # Collect feedback
            if assistant_turn_id is not None:
                feedback_input = input("Feedback (+/-/text, or press Enter to skip): ").strip()
                if feedback_input:
                    score = 0
                    text_feedback = None
                    if feedback_input == '+': 
                        score = 1
                        print("Feedback received: Positive")
                    elif feedback_input == '-': 
                        score = -1
                        print("Feedback received: Negative")
                    else: 
                        text_feedback = feedback_input
                        # Score remains 0 for purely textual feedback unless combined with +/- later
                        print(f"Feedback received: '{text_feedback}'")
                    
                    self.memory_manager.store_interaction_feedback(
                        conversation_id=self.conversation_id, 
                        turn_id=assistant_turn_id, 
                        user_id=self.user_id, 
                        feedback_score=score, 
                        feedback_text=text_feedback
                    )
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="BitZero Chat Interface")
    # Suggest using a newer checkpoint by default if available from optimized_training
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/bitzero_optimized_nano_final_items279.pt", 
                        help="Path to model checkpoint trained with the new tokenizer (e.g., VOCAB_SIZE=99)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Max sequence length for chat model and tokenizer")
    parser.add_argument("--db_path", type=str, default="bitzero_memory.db", help="Path to the SQLite database for memory.") # New arg for DB path
    parser.add_argument("--critical_ratio", type=float, default=1.0, 
                        help="Critical ratio for the model (default 1.0 for full precision checkpoints)")
    
    args = parser.parse_args()

    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Model will use initial (untrained) weights.")
        # If default checkpoint dir doesn't exist, create it to avoid error on first run if no checkpoints folder yet
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if checkpoint_dir and not os.path.isdir(checkpoint_dir): # Ensure checkpoint_dir is not empty
             os.makedirs(checkpoint_dir, exist_ok=True)
             print(f"Created directory: {checkpoint_dir}")

    chat_interface = BitZeroChat(
        checkpoint_path=args.checkpoint if (args.checkpoint and os.path.exists(args.checkpoint)) else None, 
        device=args.device,
        max_seq_len=args.max_seq_len,
        db_path=args.db_path, # Pass db_path to constructor
        critical_ratio_for_model=args.critical_ratio # Pass it
    )
    chat_interface.chat()

if __name__ == "__main__":
    main()
