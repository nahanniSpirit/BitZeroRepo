# tokenizer_config.py
# This file defines the vocabulary and special tokens for the BitZero model.
from typing import Dict, List, Tuple, Optional, Union, Any
import math

# --- Vocabulary Definition ---
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>" # End of Sequence/Answer
BOS_TOKEN = "<BOS>" # Beginning of Sequence/Answer

# Define your character set for code and math
# This set should include all characters you expect in your tasks
# and that you want the model to be able to generate.
# Added '$', '?', and '\' to the character set.
PYTHON_CODE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()[]{}+-*/=<>.,:;'\"% !&|^~#\t\n$\\?\\" # Added ? and \
EMOTIONAL_CHARS = "😀😃😄😊😉😍🥰😘🥲🤔😟🙁☹️😮😱😠😡🥳🎉👍👎❤️💔" 
EXTENDED_CHARS = PYTHON_CODE_CHARS + EMOTIONAL_CHARS

# Build vocabulary
# Ensure PAD_TOKEN is ID 0 if your padding logic relies on it.
# The order of special tokens here determines their initial IDs.
_initial_char_vocab_list = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + sorted(list(set(EXTENDED_CHARS)))


# --- Subword Tokenization for Common Patterns ---
COMMON_SUBWORDS = [
    "def ", "return ", "import ", "class ", "for ", "if ", "else:", "elif ", 
    "while ", "try:", "except:", "finally:", "with ", "print(", "self.", 
    "==", "!=", ">=", "<=", "+=", "-=", "*=", "/=", "**", "//", "->", "=>",
    "True", "False", "None"
]

# SUBWORD_BASE_ID is the index where common subwords will start in the full VOCAB_LIST
# This is calculated before COMMON_SUBWORDS are added to VOCAB_LIST.
SUBWORD_BASE_ID = len(_initial_char_vocab_list)

# VOCAB_LIST is the single source of truth for token order and IDs.
# It's formed by initial characters/special tokens, then common subwords.
# Dynamically added tokens will be appended later by update_vocabulary_from_usage_statistics.
VOCAB_LIST = _initial_char_vocab_list + COMMON_SUBWORDS

# Build initial CHAR_TO_ID, ID_TO_CHAR, and VOCAB_SIZE from the comprehensive VOCAB_LIST
CHAR_TO_ID = {token: i for i, token in enumerate(VOCAB_LIST)}
ID_TO_CHAR = {i: token for i, token in enumerate(VOCAB_LIST)}
VOCAB_SIZE = len(VOCAB_LIST)

def tokenize_with_subwords(text: str, subwords: List[str], char_to_id_map: Dict[str, int], subword_base_id: int, unk_token_id: int) -> List[int]:
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        # Check for longest subword match first (Note: current COMMON_SUBWORDS are not sorted by length for optimal matching)
        # For this specific list, iterating as is might be acceptable.
        # A more robust approach would sort `subwords` by length descending if they varied greatly.
        for idx, subword_pattern in enumerate(subwords): # Renamed `subword` to `subword_pattern` to avoid conflict
            if text[i:i+len(subword_pattern)] == subword_pattern:
                tokens.append(subword_base_id + idx) # Uses the pre-calculated SUBWORD_BASE_ID
                i += len(subword_pattern)
                matched = True
                break
        if not matched:
            tokens.append(char_to_id_map.get(text[i], unk_token_id))
            i += 1
    return tokens

# --- End Subword Tokenization ---

# --- Dynamic Vocabulary Adaptation ---
def update_vocabulary_from_usage_statistics(usage_counts: Dict[str, int], threshold: int = 100):
    """Update vocabulary based on usage statistics.
    Appends new sequences to VOCAB_LIST and rebuilds mappings.
    Existing token IDs (including common subwords) are preserved relative to their initial order.
    """
    global VOCAB_LIST, CHAR_TO_ID, ID_TO_CHAR, VOCAB_SIZE
    
    newly_added_to_list = False
    for sequence, count in usage_counts.items():
        if count > threshold and sequence not in CHAR_TO_ID: # Check current CHAR_TO_ID for existence
            VOCAB_LIST.append(sequence)
            newly_added_to_list = True
            print(f"Adding new sequence to vocab: \'{sequence}\'")
    
    if newly_added_to_list:
        # Rebuild mappings from the updated VOCAB_LIST
        CHAR_TO_ID = {token: i for i, token in enumerate(VOCAB_LIST)}
        ID_TO_CHAR = {i: token for i, token in enumerate(VOCAB_LIST)}
        VOCAB_SIZE = len(VOCAB_LIST)
        print(f"Vocabulary dynamically updated. New VOCAB_SIZE: {VOCAB_SIZE}")

# --- End Dynamic Vocabulary Adaptation ---


if __name__ == '__main__':
    print(f"Initial VOCAB_SIZE (chars + special): {len(_initial_char_vocab_list)}")
    print(f"SUBWORD_BASE_ID: {SUBWORD_BASE_ID}")
    print(f"Number of COMMON_SUBWORDS: {len(COMMON_SUBWORDS)}")
    print(f"Final VOCAB_SIZE (after adding subwords): {VOCAB_SIZE}")
    
    print(f"PAD Token ID: {CHAR_TO_ID[PAD_TOKEN]}")
    print(f"UNK Token ID: {CHAR_TO_ID[UNK_TOKEN]}")
    print(f"BOS Token ID: {CHAR_TO_ID[BOS_TOKEN]}")
    print(f"EOS Token ID: {CHAR_TO_ID[EOS_TOKEN]}")
    
    sample_text = "def example(x, path$='C:\\test\\?'):\n  return x + 1"
    print(f"\nOriginal text: '{sample_text}'")
    
    # Basic character-only tokenization test
    tokenized_ids_char_only = [CHAR_TO_ID.get(BOS_TOKEN)]
    for char_val in sample_text: 
        tokenized_ids_char_only.append(CHAR_TO_ID.get(char_val, CHAR_TO_ID[UNK_TOKEN]))
    tokenized_ids_char_only.append(CHAR_TO_ID.get(EOS_TOKEN))
    print(f"Tokenized IDs (char only): {tokenized_ids_char_only}")

    # Basic character-only detokenization test
    detokenized_chars = []
    for token_id in tokenized_ids_char_only:
        char_val = ID_TO_CHAR.get(token_id, UNK_TOKEN)
        if char_val == EOS_TOKEN or char_val == PAD_TOKEN:
            break
        if char_val != BOS_TOKEN:
             detokenized_chars.append(char_val)
    detokenized_text_char_only = "".join(detokenized_chars) # Corrected variable name
    print(f"Detokenized text (char only): '{detokenized_text_char_only}'")
    assert detokenized_text_char_only == sample_text, "Character-only detokenization mismatch!"

    # Verify that all characters in EXTENDED_CHARS are in the vocab
    for char_test in EXTENDED_CHARS:
        assert char_test in CHAR_TO_ID, f"Character '{char_test}' not in CHAR_TO_ID!"
        assert ID_TO_CHAR[CHAR_TO_ID[char_test]] == char_test, f"Mismatch for char '{char_test}'"
    print("\nVocabulary integrity check passed.")

    print("\n--- Testing Subword Tokenization ---")
    subword_sample_text = "def my_func(self, x):\n  if x == True:\n    return x + 10"
    print(f"Subword sample text: '{subword_sample_text}'")
    
    subword_tokenized_ids = [CHAR_TO_ID.get(BOS_TOKEN)] + tokenize_with_subwords(
        subword_sample_text, 
        COMMON_SUBWORDS, 
        CHAR_TO_ID, 
        SUBWORD_BASE_ID, 
        CHAR_TO_ID[UNK_TOKEN]
    ) + [CHAR_TO_ID.get(EOS_TOKEN)]
    print(f"Subword tokenized IDs: {subword_tokenized_ids}")

    detokenized_subword_text_parts = []
    for token_id in subword_tokenized_ids:
        char_or_subword = ID_TO_CHAR.get(token_id, UNK_TOKEN)
        if char_or_subword == EOS_TOKEN or char_or_subword == PAD_TOKEN:
            break
        if char_or_subword != BOS_TOKEN:
            detokenized_subword_text_parts.append(char_or_subword)
    detokenized_subword_text = "".join(detokenized_subword_text_parts)
    print(f"Subword detokenized text: '{detokenized_subword_text}'")
    assert detokenized_subword_text == subword_sample_text, "Subword detokenization mismatch!"
    print("Subword tokenization and detokenization test passed.")

    print("\n--- Testing Dynamic Vocabulary Adaptation ---")
    mock_usage_counts = {
        "new_keyword_sequence": 150,
        "another_common_pattern": 200,
        "less_common": 50,
        "def ": 5000 # Existing common subword, should not be re-added
    }
    print(f"Initial VOCAB_SIZE before dynamic update: {VOCAB_SIZE}")
    update_vocabulary_from_usage_statistics(mock_usage_counts, threshold=100)
    print(f"Final VOCAB_SIZE after dynamic update: {VOCAB_SIZE}")
    assert "new_keyword_sequence" in CHAR_TO_ID
    assert "another_common_pattern" in CHAR_TO_ID
    assert "less_common" not in CHAR_TO_ID # Below threshold
    print(f"ID for 'new_keyword_sequence': {CHAR_TO_ID['new_keyword_sequence']}")
    print("Dynamic vocabulary adaptation test passed.")

