# vocab_checker.py
import json

# --- Paste the PYTHON_CODE_CHARS string from your tokenizer_config_v2.py here ---
# Make sure this is IDENTICAL to the one in your tokenizer_config_v2.py
TOKENIZER_PYTHON_CODE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()[]{}+-*/=<>.,:;'\"% !&|^~#\t\n$\\?\\"
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def check_dataset_vocabulary(dataset_path: str, known_chars_str: str):
    """
    Checks all characters in the dataset against a known set of characters.

    Args:
        dataset_path (str): Path to the .jsonl dataset file.
        known_chars_str (str): A string containing all known characters in the vocabulary.
    """
    known_chars_set = set(known_chars_str)
    dataset_chars_set = set()
    
    print(f"Known characters in tokenizer: {sorted(list(known_chars_set))}")
    print(f"Length of known_chars_set: {len(known_chars_set)}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data_item = json.loads(line)
                    if "conversation" in data_item and isinstance(data_item["conversation"], list):
                        for turn in data_item["conversation"]:
                            if "content" in turn and isinstance(turn["content"], str):
                                dataset_chars_set.update(list(turn["content"]))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {i+1}")
                except Exception as e:
                    print(f"Warning: Error processing line {i+1}: {e}")
        
        print(f"\nUnique characters found in dataset '{dataset_path}': {sorted(list(dataset_chars_set))}")
        print(f"Total unique characters in dataset: {len(dataset_chars_set)}")

        missing_chars = dataset_chars_set - known_chars_set
        
        if not missing_chars:
            print("\nSUCCESS: All characters in the dataset are present in the tokenizer's PYTHON_CODE_CHARS.")
        else:
            print(f"\nWARNING: The following {len(missing_chars)} characters were found in the dataset but are MISSING from PYTHON_CODE_CHARS:")
            # Sort for consistent output
            print(f"Missing characters: {sorted(list(missing_chars))}")
            print("\nPlease add these characters to the PYTHON_CODE_CHARS string in your 'tokenizer_config.py' file and re-run training.")
            print("Example of adding: If '€' is missing, PYTHON_CODE_CHARS should become '...your_existing_chars...€'")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    dataset_file_path = input("Enter the path to your dataset.jsonl file: ")
    if not dataset_file_path:
        print("No dataset path provided. Exiting.")
    else:
        check_dataset_vocabulary(dataset_file_path, TOKENIZER_PYTHON_CODE_CHARS)
