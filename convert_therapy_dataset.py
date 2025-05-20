# convert_therapy_dataset.py (or rename it e.g., convert_mistral_psych_dataset.py)
import json
import time
from datasets import load_dataset 
from tqdm import tqdm
import os

def convert_mistral_psychology_dataset(output_jsonl_path="datasets/mistral_psychology_converted.jsonl", difficulty_default=0.5):
    dataset_name = "hadishaheed1234/Mistral-psychology-dataset" 
    processed_count = 0
    skipped_count = 0

    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"Loading dataset: {dataset_name}...")
        psych_dataset = load_dataset(dataset_name, split="train")
        
        print(f"Dataset loaded. Number of examples: {len(psych_dataset)}")
        print(f"Features: {psych_dataset.features}") # Good to keep for verification
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please check your internet connection and the dataset name.")
        return

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for i, example in enumerate(tqdm(psych_dataset, desc="Converting dataset")):
            # --- MODIFICATION: Use correct column names 'input' and 'output' ---
            user_content = example.get("input")    # Changed from "instruction"
            assistant_content = example.get("output") # Changed from "response"
            # --- END MODIFICATION ---

            if not user_content or not assistant_content:
                # This print can be very verbose for large datasets, uncomment if needed for debugging
                # print(f"Skipping example {i} due to missing 'input' or 'output' field. Data: {example}")
                skipped_count +=1
                continue
            
            current_timestamp = time.time() + (i * 0.000001) 
            conversation_entry = {
                "type": "instruction_psychology", 
                "difficulty": difficulty_default, 
                "conversation": [
                    {"role": "user", "content": str(user_content), "timestamp": current_timestamp},
                    {"role": "assistant", "content": str(assistant_content), "timestamp": current_timestamp + 0.000001}
                ],
                "metadata": {} 
            }
            
            outfile.write(json.dumps(conversation_entry) + '\n')
            processed_count += 1

    print(f"\nConversion complete.")
    print(f"Processed {processed_count} examples into {output_jsonl_path}.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} examples due to missing data.")

if __name__ == "__main__":
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    convert_mistral_psychology_dataset()