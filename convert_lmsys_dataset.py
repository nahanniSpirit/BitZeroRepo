import json
import time
from datasets import load_dataset
from tqdm import tqdm
import os
# from collections import defaultdict # No longer needed for grouping

def convert_lmsys_chat_dataset(
    output_jsonl_path="datasets/lmsys_chat_1m_converted.jsonl",
    difficulty_default=0.5,
    split_percentage="train[:1%]"
    ):
    dataset_name = "lmsys/lmsys-chat-1m"
    processed_input_conversations_count = 0 # Count of conversations from the input dataset
    extracted_pairs_count = 0
    
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"Loading dataset: {dataset_name} (split: {split_percentage})...")
        lmsys_dataset = load_dataset(dataset_name, split=split_percentage)
        print(f"Dataset loaded. Number of conversations (rows): {len(lmsys_dataset)}")
        print(f"Features: {lmsys_dataset.features}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        print("Processing conversations into user-assistant pairs...")
        # Each 'example' in lmsys_dataset is already a full conversation
        for example_idx, conversation_data in enumerate(tqdm(lmsys_dataset, desc="Processing conversations")):
            turns = conversation_data.get("conversation") # This is a list of turn dicts
            conv_id = conversation_data.get("conversation_id", f"unknown_id_{example_idx}") # Get original ID or make one

            if not turns or not isinstance(turns, list):
                continue # Skip if no turns list or it's not a list

            processed_input_conversations_count +=1
            
            # The turns are already in order within the 'conversation' list
            for i in range(len(turns) - 1):
                turn1 = turns[i]
                turn2 = turns[i+1]

                # Get role and content from within each turn dictionary
                role1_original = turn1.get("role")
                content1 = turn1.get("content")
                role2_original = turn2.get("role")
                content2 = turn2.get("content")

                # Map roles: 'human' -> 'user', 'gpt' -> 'assistant'
                # (The dataset card says roles are "human" and "gpt", but it's good to be flexible)
                # Let's be more specific or add a check for other roles if necessary
                role1_mapped = None
                if role1_original == "human":
                    role1_mapped = "user"
                elif role1_original == "user": # In case 'user' is already used
                    role1_mapped = "user"
                
                role2_mapped = None
                if role2_original == "gpt":
                    role2_mapped = "assistant"
                elif role2_original == "assistant": # In case 'assistant' is already used
                    role2_mapped = "assistant"


                if role1_mapped == "user" and role2_mapped == "assistant":
                    if not content1 or not content2:
                        continue 

                    current_timestamp = time.time() + (extracted_pairs_count * 0.000001)

                    conversation_entry = {
                        "type": "conversation_lmsys", 
                        "difficulty": difficulty_default, 
                        "conversation": [
                            {"role": "user", "content": str(content1), "timestamp": current_timestamp},
                            {"role": "assistant", "content": str(content2), "timestamp": current_timestamp + 0.000001}
                        ],
                        "metadata": {
                            "original_conversation_id": str(conv_id),
                            "user_turn_idx_in_original": i, # Original index of user turn
                            "assistant_turn_idx_in_original": i + 1 # Original index of assistant turn
                        } 
                    }
                    
                    outfile.write(json.dumps(conversation_entry) + '\n')
                    extracted_pairs_count += 1
            
    print(f"\nConversion complete.")
    print(f"Processed {processed_input_conversations_count} original conversations from input dataset.")
    print(f"Extracted {extracted_pairs_count} user-assistant pairs into {output_jsonl_path}.")

if __name__ == "__main__":
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
    split_to_process = "train[:1%]" 
    # split_to_process = "train" # For the full dataset 

    print(f"WARNING: Preparing to process split '{split_to_process}'.")
    if split_to_process == "train":
        print("This will process the FULL lmsys-chat-1m dataset and may take a very long time.")
    # confirm = input("Proceed? (yes/no): ")
    # if confirm.lower() == 'yes':
    #     convert_lmsys_chat_dataset(split_percentage=split_to_process)
    # else:
    #     print("Conversion cancelled by user.")
    convert_lmsys_chat_dataset(split_percentage=split_to_process)