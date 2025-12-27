import json
import random
import os

# --- Configuration ---
INPUT_FILE = "../../data/raw_data/ode_training_data.json"
DATA_DIR_REL = "../../data/processed_data"  # Folder where MLX will look for files
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# Test ratio is implicitly 0.1

def format_for_mlx():
    # 1. Create directory if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.abspath(os.path.join(base_dir, DATA_DIR_REL))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # 2. Load the raw generated data
    input_path = os.path.abspath(os.path.join(base_dir, INPUT_FILE))
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    # 3. Shuffle for better distribution across splits
    random.shuffle(raw_data)

    # 4. Map to MLX-preferred "Completions" format
    formatted_data = []
    for entry in raw_data:
        # We structure the prompt to be very explicit for an SLM
        prompt_text = f"Instruction: {entry['input']}\nResponse:"
        
        mlx_entry = {
            "prompt": prompt_text,
            "completion": entry['output']
        }
        formatted_data.append(mlx_entry)

    # 5. Calculate split indices
    total = len(formatted_data)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_set = formatted_data[:train_end]
    valid_set = formatted_data[train_end:valid_end]
    test_set = formatted_data[valid_end:]

    # 6. Save as .jsonl files
    def save_jsonl(data, filename):
        path = os.path.join(processed_data_dir, filename)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} samples to {path}")

    save_jsonl(train_set, "train.jsonl")
    save_jsonl(valid_set, "valid.jsonl")
    save_jsonl(test_set, "test.jsonl")

if __name__ == "__main__":
    format_for_mlx()