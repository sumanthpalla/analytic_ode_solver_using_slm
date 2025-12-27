import torch
import os
import gc
import json
import csv
import time
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# --- 1. Configuration ---
MODEL_CONFIGS = {
    "llama3": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "targets": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "qwen_math": {
        "id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "deepseek_r1": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "mistral_math": {
        "id": "mistralai/Mathstral-7B-v0.1",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "phi4": {
        "id": "microsoft/phi-4",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "gemma2": {
        "id": "google/gemma-2-9b-it",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
}

# Models to train in this session
MODELS_TO_TRAIN = ["llama3"]
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILES = {
    "train": os.path.join(BASE_DIR, "data/processed_data/train.jsonl"),
    "test": os.path.join(BASE_DIR, "data/processed_data/test.jsonl")
}
RESULTS_FILE = os.path.join(BASE_DIR, "training_results.csv")

def run_finetuning(model_key):
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"🚀 Starting Fine-tuning for: {model_key} ({config['id']})")
    print(f"{'='*60}")

    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['id'])
        tokenizer.pad_token = tokenizer.eos_token

        # Device & Precision Setup
        is_mps = torch.backends.mps.is_available()
        # Use bfloat16 on MPS as it is better supported than float16
        model_dtype = torch.bfloat16 if is_mps else torch.float16

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            config['id'], 
            dtype=model_dtype,
            device_map="auto"
        )

        # LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=config['targets'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        # Dataset
        dataset = load_dataset("json", data_files=DATA_FILES)

        def tokenize_function(examples):
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": c}], 
                tokenize=False
            ) for p, c in zip(examples['prompt'], examples['completion'])]
            return tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Training Setup
        training_args = TrainingArguments(
            output_dir=os.path.join(BASE_DIR, f"adapters/{model_key}_ode_adapter"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=not is_mps, 
            bf16=is_mps,
            push_to_hub=False,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Train
        start_time = time.time()
        trainer.train()
        end_time = time.time()

        # Metrics for Paper
        train_stats = trainer.state.log_history
        # Get final loss from logs
        final_train_loss = next((log['loss'] for log in reversed(train_stats) if 'loss' in log), None)
        final_eval_loss = next((log['eval_loss'] for log in reversed(train_stats) if 'eval_loss' in log), None)
        
        peak_memory = torch.mps.current_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0
        total_flos = trainer.state.total_flos

        # Log to CSV
        file_exists = os.path.isfile(RESULTS_FILE)
        with open(RESULTS_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Model ID", "Train Loss", "Eval Loss", "Time (s)", "Peak Memory (GB)", "Total FLOS"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_key,
                config['id'],
                final_train_loss,
                final_eval_loss,
                f"{end_time - start_time:.2f}",
                f"{peak_memory:.2f}",
                total_flos
            ])

        print(f"Finished {model_key}. Results saved to {RESULTS_FILE}")

    except Exception as e:
        print(f"Error training {model_key}: {e}")

    finally:
        # Cleanup - check if variables exist before deleting
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        if 'trainer' in locals(): del trainer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    for model_key in MODELS_TO_TRAIN:
        run_finetuning(model_key)
    print("\n All scheduled training runs complete.")