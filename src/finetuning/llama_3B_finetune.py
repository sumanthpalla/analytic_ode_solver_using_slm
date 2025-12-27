import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Setup & Model Loading
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# Use float16 for stability on Apple Silicon MPS
torch_dtype = torch.float16 

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="mps"  # Directing to Apple GPU
)

# 2. Configure LoRA (The "Adapter")
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target Attention Layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Load Dataset (from our Step 2.5)
dataset = load_dataset("json", data_files={"train": "../../data/processed_data/train.jsonl", "test": "../../data/processed_data/test.jsonl"})

# 4. Training Arguments optimized for M4
training_args = TrainingArguments(
    output_dir="./ode_slm_results",
    per_device_train_batch_size=2, # Keep low for MPS memory stability
    gradient_accumulation_steps=4, # Effective batch size of 8
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    # Apple Silicon Specifics:
    bf16=False, 
    fp16=True,
    dataloader_num_workers=0 # MPS prefers 0 to avoid multi-process errors
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    dataset_text_field="prompt", # Defined in Step 2.5
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# 
print("Starting Fine-Tuning on M4 GPU...")
trainer.train()

# 6. Save the specialized adapter
trainer.save_model("./math_ode_adapter")