import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os

print("=" * 60)
print("🌾 AGROEXPERT MODEL TRAINING")
print("=" * 60 + "\n")

# Check GPU
if not torch.cuda.is_available():
    print("❌ No GPU")
    exit(1)

if not os.path.exists("data/agro_train.jsonl"):
    print("❌ No dataset. Run download_agro_dataset.py first")
    exit(1)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/agro-expert"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {MODEL_NAME}\n")

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
)

model.gradient_checkpointing_enable()

# LoRA config
print("Adding LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("json", data_files="data/agro_train.jsonl", split="train")
print(f"Loaded {len(dataset)} examples\n")


# Format for TinyLlama
def format_chat(example):
    inst = example['instruction']
    out = example['output']

    text = f"""<|system|>
You are an agricultural expert assistant helping farmers.</s>
<|user|>
{inst}</s>
<|assistant|>
{out}</s>"""

    return {"text": text}


dataset = dataset.map(format_chat)


# Tokenize
def tokenize(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


print("Tokenizing...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    warmup_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("🚀 Training (10-15 minutes)...\n")
print("=" * 60)

trainer.train()

print("\n💾 Saving...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Saved to {OUTPUT_DIR}")
print("\n🌾 AgroExpert ready!")