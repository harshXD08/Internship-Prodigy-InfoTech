print("RUNNING CORRECT FILE")
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset('text', data_files={'train': 'data.txt'})

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize data
def tokenize_function(examples):
    tokens = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=50
    )
    
    tokens["labels"] = tokens["input_ids"]  # 🔥 THIS LINE FIXES ERROR
    
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Train
trainer.train()

# Save model
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")