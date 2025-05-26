from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>"
})

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Reload dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="formatted_dataset.txt",
    block_size=1024
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_resume_parser_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,  # log every 50 steps
    logging_dir="./logs",  # optional: logs for TensorBoard
    report_to="none",  # prevent sending to W&B or others
    prediction_loss_only=False,  # <- this shows training loss
    fp16=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train!
trainer.train()
trainer.save_model("./gpt2_resume_parser_model")
tokenizer.save_pretrained("./gpt2_resume_parser_model")

print("✅ Training complete and model saved!")
