from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel
import os

# 1. Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>"
})

# 2. Define file paths
formatted_data_path = "formatted_dataset.txt"
block_size = 1024  # GPT-2 context window

# 3. Load dataset from file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=formatted_data_path,
    block_size=block_size
)

# 4. Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

print("✅ Dataset is tokenized and ready.")
