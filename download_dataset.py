from datasets import load_dataset
import os
import json

# Set your target folder
output_dir = r"resume_dataset"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
dataset = load_dataset("shandilyabh/resume_parsing", split="train")

# Save input-output pairs as .txt and .json
for i, sample in enumerate(dataset):
    try:
        # Extract resume text and structured JSON string
        resume_text = sample['messages'][0]['content']
        output_json_str = sample['messages'][1]['content']
        output_json = json.loads(output_json_str)

        # Save .txt file
        with open(os.path.join(output_dir, f"resume_{i:04d}.txt"), "w", encoding="utf-8") as txt_file:
            txt_file.write(resume_text)

        # Save .json file
        with open(os.path.join(output_dir, f"resume_{i:04d}.json"), "w", encoding="utf-8") as json_file:
            json.dump(output_json, json_file, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"⚠️ Error on index {i}: {e}")

print(f"✅ Downloaded and saved {len(dataset)} resume samples to: {output_dir}")
