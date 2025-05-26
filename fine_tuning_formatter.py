#to convert normal data into fine tuning format

import json
import os

input_folder = "resume_dataset/"
output_file = "formatted_dataset.txt"

with open(output_file, "w", encoding="utf-8") as out_f:
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(input_folder, filename)
            json_path = txt_path.replace(".txt", ".json")

            if not os.path.exists(json_path):
                continue  # skip if no matching JSON

            with open(txt_path, "r", encoding="utf-8") as f_txt, \
                 open(json_path, "r", encoding="utf-8") as f_json:

                input_text = f_txt.read().strip()
                output_json = json.load(f_json)
                output_text = json.dumps(output_json, ensure_ascii=False, indent=2)

                combined = f"<|startoftext|>\nInput: {input_text}\nOutput: {output_text}\n<|endoftext|>\n"
                out_f.write(combined)
