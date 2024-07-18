import os
import json
import random

def read_dataset(file_path, category):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append({"messages": [{"role": "system", "content": "Hey I am Zenitsu, I am here to help you out"}, {"role": "user", "content": line}, {"role": "assistant", "content": category}]})
    return data

def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

# gathering data from text files and assigning the categories
dataset = []
dataset.extend(read_dataset("creat_incident_dataset.txt", "create_incident"))
dataset.extend(read_dataset("deploy_cheyenne_data.txt", "deploy_cheyenne"))
dataset.extend(read_dataset("send_mail_dataset.txt", "send_mail"))

# shuffling the data
n_shuffles =11
for _ in range(n_shuffles):
    random.shuffle(dataset)

train_size = int(len(dataset) * 0.8)  # 80% for training
train_data = dataset[:train_size]
validation_data = dataset[train_size:]

# Example usage
output_directory = "/Users/praveenallam/Desktop/function_calling/datasets"
train_output_filename = "train_dataset.jsonl"
test_output_filename = "test_dataset.jsonl"

train_output_path = os.path.join(output_directory, train_output_filename)
test_output_path = os.path.join(output_directory, test_output_filename)

# Write JSONL file
write_jsonl(train_data, train_output_path)
write_jsonl(validation_data, test_output_path)

print(f"Train JSONL file saved to: {train_output_path}")
print(f"Validation JSONL file saved to: {test_output_path}")

