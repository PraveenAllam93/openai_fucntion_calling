import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
print("Imported required libraries")

def read_dataset(file_path, category):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append((line,category))
    return data

def df_formation(data):
    df = pd.DataFrame(data, columns = ["prompt", "output"])
    print(df["output"].value_counts())
    print(df.shape)
    print(df.head())


model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "create_incident", 1: "deploy_cheyenne", 2: "send_mail"}
label2id = {"create_incident":0, "deploy_cheyenne":1, "send_mail" : 2}

dataset = []
dataset.extend(read_dataset("datasets/creat_incident_dataset.txt", "create_incident"))
dataset.extend(read_dataset("datasets/deploy_cheyenne_data.txt", "deploy_cheyenne"))
dataset.extend(read_dataset("datasets/send_mail_dataset.txt", "send_mail"))
n_shuffles =11

# Perform shuffling n times
for _ in range(n_shuffles):
    random.shuffle(dataset)

df_formation(dataset)

# Split your dataset into train and validation subsets
train_size = int(len(dataset) * 0.8)  # 80% for training
train_data = dataset[:train_size]
validation_data = dataset[train_size:]

df_formation(train_data)
df_formation(validation_data)

train_dict = [{'text': text, 'label': label2id[label]} for text, label in train_data]
validation_dict = [{'text': text, 'label': label2id[label]} for text, label in validation_data]

train_dataset = Dataset.from_dict({'label': [item['label'] for item in train_dict], 'text': [item['text'] for item in train_dict]})
validation_dataset = Dataset.from_dict({'label': [item['label'] for item in validation_dict], 'text': [item['text'] for item in validation_dict]})

dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
})

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenizer.truncation = True
tokenizer.padding = True

# # add pad token if none exists
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]
    label = examples["label"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding = True,
        max_length=512
    )
    # tokenized_inputs["label"] = torch.tensor(label)
    return tokenized_inputs

# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy") 

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=1)
#     return {"accuracy": accuracy.compute(predictions=predictions, references=labels["label"])}
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)  # Convert logits to predicted labels
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

random_indices = [random.randint(0, 305) for _ in range(11)]
check_data = [validation_data[idx] for idx in random_indices]
 
print("Untrained model predictions:")
print("----------------------------")
for text in check_data:
    # # tokenize text
    inputs = tokenizer.encode(text[0], return_tensors="pt")  # Accessing the prompt (text)
    # # compute logits
    logits = model(inputs).logits
    # # convert logits to label
    predictions = torch.argmax(logits)

    print(text[0] + " - " + id2label[predictions.tolist()])
    
"""
Untrained model predictions:
----------------------------
I'm running into a problem with something and need to flag it - send_mail
Hey, I've noticed a malfunction that requires attention - send_mail
Could you please confirm if the latest backend updates are ready for deployment? - send_mail
Hey, something's causing trouble and I need to address it - send_mail
Urgently need your help drafting an email - send_mail ### correct
Hey, I've noticed a malfunction with something and need to report it - send_mail
Logging an incident ticket for immediate attention - send_mail
Hey there, can you give me some pointers on this email? - send_mail  ### correct
Can you guide me through sending this email? - send_mail ### correct
My backend's got some new tricks. Can we show them off with a deployment? - send_mail
I've encountered an inconvenience with something and need help - send_mail
"""

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin']) 

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

lr = 1e-3
batch_size = 4
num_epochs = 10

training_args = TrainingArguments(
    output_dir= model_checkpoint + "-fine-tuning-custom-data",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu = True
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

print("Trained model predictions:")
print("--------------------------")
for text in check_data:
    inputs = tokenizer.encode(text[0], return_tensors="pt")  # Accessing the prompt (text)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text[0] + " - " + id2label[predictions.tolist()])

"""
Trained model predictions:
--------------------------
I'm running into a problem with something and need to flag it - create_incident
Hey, I've noticed a malfunction that requires attention - create_incident
Could you please confirm if the latest backend updates are ready for deployment? - deploy_cheyenne
Hey, something's causing trouble and I need to address it - create_incident
Urgently need your help drafting an email - send_mail
Hey, I've noticed a malfunction with something and need to report it - create_incident
Logging an incident ticket for immediate attention - create_incident
Hey there, can you give me some pointers on this email? - send_mail
Can you guide me through sending this email? - send_mail
My backend's got some new tricks. Can we show them off with a deployment? - deploy_cheyenne
I've encountered an inconvenience with something and need help - create_incident
"""

random_test_data = ["hey, I have added some new features and updated the documentation, so want to make it live", "There is a bug rise and need assist", "Okay I have some information to send someone"]

for text in random_test_data:
    inputs = tokenizer.encode(text, return_tensors="pt")  # Accessing the prompt (text)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text + " - " + id2label[predictions.tolist()])

"""   
hey, I have added some new features and updated the documentation, so want to make it live - deploy_cheyenne
There is a bug rise and need assist - create_incident
Okay I have some information to send someone - send_mail
"""
random_test_data = ["hey, I have added some new features and updated the documentation, so want to make it live. After finishing the deployment task please send me update over a mail and if anything goes wrong please do raise an incident"]

for text in random_test_data:
    inputs = tokenizer.encode(text, return_tensors="pt")  # Accessing the prompt (text)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text + " - " + id2label[predictions.tolist()])
    
# hey, I have added some new features and updated the documentation, so want to make it live. \ 
# After finishing the deployment task please send me update over a mail and \
# if anything goes wrong please do raise an incident - deploy_cheyenne
        
        
# updates to be done are:
# 1. track of history content 
# 2. try/ check out other fine-tuning methods
# 3. prompt on fine tuned LLM
# 4. fine-tuning gpt/gemini
# 6. RAG

predictions = trainer.predict(tokenized_dataset["validation"])

logits = predictions.predictions
labels = predictions.label_ids

# Convert logits to predicted labels
predicted_labels = np.argmax(logits, axis=1)

# Compute accuracy or other metrics
test_accuracy = accuracy_score(labels, predicted_labels)
print(f"Test Accuracy: {test_accuracy}")  
# tokenized_dataset["validation"]["label"]

# # Evaluate the model on the validation dataset
# eval_results = trainer.evaluate()

# print("Evaluation results:")
# print(eval_results)


checkpoint_path = "distilbert-base-uncased-fine-tuning-custom-data/checkpoint-3060"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)