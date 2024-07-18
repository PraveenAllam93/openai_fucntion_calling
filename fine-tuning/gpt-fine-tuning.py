# from transformers import TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
# from datasets import load_dataset
import random
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import jsonlines
import json
import numpy as np
import openai
from collections import defaultdict
from gpt_fine_tuning_functions import format_check, num_tokens_from_messages, num_assistant_tokens_from_messages, print_distribution, warnings_token_counts

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("FINE_TUNE_MODEL")
print(model) # gpt-3.5-turbo

train_data_path = "../datasets/train_dataset.jsonl"
val_data_path = "../datasets/test_dataset.jsonl"


"""
Format validation
We can perform a variety of error checks to validate that each conversation in the dataset adheres to the format expected by the fine-tuning API. Errors are categorized based on their nature for easier debugging.

Data Type Check: Checks whether each entry in the dataset is a dictionary (dict). Error type: data_type.
Presence of Message List: Checks if a messages list is present in each entry. Error type: missing_messages_list.
Message Keys Check: Validates that each message in the messages list contains the keys role and content. Error type: message_missing_key.
Unrecognized Keys in Messages: Logs if a message has keys other than role, content, weight, function_call, and name. Error type: message_unrecognized_key.
Role Validation: Ensures the role is one of "system", "user", or "assistant". Error type: unrecognized_role.
Content Validation: Verifies that content has textual data and is a string. Error type: missing_content.
Assistant Message Presence: Checks that each conversation has at least one message from the assistant. Error type: example_missing_assistant_message.
The code below performs these checks, and outputs counts for each type of error found are printed. This is useful for debugging and ensuring the dataset is ready for the next steps.
"""
print(format_check(dataset= dataset))
print(warnings_token_counts(dataset))
# Token Counting Utilities
# Warnings and tokens counts

client = OpenAI(api_key = api_key)

with open(train_data_path, "rb") as training_fd:
    training_response = client.files.create(
        file=training_fd, purpose="fine-tune"
    )

training_file_id = training_response.id

with open(val_data_path, "rb") as validation_fd:
    validation_response = client.files.create(
        file=validation_fd, purpose="fine-tune"
    )
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

response = client.fine_tuning.jobs.create(
    training_file = training_file_id,
    validation_file = validation_file_id,
    model = model,
    suffix = "initial_tuning",
)

job_id = response.id

print("Job ID:", response.id)
print("Status:", response.status)

response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

response = client.fine_tuning.jobs.list_events(job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

# Job ID: ftjob-BVk5k0jRYhjwneLwjvTk84tp
# Status: running
# Trained Tokens: None

# Created fine-tuning job: ftjob-BVk5k0jRYhjwneLwjvTk84tp
# Validating training file: file-2aCwXTqAPMXDq26c1v7dy8wE and validation file: file-I8xXgLaXTfLJicbNxPlxO5iC
# Files validated, moving job to queued state

# Training file ID: file-2aCwXTqAPMXDq26c1v7dy8wE
# Validation file ID: file-I8xXgLaXTfLJicbNxPlxO5iC

response = client.fine_tuning.jobs.retrieve("ftjob-BVk5k0jRYhjwneLwjvTk84tp")
fine_tuned_model_id = response.fine_tuned_model

if fine_tuned_model_id is None: 
    raise RuntimeError("Fine-tuned model ID not found. Your job has likely not been completed yet.")

print("Fine-tuned model ID:", fine_tuned_model_id)

test_messages =[
        # {"role": "system", "content": "Hey I am Zenitsu, I am here to help you out"},
        {"role": "user", "content": "Yo, need your assistance in proofing this email for errors"}
    ]

response = client.chat.completions.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0
)
print(response.choices[0].message.content)

openai.api_key = api_key

# Define the fine-tuned model ID
fine_tuned_model_id = 'ft:gpt-3.5-turbo-0125:personal:initial-tuning:9mN4Ip1K'

# Define test messages as a list of dictionaries

# Call OpenAI API to generate completions
response = openai.chat.completions.create(
    model=fine_tuned_model_id,
    messages=[{"role": "user", "content": "Yo, need your assistance in proofing this email for errors"}],
    temperature=0
)

# Print the generated response
print(response.choices[0].message.content)