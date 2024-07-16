from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets

initial_dataset = [
    ("I need to send an email?", "send_mail"),
    ("Could you deploy Cheyenne today?", "deploy_cheyenne"),
    ("Please deploy AlphaSite ASAP", "deploy_alphasite"),
    ("How to start the Walter deployment?", "deploy_walter"),
    ("I have made some changes to my backend can you please deploy it", "deploy_cheyenne"),
    ("I have made some changes to my frontend can you please deploy it", "deploy_alphasite"),
    ("I have made some changes to my bot can you please deploy it", "deploy_walter"),
    ("please do send a quick update", "send_mail"),
    ("Hello, may I send them an email?", "send_mail"),
    ("Whoa, I have an email to send.", "send_mail"),
    ("Alright, may I send them an email?", "send_mail"),
    ("Hello, may I please send them an email?", "send_mail"),
    ("Could you please deploy my updated backend? I made some changes to it.", "deploy_cheyenne"),
    ("Would you kindly let my revised backend go live? I modified it in a few ways.", "deploy_cheyenne"),
    ("Please, could you implement my modified backend? I changed a few things about it.", "deploy_cheyenne"),
    ("Would you kindly update my backend and deploy it? It's been modified by me.", "deploy_cheyenne"),
    ("Would you kindly allow me to use my updated Cheyenne backend? I modified it in a few ways.", "deploy_cheyenne"),
    ("Could you kindly deploy Cheyenne, my modified backend? I changed a few things about it.", "deploy_cheyenne"),
    ("Would you kindly enable my Cheyenne backend to be updated? It's been modified by me.", "deploy_cheyenne"),
    ("Would you kindly enable my upgraded Cheyenne backend? I changed it in a few ways.", "deploy_cheyenne"),
    ("Could you please deploy my updated frontend? I made some changes to it.", "deploy_alphasite"),
    ("Would you kindly let my revised frontend go live? I modified it in a few ways.", "deploy_alphasite"),
    ("Please, could you implement my modified frontend? I changed a few things about it.", "deploy_alphasite"),
    ("Would you kindly update my frontend and deploy it? It's been modified by me.", "deploy_alphasite"),
    ("Would you kindly allow me to use my updated AlphaSite frontend? I modified it in a few ways.", "deploy_alphasite"),
    ("Could you kindly deploy AlphaSite, my modified frontend? I changed a few things about it.", "deploy_alphasite"),
    ("Would you kindly enable my AlphaSite to be updated? It's been modified by me.", "deploy_alphasite"),
    ("Would you kindly enable my upgraded AlphaSite? I changed it in a few ways.", "deploy_alphasite"),
    ("Could you please deploy my updated bot? I made some changes to it.", "deploy_walter"),
    ("Would you kindly let my revised bot go live? I modified it in a few ways.", "deploy_walter"),
    ("Please, could you implement my modified bot? I changed a few things about it.", "deploy_walter"),
    ("Would you kindly update my bot and deploy it? It's been modified by me.", "deploy_walter"),
    ("Would you kindly allow me to use my updated Walter bot? I modified it in a few ways.", "deploy_walter"),
    ("Could you kindly deploy Walter, my modified bot? I changed a few things about it.", "deploy_walter"),
    ("Would you kindly enable my Walter to be updated? It's been modified by me.", "deploy_walter"),
    ("Would you kindly enable my upgraded Walter? I changed it in a few ways.", "deploy_walter"),
    ("Would you kindly allow me to use my updated Cheyenne backend? I modified it in a few ways.", "deploy_cheyenne"),

]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create a dataset from the texts
texts = [item[0] for item in initial_dataset]
dataset = Dataset.from_dict({"text": texts})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    use_cpu = True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# input_text = "Hey I have made some changes to the code, can you plese deploy cheyenne?"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')

# generated_examples = model.generate(
#     input_ids=input_ids,
#     max_length=50,
#     num_return_sequences=5,
#     num_beams = 5,
#     no_repeat_ngram_size=2,
#     top_k=50,
#     top_p=0.95,
#     temperature=0.7,
# )

# # Decode and print generated examples
# for example in generated_examples:
#     print(tokenizer.decode(example, skip_special_tokens=True))

def generate_examples(prompt, num_examples=5):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=50,
        num_return_sequences=num_examples,
        num_beams=num_examples,  # Use beam search with the same number as num_return_sequences
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

new_examples = generate_examples("Alright, may I send them an email?.", 10)
print("Generated Examples:")
for example in new_examples:
    print(example)
    initial_dataset.append((example, "deploy_cheyenne"))

# Re-train the model with the expanded dataset
trainer.train_dataset = initial_dataset
trainer.train()


### Generated examples

"""
Generated Examples:
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some of the things about it: 1.1. The backend has been updated
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has changed a lot.
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has been updated. Third
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some of the things about it: 1.1. The backend has been changed
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some of the things about it: 1.1. The backend has been upgraded
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some of the things about it: 1.1.0.2.3
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has changed a lot about
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has been updated. Thanks
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has changed a lot in
Could you please deploy my updated backend? I made some changes to it. It's been modified by me. I modified it in a few ways. First, I changed some things about the backend. Second, my modified backend has been updated. You

Generated Examples:
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I made some changes to it
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I added some changes to it
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I added some new features.
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's changed the way it works. Second, I made some changes
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I made some changes to my
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I added some new features to
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I made some changes to the
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's changed the way it works. Second, I added some changes
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I made some changes to its
Please, could you implement my modified frontend? I changed a few things about it. It's been modified by me. I modified it in a couple of ways. First, it's updated my backend. Second, I added some changes to my

Generated Examples:
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

I changed some of the things about it
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

I changed some things about the backend.
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

I changed some of the things about the
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

I changed some things about my backend.
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are:

I changed some of the way it works.

Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

I changed some things about the way it
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

* I changed some of the code.
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

* I changed some of the code to
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

* I changed some of the code in
Alright, may I send them an email?. It's been modified by me. I modified it in a few ways. First, I made some changes to it. The changes are as follows:

* I changed some of the functionality.
"""