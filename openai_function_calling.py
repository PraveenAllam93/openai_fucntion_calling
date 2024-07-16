import os
import json
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
import numpy as np
import fucntion_description as fd
import fucntions as f
import time
import transformers

print(openai.__version__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")
print(model)

try:
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "I love MonaLisa",
            },
        ],
        temperature=1,
        n = 5
    )

    # the completion is an object  <openai.openai_object.OpenAIObject>
    # print(type(completion))
    for i in range(len(completion.choices)):
        print(completion.choices[i].message.content)
except openai.error.OpenAIError as e:
    print(f"An error occurred: {e}")
# print(completion.choices[0].message.content)


user_prompt = "I want to book a flight from Mumbai to Bangkok"
completion = openai.ChatCompletion.create(
    model = model,
    messages = [
        {
            "role":  "user",
            "content": user_prompt
        }
    ],
    functions = fd.basic_function_description,
    function_call = "auto"
)

output = completion.choices[0].message
print(output)



 
origin_location = json.loads(output.function_call.arguments).get("origin_location")
destination_location = json.loads(output.function_call.arguments).get("destination_location")
params = json.loads(output.function_call.arguments)
print("Origin: {}, Destination: {}".format(origin_location, destination_location))
print(params)

chosen_function = output.function_call.name
flight_info = f.get_fucntion(chosen_function, params)
print(flight_info)



user_prompt = "Hey, I want to deploy my changes done to the backend application cheyenne."

completion = openai.ChatCompletion.create(
    model = model,
    messages = [
        {
            "role":  "user",
            "content": user_prompt
        }
    ],
    functions = fd.deploy_function_description,
    function_call = "auto",
    # n = 5,
    temperature = 2
)

output = completion.choices[0].message
print(output)

params = "No params required"
chosen_function = output.function_call.name
deployment_state = f.get_fucntion(chosen_function, params)
print(deployment_state)

def define_bot(content):
    print(content)
    completion = openai.ChatCompletion.create(
        model = model,
        messages = [
            {
                "role":  "user",
                "content": content
            }
        ],
        functions = fd.deploy_function_description,
        function_call = "auto", 
        n = 5,
        temperature = 0.2
    )

    for i in range(len(completion.choices)):
        print(completion.choices[i].message)
    print("---"*10)    
    output = completion.choices[0].message
    function_name = output.function_call.name
    print(fd.deploy_function_description[0]["description"])
    deployment_state = f.get_fucntion(function_name, params)
    print(deployment_state)

   

user_prompt = "Hey, I have done some changes (just added some new features and updated the documentation)" \
              " to our backend which is chenneye" \
              " and I want to implement the changes in production."

define_bot(user_prompt)



## Lyric completion assistant

messages_list = [
    {"role" : "system", "content" : "I am a Lyric completion bot, give a lyric and I will complete"},
    {"role" : "user", "content" : "Thought I almost died in my dream again (baby, almost died)"},
    {"role" : "assistant", "content" : "Fightin' for my life, I couldn't breathe again"},
    {"role" : "user", "content" : "I'm fallin' in too deep (oh)"},
    {"role" : "assistant", "content" : "Without you, don't wanna sleep (fallin' in)"},
    {"role" : "user", "content" : "Cause my heart belongs to you"},
]

for i in range(5):
    completion = openai.ChatCompletion.create(
        model = model,
        max_tokens = 15,
        n = 1,
        temperature = 0.2,
        messages = messages_list
    )

    print(completion.choices[0].message.content)
    # print("-" * 40)
    new_message = {"role" : "assistant", "content" : completion.choices[0].message.content}
    messages_list.append(new_message)
    time.sleep(0.1)



### transformers -> text classification

import transformers

# Using a PyTorch-based model for text classification
classifier = transformers.pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Example usage
texts = ["I love this!", "I hate this!"]
results = classifier(texts)

for result in results:
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")