import configparser
import subprocess
import json
import os
import openai
import tiktoken
import csv
import os.path
import time
import csv
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
from datetime import datetime
from IPython.display import clear_output


config_dir = os.path.dirname(__file__)
config = configparser.ConfigParser()
config.read(os.path.join(config_dir, 'gpt_local_config.cfg'))

openai.api_key = config.get('token', 'GPT_TOKEN')
fine_tune_model = config.get('model', 'model_for_fine_tune')
fine_tune_model_id = config.get('model', 'fine_tune_model_id')
model_for_chat  = config.get('model', 'model_for_chat')
praperation_file = config.get('tools', 'data_praperation_script')

def test():
    print("fine_tune_model: ",fine_tune_model)
    print("praperation_file: ",praperation_file)
    print("config_dir: ",config_dir)

def create_fine_tune_model():
    response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    n=1,
    stop=None,
    frequency_penalty=0,
    presence_penalty=0
    )   

def data_preparation(filedirs, filenames): 
    
    # config_dir = os.path.dirname(__file__)
    # script = config_dir + "\\" + praperation_file
    # script = r"{}".format(script.replace("\\", "\\\\"))
    # print(script)
    
    # filedirs = json.dumps(filedirs)
    # filenames = json.dumps(filenames)
    if len(filedirs) != len(filenames):
        print("number of directory and filename input mismatch")
        return None
    
    output_list=[]
    for i in range(len(filedirs)):
        file = filedirs[i] + "\\" + filenames[i]
        file = r"{}".format(file.replace("\\", "\\\\"))
        # file = r"{}".format(file.replace("\\\\", "\\"))
        print("Processing ", file, "Now.")
        
        yes_strings = ["yes"] * 20
        yes_input = "\n".join(yes_strings).encode()
        command = ["openai", 
                   "tools", 
                   "fine_tunes.prepare_data", 
                   "-f", 
                   f"{file}"]

        yes_pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
        output = yes_pipe.communicate(input=yes_input)
        output_list.append(output)
    return output_list

def call_chatGPT(messages):
    response = openai.ChatCompletion.create(
    model=model_for_chat,
    messages=messages,
    temperature=0,
    )
    return response
    
def format_message(role: str, content: str):
    message = {"role": role, "content": content}
    return message  

def list_fine_tune():
    models = openai.Model.list()
    for model in models['data']:
        owner = model['owned_by']
        if 'user' in owner:
            print(model['id'],owner)

def call_fine_tune(prompt, fine_tune_model_id):
    prompt = prompt + " ->"
    result = openai.Completion.create(
            model = fine_tune_model_id, 
            prompt = prompt,
            temperature = 0.05,
            max_tokens = 150)
    
    # full = gpt.create(prompt, max_length=120)
    text = result.choices[0].text
    return result, text

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model="gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        
def append_to_csv(file_path, headers, data_rows):
    """
    Append rows of data to a CSV file. If the file doesn't exist, it will be created with the specified headers.
    """
    # Create the CSV file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    # Append rows to the CSV file
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_rows:
            writer.writerow(row)

def ask_and_log_fine_tune(question,filename,headers):
    full, answer = call_fine_tune(question)
    data_row=[question,answer,full]
    append_to_csv(filename,headers,data_row)
    print(answer)
    
def get_questions(context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write questions based on the text below\n\nText: {context}\n\nQuestions:\n1." }],
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        complete_q = "1. " + response['choices'][0].message['content']
        return complete_q
    except:
        return ""

def gpt_get_answers(row):
    context = row[0]
    question = row[1]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user",
                       "content": f"Write answer based on the text below\n\nText: {context}\n\nQuestions:\n{question}"}],
            temperature=0,
            max_tokens=350,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0]['message']['content']
    except Exception as e:
        print (e)
        return ""

    
def gpt_process_questions(df, output_path):
    answer_list = []
    df_tuple = [tuple(x) for x in df.to_numpy()]

    with Pool() as pool, open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])
        for answer in tqdm(pool.imap(gpt_get_answers, df_tuple), total=len(df)):
            answer_list.append(answer)
            writer.writerow([df.loc[answer_list.index(answer)][1], answer])

    return pd.DataFrame({"Question": df.iloc[:, 1], "Answer": answer_list})


def parse_finetune_events(events):
    results = {"created_at": [], "level": [], "message": []}
    for event in events:
        results["created_at"].append(event.created_at)
        results["level"].append(event.level)
        results["message"].append(event.message)
    return pd.DataFrame(results)

def monitor_finetune_events(model_id):
    # Get initial status
    retrieve_response = openai.FineTune.retrieve(id=model_id)
    df = parse_finetune_events(retrieve_response['events'])
    clear_output(wait=True)
    display(df)
    
    # Check for completion
    if df.iloc[-1]['message'] == 'Fine-tune succeeded':
        return
    
    # Wait for some time
    time.sleep(30)
    
    # Call this function again recursively
    monitor_finetune_events(model_id)

