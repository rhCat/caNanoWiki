import os
import sys
import time
import pickle
import openai
import configparser
from flask import Flask, render_template, request, redirect, url_for
dir_path = os.path.abspath(os.getcwd())

src_path = dir_path + "/src"
sys.path.append(src_path)

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
config_dir = dir_path + "/src/utils"
config = configparser.ConfigParser()
config.read(os.path.join(config_dir, 'gpt_local_config.cfg'))
openai.api_key = config.get('token', 'GPT_TOKEN')

import embedding_qa as emq

# Specify the path to your pickle file
pickle_file_path = 'caNano_embedding_pack_5_14.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

document_df = loaded_data['df']
document_embedding = loaded_data['embedding']

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the
    # most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 800,
    "model": "gpt-3.5-turbo"
}

app = Flask("caNanoWiki_AI")

# Set the passcode for authentication
PASSCODE_auth = ""

# Define a variable to track if the user is authenticated
authenticated = False
last_activity_time = 0

# Timeout duration in seconds
timeout_duration = 5 * 60

# Session Length
session_duration = 30 * 60


@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>')


@app.route('/', methods=['GET', 'POST'])
def index():
    global authenticated, last_activity_time, login_time

    if not authenticated:
        return redirect(url_for('login'))

    # Check for timeout
    current_time = time.time()
    if current_time - last_activity_time > timeout_duration:
        authenticated = False
        return redirect(url_for('login'))

    # Check for session timeout
    if current_time - login_time > session_duration:
        authenticated = False
        return redirect(url_for('login'))

    # Update last activity time
    last_activity_time = current_time

    user_input = ""
    processed_input = None
    if request.method == 'POST':
        user_input = request.form['user_input']

        processed_input, chosen_sec_idxes = emq.answer_query_with_context(
            user_input,
            document_df,
            document_embedding
        )

        return render_template(
            'index.html',
            processed_input=processed_input,
            source_sections=chosen_sec_idxes,
            user_input=user_input,
            authenticated=authenticated)

    return render_template('index.html', authenticated=authenticated)


@app.route('/login', methods=['GET', 'POST'])
def login():
    global authenticated, last_activity_time, login_time

    if request.method == 'POST':
        password = request.form['passcode']
        if password == PASSCODE_auth:
            authenticated = True
            last_activity_time = time.time()
            login_time = time.time()
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='Incorrect password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    global authenticated
    authenticated = False
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
