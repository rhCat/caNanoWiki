# OpenAI GPT-3.5 Turbo Utilities and Embedding Utilities

This repository contains two Python scripts, `gpt_utils.py` and `embedding_util.py`, designed to interact with OpenAI's GPT-3.5 Turbo model and the OpenAI Embedding API. 

## gpt_utils.py

This script provides a set of utility functions for interacting with the GPT-3.5 Turbo model and preparing data for fine-tuning.

Key functions include:

- `create_fine_tune_model()`: Creates a fine-tuned model using the OpenAI API.
- `data_preparation(filedirs, filenames)`: Prepares data for fine-tuning by running the OpenAI `fine_tunes.prepare_data` tool on a set of files.
- `call_chatGPT(messages)`: Sends a list of messages to the GPT-3.5 Turbo model and returns the model's response.
- `format_message(role, content)`: Formats a message for the GPT-3.5 Turbo model.
- `list_fine_tune()`: Lists all fine-tuned models owned by the user.
- `call_fine_tune(prompt, fine_tune_model_id)`: Sends a prompt to a fine-tuned model and returns the model's response.
- `num_tokens_from_string(string, model)`: Returns the number of tokens in a string.
- `num_tokens_from_messages(messages, model)`: Returns the number of tokens used by a list of messages.
- `append_to_csv(file_path, headers, data_rows)`: Appends rows of data to a CSV file.
- `ask_and_log_fine_tune(question,filename,headers)`: Sends a question to a fine-tuned model, logs the model's response to a CSV file, and prints the response.
- `get_questions(context)`: Generates a list of questions based on a context using the GPT-3.5 Turbo model.
- `gpt_get_answers(row)`: Sends a question to the GPT-3.5 Turbo model and returns the model's response.
- `gpt_process_questions(df, output_path)`: Sends a list of questions to the GPT-3.5 Turbo model and writes the model's responses to a CSV file.
- `parse_finetune_events(events)`: Parses the events from a fine-tuning operation.
- `monitor_finetune_events(model_id)`: Monitors the events from a fine-tuning operation and displays them in real time.

## embedding_util.py

This script provides a set of utility functions for creating and working with embeddings using the OpenAI Embedding API.

Key functions include:

- `get_embedding(text: str, model: str=EMBEDDING_MODEL)`: Returns the embedding for a text string.
- `compute_doc_embeddings(df: pd.DataFrame)`: Returns a dictionary mapping each row in a dataframe to its embedding.
- `load_embeddings(fname: str)`: Loads embeddings from a CSV file.
- `vector_similarity(x: list[float], y: list[float])`: Returns the similarity between two vectors.
- `order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array])`: Returns a list of document sections sorted by relevance to a query.
- `construct_prompt(user_question, context_embeddings: dict, df: pd.DataFrame)`: Constructs a prompt for the GPT-3.5 Turbo model based on a user's question and the most relevant document sections.
- `answer_query_with_context(question, document_df: pd.DataFrame, document_embeddings: dict[(str, str), np.array], show_prompt: bool = False)`: Sends a question to the GPT-3.5 Turbo model and returns the model's response.
- `context_based_QA(parsed_tuple)`: Sends a question to the GPT-3.5 Turbo model and returns the model's response, along with the file paths and figure titles associated with the question.

## Usage

To use these scripts, you will need to import them into your Python project and call the functions as needed. For example, you might use the `compute_doc_embeddings` function from `embedding_util.py` to create embeddings for your document sections, and the `call_chatGPT` function from `gpt_utils.py` to send a list of messages to the GPT-3.5 Turbo model.

## Dependencies

These scripts require the OpenAI Python package, which can be installed using pip:

```bash
pip install openai
