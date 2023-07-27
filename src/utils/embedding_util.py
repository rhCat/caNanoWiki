import os
import sys
import pandas as pd
import openai
import configparser
import re
import numpy as np
import csv
import tiktoken
import math
from openai.error import RateLimitError

EMBEDDING_MODEL = "text-embedding-ada-002"
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
MAX_SECTION_LEN = 1200
ALLOWED_TOKEN = 3800

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


    
def construct_prompt(user_question, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(user_question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    string_list = [str(item) for item in chosen_sections]
    chosen_sections_str = ''.join(string_list)
    
    header = """Answer the question strictly using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + chosen_sections_str + "\n\n Question: " + str(user_question) + "\n A:"

def answer_query_with_context(
    question,
    document_df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    
    prompt = construct_prompt(
        question,
        document_embeddings,
        document_df
    )
    
    prompt_len = len(encoding.encode(prompt))
    
    if show_prompt:
        print(prompt)
        
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user",
               "content": prompt}],
    temperature=0,
    max_tokens=ALLOWED_TOKEN - prompt_len
    # top_p=1,
    # frequency_penalty=0,
    # presence_penalty=0
    )
    return prompt, response.choices[0]['message']['content']


def context_based_QA(parsed_tuple):
    ueser_question, document_context, document_embeddings = parsed_tuple
    prompt, answer = embedding_util.answer_query_with_context(ueser_question, document_context, document_embeddings)
    if isinstance(files, list):
        if any(isinstance(item, list) for item in files):
            files = [elem for sublist in files for elem in sublist]
        file_path = ";".join(files)
        return [str(file_path), str(figure_title), str(answer)]
    else:
        return [str(files), str(figure_title), str(answer)]
