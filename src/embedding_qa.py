import os
import openai
import tiktoken
import warnings
import numpy as np
import pandas as pd
import configparser

# Mute the PerformanceWarning
warnings.filterwarnings("ignore", category=Warning)
dir_path = os.path.abspath(os.getcwd())
config_dir = dir_path + "/src"
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
config = configparser.ConfigParser()
config.read(os.path.join(config_dir, 'gpt_local_config.cfg'))
openai.api_key = config.get('token', 'GPT_TOKEN')
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
MAX_SECTION_LEN = 2000
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# The embedding functions were inspired by example
# "Question answering using embeddings-based search"
# in the OpenAI Cookbook repo (https://github.com/openai/openai-cookbook)
# which hosts a great number of example applications
# using OpenAI APIs. The content is fast evolving and the
# current example is far different then what I saw before.
# It is a great resource to learn and get inspired!


def get_embedding(
    text: str,
    model: str = EMBEDDING_MODEL
) -> list[float]:

    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(
    df: pd.DataFrame
) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe
    using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding
    vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ...
        up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([
        int(c) for c in df.columns if c != "title" and c != "heading"
    ])
    return {
           (r.title, r.heading): [
                r[str(i)] for i in range(max_dim + 1)
            ] for _, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    Because OpenAI Embeddings are normalized to length 1,
    the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(
    query: str,
    contexts: dict[(str, str), np.array]
) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query,
    and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections,
    sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(
            query_embedding,
            doc_embedding
        ), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(
    question: str,
    context_embeddings: dict,
    df: pd.DataFrame,
    show_section=False
) -> str:
    """
    Fetch relevant
    """
    most_relevant_doc_secs = order_document_sections_by_query_similarity(
        question,
        context_embeddings
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_doc_secs:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens.values[0] + \
            separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(
            SEPARATOR +
            document_section.content.values[0].replace("\n", " ")
        )
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    if show_section:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))

    string_list = [str(item) for item in chosen_sections]
    chosen_sections_str = ''.join(string_list)
    header = "Answer the question strictly using the provided context," + \
        " and if the answer is not contained within the text below," + \
        " say 'Sorry, your inquiry is not in the Wiki. For further" + \
        " assistance, please contact caNanoLab-Support@ISB-CGC.org' " + \
        "\n\nContext:\n"
    prompt = header + chosen_sections_str + "\n\n Q: " + question + "\n A:"

    return prompt, chosen_sections_indexes


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False,
    show_source: bool = False
) -> str:
    prompt, chosen_sections_indexes = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0,
        max_tokens=500
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    )
    msg = response.choices[0]['message']['content']
    chosen_sections_indexes = "<br>".join(chosen_sections_indexes)

    return msg, chosen_sections_indexes
