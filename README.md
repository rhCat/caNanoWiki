# caNanoWiki_AI Web Application
This hobby project aims to provide a chatGPT powered digital assistant to help users look for answers in a Wiki. The project is inspired by the OpenAI cookbook  project, but realizing there are a lot of concepts to understand and infrastructures to make it work.

This repository contains a Flask-based web application designed to interact with OpenAI's GPT-3.5 Turbo model. The application is primarily used for answering queries with context, leveraging the capabilities of the GPT-3.5 Turbo model.

## Key Features

1. **Authentication**: The application has a simple authentication system. A user must enter a passcode to access the main page of the application. If the passcode is correct, the user is authenticated and can access the application. The application also includes a timeout feature, where the user is automatically logged out after a certain period of inactivity or after a session duration.

2. **Query Processing**: The application allows users to input queries, which are then processed by the `embedding_qa.answer_query_with_context` function. This function uses a document dataframe and document embeddings (loaded from a pickle file) to provide context for the query.

3. **Interaction with GPT-3.5 Turbo**: The application uses OpenAI's GPT-3.5 Turbo model to generate responses to user queries. The parameters for the model, such as temperature and max tokens, are defined in the application.

4. **Web Interface**: The application provides a web interface for users to interact with. This includes a login page, a logout function, and an index page where users can input queries and view responses.

5. **Configuration**: The application uses a configuration file (`gpt_local_config.cfg`) to set the OpenAI API key and other parameters.

The application is designed to be run on a local server, with the host set to '0.0.0.0' and the port set to 5000.

## Usage

To run the application, navigate to the directory containing the application and run the command `python app.py`. This will start the application on your local server.

## Dependencies

The application requires the following Python packages:

- Flask
- OpenAI
- Tiktoken
- Numpy
- Pandas
- Configparser
- Pickle

These can be installed using pip:

```bash
pip install flask openai tiktoken numpy pandas configparser pickle
