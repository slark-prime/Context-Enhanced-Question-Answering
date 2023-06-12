# Context-Augmented Question Answering (CAQA) System

This repository contains a Context-Augmented Question Answering (CAQA) system. The goal of this project is to build a system that can provide detailed responses to user queries based on the specific context documents provided. This system is highly effective for use cases where responses to queries are context-dependent, such as legal, medical, and technical domains.

## Overview

The system has two main modules:

- **Ingest module**: This module loads the documents, splits them into chunks, creates embeddings using the specified model, and persists them to a Chroma database. 

- **CAQA module**: This module retrieves the context from the Chroma database, uses a large language model (LLM) to generate answers based on the retrieved context, and returns the generated response.

## System Flow

1. **Ingest Documents**: This stage loads documents from a given directory. It supports `.txt`, `.pdf`, and `.csv` files. Each document is divided into chunks of text.

2. **Embedding**: The chunks are transformed into embeddings using HuggingFaceInstructEmbeddings. The embedded documents are persisted in a Chroma database for future retrieval.

3. **Context Retrieval**: During the query process, the system retrieves the relevant context for a given query from the Chroma database.

4. **Question Answering**: With the relevant context at hand, the system generates a response to the query using a Large Language Model (LLM).

## Code Usage

The code consists of three main parts:

- `ingest.py` is used to load the documents, split them into chunks, create the embeddings and persist them into a Chroma database.

- `CAQA.py` is the main system module. It retrieves the context from the Chroma database and uses a large language model (LLM) to generate the answers.

- `main.py` is an example script to interact with the system. It sets up the system, sends a list of queries, gets the responses, and saves the answers to a JSON file.

To interact with the system:

```python
from CAQA import CAQA
from dotenv import load_dotenv

# Specify the path of your context documents
contextPath = "context_files/"
# List your queries
queries = ["What are the Federalist Papers?"]

# Initialize the system
myCAQA = CAQA(contextPath, llmRepoId="openai", embeddingModel="hkunlp/instructor-xl")

# Send queries and get responses
for query in queries:
    answer, sourceDocs = myCAQA.generateResponse(query)
    print("Question: " + query)
    print("Answer: " + answer + '\n')

# Delete the instance after usage
del myCAQA
```

## Dependencies

Make sure to install the dependencies in your Python environment:

```
pip install -r requirements.txt
```

This system has been tested on Python 3.7+. 

## Contribution

Feel free to contribute to the project. Please make sure to read the contribution guidelines before making a pull request.

## License

This project is licensed under the terms of the MIT license.
