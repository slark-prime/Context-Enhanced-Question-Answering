
# CAQA: Context-Enhanced Question Answering

CAQA is a Python module that allows you to perform question answering on a directory of documents using different large language models (LLMs) and embeddings. It uses the langchain package to access various LLMs and embeddings from the Hugging Face Hub and OpenAI. It also uses the PyPDF2 and docx packages to read PDF and Word files, respectively.

## Installation

To install CAQA, you need to have Python 3.8 or higher and pip installed on your system. You also need to install the dependencies in requirements.txt


## Usage

To use CAQA, you need to create an instance of the CAQA class by passing the following arguments:

- `directoryPath`: The path to the directory that contains the documents that you want to use as the context for question answering. The documents can be in PDF, Word, or text format.
- `llmRepoId`: The repository ID of the LLM that you want to use for question answering. It can be one of `"openai"`, `"google/flan-t5-small"`, `"mosaicml/mpt-7b"`, or `"gpt2"`. The default value is `"openai"`.
- `embeddingModel`: The name of the embedding model that you want to use for similarity search. It can be one of `"openai"`, `"paraphrase-MiniLM-L6-v2"`, or any other model name supported by the HuggingFaceEmbeddings class. The default value is `"openai"`.

For example:

```python
from CAQA import CAQA
myCAQA = CAQA(directoryPath="data", llmRepoId="google/flan-t5-small", embeddingModel="paraphrase-MiniLM-L6-v2")
```

To generate a response for a given query, you need to call the `generateResponse` method of the CAQA instance by passing the following arguments:

- `query`: The question that you want to ask.
- `chainType`: The type of question answering chain that you want to use. It can be one of `"stuff"`, `"map_reduce"`, `"map_rerank"`, or `"refine"`. The default value is `"stuff"`.

For example:

```python
response = myCAQA.generateResponse(query="Who is the author of Harry Potter?", chainType="map_reduce")
print(response)
```

The output will be something like:

```
J.K. Rowling
```

## Results
<img width="981" alt="image" src="https://github.com/slark-prime/Context-Enhanced-Question-Answering/assets/43880036/60649e61-31c6-4030-84e4-231ec066e34b">
<img width="1138" alt="image" src="https://github.com/slark-prime/Context-Enhanced-Question-Answering/assets/43880036/d309292d-59e0-48ed-bdd9-c2872fbbb911">

