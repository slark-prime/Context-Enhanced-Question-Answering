
# Context-Augmented Question Answering (CAQA) System

The CAQA system is a sophisticated question-answering system that uses state-of-the-art language models, embeddings, and retrieval techniques to provide accurate and context-aware answers to user queries.

## Overview

The system consists of two main components, represented by the `CAQA.py` and `ingest.py` scripts:

- `CAQA.py`: Defines the `CAQA` and `CAQABuilder` classes, which represent the CAQA system and a builder for the system respectively. The `CAQA` system uses a language model and a question-answering chain to generate responses to queries.

- `ingest.py`: A script that loads documents from a local directory, splits them into chunks, and creates embeddings for the chunks using HuggingFace InstructEmbeddings. The embeddings are then stored in a Chroma vector store.

## Requirements

This code is written in Python and uses several libraries, including:

- PyTorch
- HuggingFace Transformers
- LangChain

Please ensure these dependencies are installed in your environment. 

## Usage

### CAQA

Here's an example of how to use the `CAQA.py` script to create a CAQA system:

```python
from CAQA import CAQABuilder

# Initialize the builder
builder = CAQABuilder()

# Configure the builder in the chain format
customized_builder = builder.set_embedding_model('embedding_model')
                            .set_llm('llm_repo_id')
                            .set_llm_params(temperature=0.001, max_new_tokens=500, max_length=2048)
                            .set_chain_type('chain_type')

# Or one by one
customized_builder.set_prompt(prompt)

# Build the CAQA
caqa = builder.build()

# Use the CAQA to generate a response to a query
response = caqa.generate_response('query')
```

### Ingest

You can run the `ingest.py` script to load and process your documents:

```bash
python ingest.py
```

Please replace `'embedding_model'`, `'llm_repo_id'`, `'chain_type'`, and `'query'` with your actual values.

## Contributing

Contributions are welcome! Please read the contributing guidelines to get started.

## License

This project is licensed under the terms of the MIT license.

