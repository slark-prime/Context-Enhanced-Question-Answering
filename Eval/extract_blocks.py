from langchain.docstore.document import Document
from typing import List
import random
import json


def extract_blocks_from_documents(docs: List[Document], docs_num: int, block_size=4000):
    # see whether the requested text chunks exceeds the document amount
    if docs_num > len(docs):
        docs_num = len(docs)

    selected_docs = random.sample(docs, docs_num)
    blocks = []
    with open("Eval/eval_blocks.json", 'w') as f:
        # get text chunks
        for doc in selected_docs:
            content_length = len(doc.page_content)
            start_index = content_length // 2 - block_size // 2
            end_index = start_index + block_size
            blocks.append(doc.page_content[start_index:end_index])

        # write text chunks into json file
        for i, block in enumerate(blocks, start=1):
            # Construct a dictionary with a single key-value pair, convert to JSON, and write to the file
            line = {
                "block_id": i,
                "text": block
            }
            f.write(json.dumps(line) + "\n")
