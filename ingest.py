from Loader.LocalDirectoryLoader import LocalDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, INGEST_MODEL
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch

torch.backends.mps.enabled = True
torch.backends.mps.max_concurrency = 1
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

EVAL = False

def main():
    #  Load documents and split in chunks
    dir_loader = LocalDirectoryLoader(SOURCE_DIRECTORY)
    documents = dir_loader.load()

    if EVAL:
        from Eval.DataExtractor import DataExtractor
        DataExtractor().extract_to_jsonl_file(documents, need_list=False)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_documents(documents)


    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    #     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
    #                                                 model_kwargs={"device": "cuda"})
    embeddings = HuggingFaceInstructEmbeddings(model_name=INGEST_MODEL)

    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
