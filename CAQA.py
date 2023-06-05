from PyPDF2 import PdfReader
import docx
import os

from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



# Functions to read different file types
def readPdf(filePath):
    with open(filePath, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for pageNum in range(len(reader.pages)):
            text += reader.pages[pageNum].extract_text()
    return text


def readWord(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def readTxt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text


def readDocumentsFromDirectory(directory):
    combinedText = ""
    for filename in os.listdir(directory):
        filePath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combinedText += readPdf(filePath)
        elif filename.endswith(".docx"):
            combinedText += readWord(filePath)
        elif filename.endswith(".txt"):
            combinedText += readTxt(filePath)
    return combinedText



class CAQA:
    model_upper_limit_k = { # maximum k documents for similarity search
        'gpt2': 3,
        'google/flan-t5-small': 6,
        'mosaicml/mpt-7b': 3    # model is too big
    }

    def __init__(self, directoryPath, llmRepoId="openai", embeddingModel="openai"):
        '''

        :param directoryPath: context directory
        :param llmRepoId: llm repo id in huggingface hub
        :param embeddingModel: embedding model used
        '''
        self.directoryPath = directoryPath
        self.llmRepoId = llmRepoId
        self.embeddingModel = embeddingModel

        text = readDocumentsFromDirectory(directoryPath)

        # Split the text string to piled chunks
        spliter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        textChunks = spliter.split_text(text)

        # Check embedding

        # for example 'paraphrase-MiniLM-L6-v2'
        if embeddingModel != "openai":
            # model = SentenceTransformer(embeddingModel)
            embedding = HuggingFaceEmbeddings(model_name=embeddingModel)
            # embeddings = model.encode(textChunks)
        else:
            embedding = OpenAIEmbeddings()

        # do some similarity search
        self.docSearch = FAISS.from_texts(textChunks, embedding)

        # "google/flan-t5-small","mosaicml/mpt-7b", "gpt2"
        if llmRepoId != "openai":
            self.llm = HuggingFaceHub(repo_id=llmRepoId)
        else:
            self.llm = OpenAI()

    def generateResponse(self, query, chainType = "stuff"):
        '''

        :param query:
        :param chainType: type for load_qa_chain, can be: "stuff", "map_reduce", "map_rerank", "refine"
        :return: Generated Response
        '''
        with get_openai_callback() as cb:  # calculate token usage
            # build the qa chain
            chain = load_qa_chain(self.llm, chain_type=chainType)

            searchResultForQuery = self.docSearch.similarity_search(query, k=self.model_upper_limit_k[self.llmRepoId])

            response = chain.run(input_documents=searchResultForQuery, question=query)

            if self.llmRepoId == "openai":
                print(cb)

        return response
