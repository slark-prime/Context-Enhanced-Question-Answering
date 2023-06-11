from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY


class CAQA:
    def __init__(self, directoryPath, llmRepoId="openai", embeddingModel="hkunlp/instructor-xl"):
        """

        :param directoryPath: context directory
        :param llmRepoId: llm repo id in huggingface hub
        :param embeddingModel: embedding model used
        """
        self.directoryPath = directoryPath
        self.llmRepoId = llmRepoId
        self.embeddingModel = embeddingModel

        # embedding Model
        if embeddingModel == "hkunlp/instructor-xl":
            embeddings = HuggingFaceInstructEmbeddings(model_name=embeddingModel)

        elif embeddingModel == "openai":
            embeddings = OpenAIEmbeddings()

        else:
            embeddings = HuggingFaceEmbeddings(model_name=embeddingModel)

        # load the vectorstore
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()



        # LLM
        if llmRepoId != "openai":
            self.llm = HuggingFaceHub(repo_id=llmRepoId)
        else:
            self.llm = OpenAI()

        self.qaChain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    def generateResponse(self, query):
        """
        :param query:
        :param chainType: type for load_qa_chain, can be: "stuff", "map_reduce", "map_rerank", "refine"
        :return: Generated Response
        """
        with get_openai_callback() as cb:  # calculate token usage if openai LLM is used
            # build the qa chain
            response = self.qaChain(query)
            answer, sourceDocs = response['result'], response['source_documents']


            if self.llmRepoId == "openai":
                print(cb)

        return answer, sourceDocs
