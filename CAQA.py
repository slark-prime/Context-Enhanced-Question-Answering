from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

class CAQA:
    def __init__(self):
        self.qa_chain = None
        self.llm_repo_id = "openai"
        self.embedding_model = "hkunlp/instructor-xl"
        self.llm = None
        self.llm_kwargs = {"temperature": 0.1, "max_new_tokens": 500}

    def generate_response(self, query):
        """
        :param query:
        :param chainType: type for load_qa_chain, can be: "stuff", "map_reduce", "map_rerank", "refine"
        :return: Generated Response
        """
        with get_openai_callback() as cb:  # calculate token usage if openai LLM is used
            # build the qa chain
            response = self.qa_chain(query)
            answer, source_docs = response['result'], response['source_documents']

            if self.llm_repo_id == "openai":
                print(cb)

        return answer, source_docs


class CAQABuilder:
    def __init__(self):
        self.caqa = CAQA()

    def set_embedding_model(self, embedding_model):
        self.caqa.embedding_model = embedding_model
        return self

    def set_llm(self, llm_repo_id):
        self.caqa.llm_repo_id = llm_repo_id
        return self

    def set_llm_params(self, **kwargs):
        self.caqa.llm_kwargs.update(kwargs)
        return self

    def build_qa_chain(self):
        # embedding Model
        if self.caqa.embedding_model == "hkunlp/instructor-xl":
            embeddings = HuggingFaceInstructEmbeddings(model_name=self.caqa.embedding_model)

        elif self.caqa.embedding_model == "openai":
            embeddings = OpenAIEmbeddings()

        else:
            embeddings = HuggingFaceEmbeddings(model_name=self.caqa.embedding_model)

        # load the vectorstore
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()

        # LLM
        if self.caqa.llm_repo_id != "openai":
            self.llm = HuggingFaceHub(repo_id=self.caqa.llm_repo_id,
                                      model_kwargs=self.caqa.llm_kwargs)
        else:
            self.caqa.llm = OpenAI()

        self.caqa.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever,
                                                         return_source_documents=True)

    def build(self):
        self.build_qa_chain()
        return self.caqa
