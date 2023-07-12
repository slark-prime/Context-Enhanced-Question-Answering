import transformers
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import pipeline, AutoModelForQuestionAnswering


class CAQA:
    def __init__(self):
        self.qa_chain = None
        self.llm_repo_id = "openai"
        self.embedding_model = "hkunlp/instructor-xl"
        self.llm = None
        self.llm_kwargs = {"temperature": 0.1, "max_new_tokens": 500, "repetition_penalty": 1.2, "device":"cuda:0"}

    def generate_response(self, query):
        """

        :param query:
        :return:
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
        self.chain_type = "stuff"
        self.task = "text-generation"

    def set_embedding_model(self, embedding_model):
        self.caqa.embedding_model = embedding_model
        return self

    def set_llm(self, llm_repo_id):
        self.caqa.llm_repo_id = llm_repo_id
        return self

    def set_llm_params(self, **kwargs):
        self.caqa.llm_kwargs.update(kwargs)
        return self

    def set_chain_type(self, chain_type: str):
        """
        chain_type: "stuff", "map_reduce", "map_rerank", and "refine".
        """
        self.chain_type = chain_type
        return self

    def set_task(self, task):
        self.task = task
        return self

    def load_modal(self):
        model_name = self.caqa.llm_repo_id
        if model_name == "openai":
            self.caqa.llm = OpenAI()

        elif self.task in ['text2text-generation', 'text-generation']:
        
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.caqa.llm = HuggingFacePipeline(
                pipeline=pipeline(
                    "text-generation", model=model, tokenizer=tokenizer, **self.caqa.llm_kwargs
                )
            )
       # else: self.caqa.llm = HuggingFaceHub(repo_id=self.caqa.llm_repo_id,model_kwargs=self.caqa.llm_kwargs)
        elif self.task == "question-answering":
            model_name = self.caqa.llm_repo_id
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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
        self.load_modal()

        chain_type_kwargs = {"prompt":CAQABuilder.build_prompt_template()}
        self.caqa.qa_chain = RetrievalQA.from_chain_type(llm=self.caqa.llm,
                                                         chain_type=self.chain_type,
                                                         retriever=retriever,
                                                         chain_type_kwargs=chain_type_kwargs,
                                                         return_source_documents=True)

    def build(self):
        self.build_qa_chain()
        return self.caqa

    @staticmethod
    def build_prompt_template():
        from langchain.prompts import PromptTemplate
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        return PROMPT