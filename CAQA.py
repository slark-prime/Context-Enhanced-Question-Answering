import logging
import torch
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
FROM_HUB = True


class CAQA:
    def __init__(self):
        self.qa_chain = None
        self.llm_repo_id = "openai"
        self.embedding_model = "hkunlp/instructor-xl"
        self.llm = None

        # model params that make difference to model generations
        self.llm_kwargs = {
            "temperature": 0.1,
            "max_new_tokens": 500,
            "max_length": 2048,
            "repetition_penalty": 1.2,
            "top_p": 0.95,
            "device": "cuda:0"
        }

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

    def load_embedding_model(self):
        # Embedding Model
        embedding_model = self.caqa.embedding_model
        if embedding_model == "hkunlp/instructor-xl":
            return HuggingFaceInstructEmbeddings(model_name=embedding_model)
        elif embedding_model == "openai":
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings(model_name=embedding_model)

    def load_vectorstore(self):
        # Load the vectorstore
        db = Chroma(persist_directory=PERSIST_DIRECTORY,
                    embedding_function=self.load_embedding_model(),
                    client_settings=CHROMA_SETTINGS)

        return db.as_retriever()

    def load_model(self):

        model_name = self.caqa.llm_repo_id
        logging.info(f"Loading Model: {model_name}")

        if model_name == "openai":
            self.caqa.llm = OpenAI()
            return

        if FROM_HUB:
            self.caqa.llm = HuggingFaceHub(repo_id=self.caqa.llm_repo_id, model_kwargs=self.caqa.llm_kwargs)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name,
                                                             device_map="auto",
                                                             torch_dtype=torch.float16,
                                                             low_cpu_mem_usage=True,
                                                             trust_remote_code=True,
                                                             max_memory={0: "15GB"} # change according to RAM available
                                                             )

                logging.info("Loading Tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Load configuration from the model to avoid warnings
                generation_config = GenerationConfig.from_pretrained(model_name)

                # Create a pipeline for text generation

                self.caqa.llm = HuggingFacePipeline(
                    pipeline=pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        **self.caqa.llm_kwargs
                    )
                )
            except Exception as e:
                logging.error("Error loading the language model: %s", str(e))

            logging.info("Local LLM Loaded")

    def build_qa_chain(self):

        retriever = self.load_vectorstore()

        # Load the language model
        self.load_model()

        chain_type_kwargs = {"prompt": CAQABuilder.build_prompt_template()}
        self.caqa.qa_chain = RetrievalQA.from_chain_type(
            llm=self.caqa.llm,
            chain_type=self.chain_type,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

    def build(self):
        self.build_qa_chain()
        return self.caqa

    @staticmethod
    def build_prompt_template():
        from langchain.prompts import PromptTemplate
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know 
        the answer, just say that you don't know, don't try to make up an answer. 

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        return PROMPT
