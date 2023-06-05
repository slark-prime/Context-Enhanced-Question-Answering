from CAQA import *
from dotenv import load_dotenv


load_dotenv()
trainPath = "context_files/"
queries = [
    "What is the main objective of the paper 'INSTRUCTION TUNING WITH GPT-4'?",
    "What is the methodology used by the authors in the paper 'INSTRUCTION TUNING WITH GPT-4'?",
    "What are the key findings or results of the study presented in the paper 'INSTRUCTION TUNING WITH GPT-4'?",
    "How does the paper 'INSTRUCTION TUNING WITH GPT-4' contribute to the understanding or development of GPT-4?",
    "What are the implications of the study 'INSTRUCTION TUNING WITH GPT-4' for future research or applications?",
    "How to fine-tune GPT4?"
]


myCAQA = CAQA(trainPath, llmRepoId="google/flan-t5-small", embeddingModel="paraphrase-MiniLM-L6-v2")
# "google/flan-t5-small","mosaicml/mpt-7b", "gpt2"
for query in queries:

    # "stuff", "map_reduce", "map_rerank", "refine"
    response = myCAQA.generateResponse(query, chainType="stuff")

    print("Embedding model used:  " + myCAQA.embeddingModel)
    print("Large Language Model used:  " + myCAQA.llmRepoId)
    print("Question: " + query)
    print("Answer: " + response)