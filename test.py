from CAQA import *
import json
from dotenv import load_dotenv

# load queries from json line file
with open("question_block.jsonl", 'r') as f:
    queries = [
        json.loads(line)["question"] for line in f
    ]

load_dotenv()

llm = ["google/flan-t5-xxl",
       "mosaicml/mpt-7b-instruct",
       "tiiuae/falcon-7b-instruct",
       "tiiuae/falcon-7b",
       "bigscience/bloom-560m",
       "bigscience/bloomz",
       "lmsys/vicuna-7b-v1.3"]


embedding_model = "hkunlp/instructor-xl"

# default builder
caqa_builder = CAQABuilder()

# customized builder
customized_builder = caqa_builder.set_llm(llm[1])\
                    .set_embedding_model(embedding_model)\
                    .set_llm_params(temperature = 0.5, max_new_tokens = 250)\
                    .set_chain_type("stuff")

# build the system based on customized builder
myCAQA = customized_builder.build()
print("*******Embedding model used:  " + myCAQA.embedding_model + "*******")
print("*******Large Language Model used:  " + myCAQA.llm_repo_id + "*******")

answers = []
for query in queries:
    answer, source_docs = myCAQA.generate_response(query)
    answers.append(answer)
    print("Question: " + query)
    print(len(source_docs))
    print("Answer: " + answer + '\n')
    print("*****************")

print(len(answers))

while True:
    query = input("\nEnter a query: ")

    if query == "cont":
        break

file_name = "answers_" + myCAQA.llm_repo_id.split('/')[-1] + "_" + myCAQA.embedding_model.split('/')[-1] + ".jsonl"
# Write to answers to json line file
with open(file_name, 'w') as f:
    for i, answer in enumerate(answers, start=1):
        line = {
            "question_id": i,
            "answer": answer,
            "LLM": myCAQA.llm_repo_id,
            "embedding_model": myCAQA.embedding_model
        }
        f.write(json.dumps(line) + "\n")

