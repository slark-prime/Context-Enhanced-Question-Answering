from Eval.EvalPipeline import EvaluationPipeline
from Eval.EvaluationWrappers import CAQAWrapper
from CAQA import *
import json
from dotenv import load_dotenv
import argparse
import ast

# Create the parser
parser = argparse.ArgumentParser(description='Process some inputs.')

# Add the arguments
parser.add_argument('--model', type=str, required=True, help='The model to use')
parser.add_argument('--embedding_model', type=str, required=False, help='The embedding model to use')
parser.add_argument('--params', type=str, required=False, help='The dictionary of parameters to use')
parser.add_argument('--prompt', type=str, required=False, help='The prompt to use')
parser.add_argument('--search_k', type=int, required=False, help='search_k')
parser.add_argument('-f', '--fine-tuned', action='store_true', help='Whether the model is fine-tuned or not')


load_dotenv()

# Default builder
caqa_builder = CAQABuilder()

# Parse the arguments
args = parser.parse_args()

# Use the arguments
model = args.model

if args.fine_tuned:
    customized_builder = caqa_builder.set_peft_model(model).set_chain_type("stuff")
else:
    customized_builder = caqa_builder.set_llm(model).set_chain_type("stuff")

if args.search_k:
    search_k = args.search_k
    customized_builder.set_search_k(search_k)

if args.params:
    params = ast.literal_eval(args.params)
    customized_builder.set_llm_params(**params)

if args.embedding_model:
    embedding_model = args.embedding_model
    customized_builder.set_embedding_model(embedding_model)

if args.prompt:
    prompt = args.prompt
    customized_builder.set_prompt(prompt)

# build the system based on customized builder
myCAQA = customized_builder.build()
print("*******Embedding model used:  " + myCAQA.embedding_model + "*******")
print("*******Large Language Model used:  " + myCAQA.language_model_name+ "*******")

wrappedCAQA = CAQAWrapper(myCAQA)

# Load queries from json line file
with open("Eval/question_block.jsonl", 'r') as f:
    queries = [
        json.loads(line) for line in f
    ]

EvaluationPipeline(wrappedCAQA).generate_answers_to_jsonl_file(question_jsons=queries)
