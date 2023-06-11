from CAQA import *
from dotenv import load_dotenv
import json

def queryAnswerJson(model):
    load_dotenv()
    contextPath = "context_files/"
    queries = [
        "What is the main purpose of the United States Constitution?",
        "Who are considered the 'Framers' of the Constitution?",
        "How many amendments have been made to the Constitution as of 2021?",
        "What are the three branches of government established by the Constitution, and what are their primary functions?",
        "What is the significance of the Bill of Rights?",
        "How can an amendment be added to the U.S. Constitution?",
        "What is the Supremacy Clause and why is it important?",
        "Explain the concept of 'checks and balances' as outlined in the Constitution.",
        "What are some of the powers granted to Congress by the Constitution?",
        "What are some of the key protections offered by the First Amendment?",
        "What is the process for impeachment as outlined in the Constitution?",
        "Why was the Constitution initially criticized for not having a declaration of individual rights?",
        "How does the Constitution address the issue of state sovereignty?",
        "What does the term 'enumerated powers' mean in the context of the Constitution?",
        "What is the Electoral College and how is it defined in the Constitution?",
        "Describe the concept of 'separation of powers' as established by the Constitution.",
        "What is the 14th Amendment and why is it significant?",
        "What is the purpose of the Preamble to the Constitution?",
        "Who was the oldest and the youngest signer of the U.S. Constitution?",
        "What are the Federalist Papers and what role did they play in the ratification of the Constitution?"
    ]

    model_answers = {}

    myCAQA = CAQA(contextPath, llmRepoId=model, embeddingModel="hkunlp/instructor-xl")

    model_answers[model] = {}

    print("*******Embedding model used:  " + myCAQA.embeddingModel + "*******")
    print("*******Large Language Model used:  " + myCAQA.llmRepoId + "*******")

    for query in queries:
        answer, sourceDocs = myCAQA.generateResponse(query)
        model_answers[model][query] = answer

        print("Question: " + query)
        print("Answer: " + answer + '\n')
        print("*****************")

    # Save the answers to a JSON file
    with open('model_answers.json', 'w') as json_file:
        json.dump(model_answers, json_file, indent=4)

    del myCAQA

