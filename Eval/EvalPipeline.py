# Import required modules
import json
from Eval.Evaluator import Evaluator
from Eval.QuestionGenerator import AutoQuestionGenerator
from Eval.DataExtractor import DataExtractor
from Loader.LocalDirectoryLoader import LocalDirectoryLoader
from Eval.EvaluationWrappers import BaseEvaluationWrapper
import datetime
import logging
from typing import List, Dict

# Initialize logger
logger = logging.getLogger(__name__)


# Define the class for evaluation pipeline
class EvaluationPipeline:

    # Initialize the pipeline with the system
    def __init__(self, system: BaseEvaluationWrapper, output_dir='Eval/'):
        """
        Initializes the EvaluationPipeline instance.

        Parameters
        ----------
        system : BaseEvaluationWrapper
            The system (a model or API that generates responses) for which we want to run the evaluation pipeline.
        """
        # Define text extractor, question generator, evaluator, and system attributes
        self.text_extractor = DataExtractor(extract_num=30, output_dir="Eval")
        self.question_generator = AutoQuestionGenerator(output_dir="Eval")
        self.evaluator = Evaluator()
        self.output_dir = output_dir
        self.system = system

    # Define the main pipeline function
    def run(self):
        """
        Executes the entire evaluation pipeline.
        """
        # Load documents from directory
        docs = self.load_data(source_dir="context_files")

        # Extract blocks of text from documents and save to jsonl file
        block_list = self.text_extractor.extract_to_jsonl_file(docs,
                                                               file_name="Blocks/eval_blocks.jsonl",
                                                               need_list=True
                                                               )

        # Generate questions from extracted blocks and save to jsonl file
        question_jsons = self.question_generator.generate_to_jsonl_file(block_list,
                                                                        file_name="Questions/question_block.jsonl",
                                                                        need_list=True
                                                                        )

        # Generate answers for the questions and save to jsonl file
        answer_jsonl_file_name = self.generate_answers_to_jsonl_file(question_jsons)

        # Evaluate the generated answers and save to jsonl file
        self.evaluator.evaluate_to_jsonl("question_block.jsonl", answer_jsonl_file_name)

        # Log the path of the evaluation jsons
        logger.info("The evaluation jsons are stored in the {path}".format(path=answer_jsonl_file_name))

    # Define the function for generating answers
    def generate_answers(self, question_jsons) -> List[Dict]:
        """
        Generates answers for the provided questions using the system (model/API).

        Parameters
        ----------
        question_jsons : list[dict]
            List of questions in the form of JSON objects.

        Returns
        -------
        answer_jsons : List[Dict]
            List of generated answers in the form of JSON objects.
        """
        # Extract queries and their ids from question_jsons
        queries_with_id = [(question_json["question"], question_json["question_id"]) for question_json in
                           question_jsons]
        answer_jsons = []

        # Generate answers for each query and append to answer_jsons
        for query, qid in queries_with_id:
            if self.system.has_docs:
                answer, docs = self.system.generate_response(query)
            else:
                answer = self.system.generate_response(query)

            logger.info("QUESTION: " + query + '\n' + "ANSWER: " + answer)
            # Add answer and system info to the answer_jsons list
            answer_jsons.append(
                {
                    **{
                        "question_id": qid,
                        "answer": answer
                    },
                    **self.system.get_info()
                }
            )
        return answer_jsons

    # Define the function for saving answers to jsonl file
    def generate_answers_to_jsonl_file(self, question_jsons: List[Dict], file_name=None, need_list=False):
        """
       Generates answers and saves them to a JSONL file.

       Parameters
       ----------
       file_name : str, optional
           The name of the file to which the answers will be written.
       need_list : bool, optional
           If True, returns the list of answers. Default is False.
       question_jsons: List[Dict]
       Returns
       -------
       str or List[Dict]
           Depending on the value of `need_list`, either returns the file name or the list of answers.

       """
        # Generate answers and get the current date
        answer_jsons = self.generate_answers(question_jsons)
        date = datetime.datetime.now()

        # Define the file name for the answers jsonl file
        answer_jsonl_file_name = f"answers_{date}.jsonl".format(date=date.strftime("%Y-%m-%d"))
        file_path = self.output_dir + '/' + answer_jsonl_file_name

        # Write answers to the jsonl file
        with open(file_path, 'a') as f:
            for answer_json in answer_jsons:
                f.write(json.dumps(answer_json) + '\n')

        # Return the list of answers or the file name based on need_list
        if need_list:
            return answer_jsons

        return file_name

    # Define the function for loading data
    @staticmethod
    def load_data(source_dir="context_files"):
        """
        Loads data from a specified directory.

        Parameters
        ----------
        source_dir : str, optional
            The directory from which to load the data. Default is "context_files".

        Returns
        -------
        list[Document]
            The loaded data.
        """
        # Initialize a directory loader
        dir_loader = LocalDirectoryLoader(source_dir)

        # Load and return data from the directory
        return dir_loader.load()
