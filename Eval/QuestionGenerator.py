import pyperclip
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict
from abc import ABC, abstractmethod
from Eval.Evaluator import get_openai_response


class BaseQuestionGenerator(ABC):
    """
    Class for generating questions from a data block.
    """

    def __init__(self, output_dir="Eval/Questions"):
        self.output_dir = output_dir

    def _generate_prompt(self, block):
        """Generate prompt to send to GPT-3.5"""
        self.prompt = block + "Please generate four questions based on the provided text. The questions should represent different types, specifically factoid, list, polar (yes/no), and explanatory. Craft each question in a way that it corresponds to a single context within the text, reducing potential for multiple interpretations - use determiners where necessary for clarity. The answer should be locatable within the text. Provide the responses to these questions in an accurate and concise manner.  Format the output like this: {“question”: “The text of the question goes here”, “type”: “The type of the question goes here”, “answer”:”the text of the answer goes here”}"

    @abstractmethod
    def _retrieve_questions(self) -> List[Dict]:
        """Retrieve Questions from GPT3.5 whether by API or by Hands"""
        pass

    def generate_question_jsons(self, block_list: List[str]):
        """
        Generate questions from the data.
        The implementation would depend on your specific use case.
        """
        questions = []
        for block_id, block in enumerate(block_list, start=1):
            self._generate_prompt(block)

            # Assuming you name the downloaded HTML file as 'block.html'
            block_questions = self._retrieve_questions()
            for qid, question_json in enumerate(block_questions, start=1):
                question_json["question_id"] = qid
                question_json["block_id"] = block_id
                questions.append(question_json)

        return questions

    def generate_to_jsonl_file(self, block_list, file_name="questions.jsonl", need_list=False):
        output_path = self.output_dir + file_name
        with open(output_path, 'w') as f:
            # write text chunks into json file
            question_jsons = self.generate_question_jsons(block_list=block_list)
            for question_json in question_jsons:
                # Construct a dictionary with a single key-value pair, convert to JSON, and write to the file
                f.write(json.dumps(question_json) + '\n')

        if need_list:
            return question_jsons

        return file_name


class AutoQuestionGenerator(BaseQuestionGenerator):
    """
    Class for generating questions from a data block.
    """

    def _generate_prompt(self, block):
        super()._generate_prompt(block)
        self.system_prompt = "You are an Question Generator generating questions based on given text and guidance"

    def _retrieve_questions(self) -> List[Dict]:
        """Retrieve Questions from GPT3.5 whether by API or by Hands"""
        questions_str, token_spent = get_openai_response(self.system_prompt, self.prompt, 1024)

        lines = questions_str.strip().split("\n")
        questions = [json.loads(line) for line in lines]

        return questions


class ManualQuestionGenerator(BaseQuestionGenerator):
    """
    Class for generating questions from a data block.
    """

    def __init__(self, html_file):
        super().__init__()
        self.html_file = html_file

    @staticmethod
    def parse_html(html_file) -> List[Dict]:
        """Extract the questions from downloaded HTML."""
        with open(html_file, 'r') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        p_tags = soup.find_all('p')
        json_pattern = re.compile(r'\{.*?}')

        questions = []
        for tag in p_tags:
            text = tag.get_text(strip=True)
            potential_jsons = json_pattern.findall(text)
            for potential_json in potential_jsons:
                try:
                    data = json.loads(potential_json)
                    questions.append(data)
                except json.JSONDecodeError:
                    pass
        return questions

    def _retrieve_questions(self):
        pyperclip.copy(self.prompt)
        return self.parse_html(self.html_file)
