from abc import ABC, abstractmethod
from CAQA import CAQA
from typing import Dict

class BaseEvaluationWrapper(ABC):
    def __init__(self, system):
        self.system = system
        self.has_docs = False

    @abstractmethod
    def generate_response(self, query):
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        pass


class CAQAWrapper(BaseEvaluationWrapper):
    def __init__(self, system: CAQA):
        assert isinstance(system, CAQA)
        super().__init__(system)
        self.has_docs = True

    def generate_response(self, query):
        return self.system.generate_response(query)

    def get_info(self):
        if self.system.fine_tuned:
            return {
                "llm": self.system.peft_model_name,
                "embedding_model": self.system.embedding_model,
                "params": self.system.llm_kwargs
            }
        else:
            return {
                "llm": self.system.language_model_name,
                "embedding_model": self.system.embedding_model,
                "params": self.system.llm_kwargs
            }
