
import random
from regex import F
import torch
from stream_net.data.answer_generator.answer_generator import AnswerGenerator,get_number_from_word

from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any


class Count_AnswerGenerator(AnswerGenerator):
    """
    used to generate answer for count question, the answer cannot be calculated by the numbers in the question and passage
    """
    def __init__(self, max_numbers=10):
        self.max_numbers = max_numbers
        
    def generate_answer(self, raw_data, instance: Dict[str, Any], update_instance=True) -> None:
        answer_texts = raw_data['answer']['number']
        target_numbers = []
        for answer_text in answer_texts:
            number = get_number_from_word(answer_text)
            if number is not None:
                target_numbers.append(number)
        if len(target_numbers) == 0:
            return False
        elif type(target_numbers[0]) != int:
            return False
        elif target_numbers[0]>self.max_numbers:
            return False
        else:
            if update_instance:
                instance['target_numbers'] = target_numbers[0]
            return True
