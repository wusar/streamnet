
from turtle import update
import torch
from stream_net.data.answer_generator.answer_generator import AnswerGenerator
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any

def get_answer_type(answer):
    if answer['number']:
        return 'number'
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return 'spans'
        return 'spans'
    elif any(answer['date'].values()):
        return 'date'
    else:
        return None

# head_type: single_span_head and tagged_spans_head = 0
# number_head = 1
# count_head = 2



class head_type_answer_generator(AnswerGenerator):
    """
    used to classify the answer type and choose the correct answer generator
    """
    def __init__(self, tokenizer, answer_generator: Dict[str, AnswerGenerator]) -> None:
        self._tokenizer = tokenizer
        self.answer_generator = answer_generator
        
    def generate_answer(self, raw_data, instance: Dict[str, Any]) -> None:
        head_type=[0 for _ in range(len(self.answer_generator))]
        i=0
        for key, generator in self.answer_generator.items():
            if generator.generate_answer(raw_data, instance,update_instance=False):
                head_type[i]=1
            i+=1
        if sum(head_type)==0:
            return False
        else:
            instance['head_type'] = head_type
            return True
