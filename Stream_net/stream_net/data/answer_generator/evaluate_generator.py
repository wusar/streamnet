
import random
from turtle import shape
from regex import F
import torch
from stream_net.data.answer_generator.answer_generator import AnswerGenerator,get_number_from_word
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any


class evaluate_AnswerGenerator(AnswerGenerator):
    def __init__(self,tokenizer,max_numbers=200) -> None:
        self._tokenizer = tokenizer
        self.max_numbers=max_numbers
        self.special_numbers=[1,100]
    def generate_answer(self, raw_data, instance: Dict[str, Any], update_instance=True) -> None:
        question_passage_ids = instance['input_ids']
        question_passage_tokens = []
        for i in question_passage_ids:
            question_passage_tokens.append(
                self._tokenizer.convert_ids_to_tokens([i])[0])
        number_indices = []
        number_values = []

        for i in range(len(question_passage_tokens)):
            num = get_number_from_word(question_passage_tokens[i])
            if num is not None:
                number_indices.append(i)
                number_values.append(num)

        number_values = self.special_numbers+number_values
        number_indices += [0]*(self.max_numbers-len(number_indices))
        number_mask = [1]*(len(number_indices))
        number_mask += [0]*(self.max_numbers-len(number_mask))
        number_mask.insert(0, 1)
        number_mask.insert(0, 1)
        number_values += [0]*(self.max_numbers-len(number_values))
        instance['number_values'] = number_values
        instance['number_indices'] = number_indices
        instance['number_mask'] = number_mask
        for key in instance.keys():
            instance[key] = torch.tensor(instance[key]).unsqueeze_(0)
        if raw_data['answer_type']=='date':
            return False
        if raw_data['answer_type']=='number':
            instance['answer_texts'] = str(raw_data['answer']['number'])
        if raw_data['answer_type']=='spans':
            answer_text=''
            for i in range(len(raw_data['answer']['spans'])):
                answer_text+=' '+raw_data['answer']['spans'][i]
            instance['answer_texts'] = answer_text#raw_data['answer']['spans'][0]
        return True
