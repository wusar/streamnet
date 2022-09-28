
import random
import torch
from stream_net.data.answer_generator.answer_generator import AnswerGenerator,get_number_from_word
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any

from word2number.w2n import word_to_num
import re
import itertools


def find_valid_add_sub_expressions_with_rounding(
        numbers: List[int],
        targets: List[int],
        max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
    valid_signs_for_add_sub_expressions = []
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        possible_signs = list(itertools.product(
            (-1, 1), repeat=number_of_numbers_to_consider))
        for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
            indices = [it[0] for it in number_combination]
            values = [it[1] for it in number_combination]
            for signs in possible_signs:
                eval_value = sum(sign * value for sign,
                                 value in zip(signs, values))
                # our added rounding, our only change compared to `find_valid_add_sub_expressions`
                eval_value = round(eval_value, 5)
                # end of our added rounding
                if eval_value in targets:
                    # 0 represents ``not included''.
                    labels_for_numbers = [0] * len(numbers)
                    for index, sign in zip(indices, signs):
                        # 1 for positive, 2 for negative
                        labels_for_numbers[index] = 1 if sign == 1 else 2
                    valid_signs_for_add_sub_expressions.append(
                        labels_for_numbers)
    return valid_signs_for_add_sub_expressions


class Arithmetic_AnswerGenerator(AnswerGenerator):
    """
    used to generate answer for arithmetic question, the ground truth is the signs of numbers occurring in the question and passage
    """
    def __init__(self, tokenizer, max_numbers=200):
        self._tokenizer = tokenizer
        self.max_numbers = max_numbers
        self.special_numbers = [1, 100]  # 1 and 100 are special numbers, we add them to the list of numbers

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

        answer_texts = raw_data['answer']['number']
        target_numbers = []
        for answer_text in answer_texts:
            number = get_number_from_word(answer_text)
            if number is not None:
                target_numbers.append(number)
        number_values = self.special_numbers+number_values
        valid_expr_signs_set = find_valid_add_sub_expressions_with_rounding(
            number_values, target_numbers)

        if len(valid_expr_signs_set) == 0:
            return False
        else:
            if update_instance:
                valid_expr_signs = random.choice(valid_expr_signs_set)
                special_expr_signs = valid_expr_signs[:len(self.special_numbers)]
                normal_expr_signs = valid_expr_signs[len(self.special_numbers):]
                number_indices += [0]*(self.max_numbers-len(number_indices))
                normal_expr_signs += [0]*(self.max_numbers-len(normal_expr_signs))
                number_mask = [1]*(len(number_indices))
                number_mask += [0]*(self.max_numbers-len(number_mask))
                number_mask.insert(0, 1)
                number_mask.insert(0, 1)
                number_values += [0]*(self.max_numbers-len(number_values))
                instance['number_values'] = number_values
                instance['number_indices'] = number_indices
                instance['normal_expr_signs'] = normal_expr_signs
                instance['special_expr_signs'] = special_expr_signs
                instance['number_mask'] = number_mask
                # print(len(instance['number_values']))
                # print(len(instance['number_indices']))
                # print(len(instance['normal_expr_signs']))
                # print(len(instance['special_expr_signs']))
                # print(len(instance['number_mask']))
            return True
