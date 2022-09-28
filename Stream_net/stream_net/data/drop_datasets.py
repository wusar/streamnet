from typing import Dict
from torch.utils.data.dataset import IterableDataset
import json
# from reinforce_net.data.answer_generator.tagged_spans_answer_generator import Tagged_Spans_AnswerGenerator
# from reinforce_net.data.answer_generator.single_span_answer_generator import Single_Span_AnswerGenerator
# from reinforce_net.data.answer_generator.arithmetic_generator import Arithmetic_AnswerGenerator
# from reinforce_net.data.answer_generator.count_generator import Count_AnswerGenerator
# from reinforce_net.data.answer_generator.head_type_answer_generator import head_type_answer_generator
from .answer_generator import *

def get_answer_type(answer):
    if answer['number']:
        return 'number'
    elif answer['spans']:
        return 'spans'
    elif any(answer['date'].values()):
        return 'date'
    else:
        return None

class drop_datasets(IterableDataset):
    """
    file_path: path to the dataset file
    tokenizer: tokenizer to be used
    answer_type: type of answer to be generated
    max_length: maximum length of the answer
    """

    def __init__(self, file_path: str, tokenizer, answer_generator_name: str, max_length=512) -> None:
        super().__init__()
        with open(file_path, encoding='utf8') as dataset_file:
            self.dataset = json.load(dataset_file)
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.answer_generator_name = answer_generator_name
        normal_answer_generator = {}
        normal_answer_generator['arithmetic'] = Arithmetic_AnswerGenerator(tokenizer)
        normal_answer_generator['count'] = Count_AnswerGenerator()
        normal_answer_generator['single_span'] = Single_Span_AnswerGenerator(
            tokenizer, self.max_length)
        normal_answer_generator['tagged_spans'] = Tagged_Spans_AnswerGenerator(
            tokenizer, self.max_length)
        if self.answer_generator_name == 'head_type':
            self.answer_generator = head_type_answer_generator(
                self._tokenizer, normal_answer_generator)
        elif self.answer_generator_name == 'evaluate':
            self.answer_generator = evaluate_AnswerGenerator(self._tokenizer)
        else:
            self.answer_generator = normal_answer_generator[self.answer_generator_name]
        self.answer_type2generator_name = {}
        self.answer_type2generator_name['spans'] = [
            'single_span', 'tagged_spans','head_type','evaluate']
        self.answer_type2generator_name['number'] = ['arithmetic','count','head_type','evaluate']
        self.answer_type2generator_name['date'] = ['arithmetic','count','head_type','evaluate']
        self.answer_type2generator_name[None] = []

    def __iter__(self):
        for passage_id, passage_info in self.dataset.items():
            passage_text = passage_info['passage']
            for question_index, qa_pairs in enumerate(passage_info['qa_pairs']):
                answer = qa_pairs['answer']
                if self.answer_generator_name not in self.answer_type2generator_name[get_answer_type(answer)]:
                    continue
                question_text = qa_pairs['question']
                # for passage_text_part in self._split_passage(passage_text, self.max_length-len(question_text)):
                raw_data = dict()
                raw_data['passage_text'] = passage_text
                raw_data['question_text'] = question_text
                raw_data['answer'] = answer
                raw_data['answer_type'] = get_answer_type(answer)
                # raw_data['metadata'] = (passage_id, question_index)
                # turn the raw data into an instance
                instance = self.text_to_instance(raw_data)
                if instance is not None:
                    yield instance
                else:
                    continue

    def text_to_instance(self, raw_data: dict) -> dict:
        question_text = raw_data['question_text']
        passage_text = raw_data['passage_text']
        # get the tokenized question and passage ids and masks
        encoded_inputs = self._tokenizer.encode_plus(question_text, passage_text,
                                                     add_special_tokens=True, max_length=self.max_length,
                                                     truncation_strategy='only_second',
                                                     return_token_type_ids=True,
                                                     return_special_tokens_mask=True, truncation=True, padding=True)
        instance = {}
        instance['input_ids'] = encoded_inputs['input_ids']
        instance['token_type_ids'] = encoded_inputs['token_type_ids']
        instance['attention_mask'] = encoded_inputs['attention_mask']
        instance['special_tokens_mask'] = encoded_inputs['special_tokens_mask']
        # use the answer generator to preprocess the answer
        if self.answer_generator.generate_answer(raw_data, instance) == False:
            return None
        else:
            return instance