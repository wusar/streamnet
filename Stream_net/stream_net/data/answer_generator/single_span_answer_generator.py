
import torch
from stream_net.data.answer_generator.answer_generator import AnswerGenerator
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any

IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + \
    "".join([u"‘", u"’", u"´", u"`", "_"])


def find_valid_spans(passage_token_ids: List[int],
                     all_answer_token_ids: List[List[int]]) -> List[Tuple[int, int]]:
    word_positions: Dict[int, List[int]] = defaultdict(list)
    for i, token_id in enumerate(passage_token_ids):
        word_positions[token_id].append(i)
    spans = []
    for answer_token_id in all_answer_token_ids:
        num_answer_token_ids = len(answer_token_id)
        if answer_token_id[0] not in word_positions:
            continue
        for span_start in word_positions[answer_token_id[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_token_ids and span_end + 1 < len(passage_token_ids):
                token_id = passage_token_ids[span_end + 1]
                if answer_token_id[answer_index] == token_id:
                    answer_index += 1
                    span_end += 1
                else:
                    break
            if num_answer_token_ids == answer_index:
                spans.append((span_start, span_end))
    return spans


class Single_Span_AnswerGenerator(AnswerGenerator):
    """
    generate answer as span starts and ends
    """
    def __init__(self, tokenizer, max_seq_length=512):
        self._tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def generate_answer(self, raw_data, instance: Dict[str, Any],update_instance=True) -> None:
        question_passage_ids = instance['input_ids']
        answer_texts = raw_data['answer']["spans"]
        answer_tokens = [self._tokenizer.tokenize(
            answer_texts[i]) for i in range(len(answer_texts))]
        answer_tokens_ids = [self._tokenizer.convert_tokens_to_ids(
            answer_tokens[i]) for i in range(len(answer_tokens))]
        # passage_tokens = split_tokens_by_hyphen(passage_tokens)
        spans = find_valid_spans(question_passage_ids, answer_tokens_ids)
        if len(spans) != 1:
            return False
        else:
            if update_instance:
                instance['answer_as_span_starts'] = spans[0][0]
                instance['answer_as_span_ends'] = spans[0][1]
            return True
