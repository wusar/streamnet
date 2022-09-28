
from turtle import forward

from matplotlib.pyplot import axis
from sklearn.feature_selection import SelectKBest
from stream_net.model.heads.head import head
from typing import Dict, List, Union, Tuple, Any
import torch
import torch.nn as nn


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    vector = vector + (mask + 1e-13).log()
    return torch.softmax(vector, dim=dim)


class single_span_head(head):
    def __init__(self, input_dims=768, hidden_dims=768,max_span_length=7) -> None:
        super().__init__()
        self.max_span_length = max_span_length
        self.start_output_layer1 = nn.Linear(input_dims, hidden_dims)
        self.start_output_layer2 = nn.Linear(hidden_dims, 1)
        self.end_output_layer1 = nn.Linear(input_dims, hidden_dims)
        self.end_output_layer2 = nn.Linear(hidden_dims, 1)

    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_representations = head_input['token_representations']
        mask = (1-head_input['special_tokens_mask']) * \
            head_input['attention_mask']
        start_hidden_layer = self.start_output_layer1(token_representations)
        # shape: (batch_size, seqlen, hidden_dims)
        start_hidden_layer = torch.relu(start_hidden_layer)
        start_output = self.start_output_layer2(start_hidden_layer)
        # shape: (batch_size, seqlen ,1 )
        end_hidden_layer = self.end_output_layer1(token_representations)
        end_hidden_layer = torch.relu(end_hidden_layer)
        end_output = self.end_output_layer2(end_hidden_layer)
        # shape: (batch_size, seqlen, 1)
        start_output = start_output.squeeze(-1)
        # shape: (batch_size, seqlen)
        end_output = end_output.squeeze(-1)
        # shape: (batch_size, seqlen)
        start_probs = masked_log_softmax(start_output, mask, dim=-1)
        # shape: (batch_size, seqlen)
        end_probs = masked_log_softmax(end_output, mask, dim=-1)
        # shape: (batch_size, seqlen)
        head_output = {}
        head_output['start_probs'] = start_probs
        head_output['end_probs'] = end_probs
        return head_output

    def loss_fun(self, head_output:Dict[str,torch.Tensor], instance: Dict[str, torch.Tensor]) -> torch.Tensor:
        gold_span_starts = instance['answer_as_span_starts']
        gold_span_ends = instance['answer_as_span_ends']
        start_probs = head_output['start_probs']
        end_probs = head_output['end_probs']
        gold_span_starts = gold_span_starts.unsqueeze(-1)
        gold_span_ends = gold_span_ends.unsqueeze(-1)
        log_likelihood_for_span_starts = torch.gather(
            start_probs, 1, gold_span_starts)
        log_likelihood_for_span_ends = torch.gather(
            end_probs, 1, gold_span_ends)
        # Shape: (batch_size, # of answer spans)
        loss = -(log_likelihood_for_span_starts + log_likelihood_for_span_ends)
        loss = loss.mean()
        return loss

    def predict(self, instance) -> Dict[str, any]:
        head_output = self.forward(instance)
        start_probs = head_output['start_probs']
        end_probs = head_output['end_probs']
        start_probs = start_probs.cpu().numpy()
        end_probs = end_probs.cpu().numpy()
        start_pos = start_probs.argmax(axis=-1)
        end_pos = end_probs.argmax(axis=-1)
        input_ids = instance['input_ids'].cpu().numpy()
        
        answer_ids=input_ids[0][start_pos[0]:end_pos[0]+1]
            # answer_texts.append(tokenizer.decode(answer_ids[i]))
        decode_output = {}
        decode_output['answer_as_span_starts'] = start_pos
        decode_output['answer_as_span_ends'] = end_pos
        decode_output['answer_ids'] = answer_ids
        if start_pos>end_pos or end_pos-start_pos>self.max_span_length:
            decode_output['has_answer'] = False
        else:
            decode_output['has_answer'] = True
        # decode_output['answer_texts'] = answer_texts
        return decode_output
