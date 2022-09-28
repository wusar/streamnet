

from turtle import forward

from matplotlib.pyplot import axis
from stream_net.model.heads.head import head
from typing import Dict, List, Union, Tuple, Any
import torch
import torch.nn as nn


class tagged_span_head(head):
    def __init__(self, input_dims=768, hidden_dims=768) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, 2, bias=False)
        
    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_representations = head_input['token_representations']
        hidden_layer = self.layer1(token_representations)
        hidden_layer = torch.relu(hidden_layer)
        ans_score = self.layer2(hidden_layer)
        # head_output=torch.softmax(head_output,dim=-1)
        head_output = {}
        head_output['ans_score'] = ans_score
        return head_output

    def loss_fun(self, head_output: Dict[str, torch.Tensor], instance: Dict[str, torch.Tensor]) -> torch.Tensor:
        ans_score = head_output['ans_score']
        mask = instance['attention_mask'] * \
            (1-instance['special_tokens_mask'])
        ground_truth_label = instance['answer_as_tagged_span']
        ground_truth_label = ground_truth_label[:, 0:ans_score.shape[1]]
        log_probs = torch.log(torch.softmax(ans_score, dim=-1))
        log_likelihoods = \
            torch.gather(log_probs, dim=-1,
                         index=ground_truth_label.unsqueeze(-1)).squeeze(-1)
        log_likelihoods = torch.masked_select(log_likelihoods, mask.bool())
        loss = -log_likelihoods.mean()
        return loss

    def predict(self, instance) -> Dict[str, any]:
        head_output = self.forward(instance)
        mask = instance['attention_mask'] * \
            (1-instance['special_tokens_mask'])
        ans_score = head_output['ans_score']
        is_answer = torch.argmax(ans_score, dim=-1)
        # print("is_answer_probs:", is_answer)
        input_ids = instance['input_ids']
        is_answer = is_answer*mask.float()
        is_answer = is_answer.bool()
        # answer_texts = []
        decode_output = {}
        decode_output['answer_ids'] = torch.masked_select(input_ids[0], is_answer[0])
        # decode_output['answer_texts'] = answer_texts
        return decode_output
