
from tkinter.messagebox import NO
from turtle import forward

from matplotlib.pyplot import axis
from stream_net.model.heads.head import head
from typing import Dict, List, Union, Tuple, Any
import torch
import torch.nn as nn


class count_head(head):
    def __init__(self, input_dims=768, hidden_dims=768, max_count=10) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, max_count)
        self.critierion = nn.CrossEntropyLoss().cuda()

    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooler_output = head_input['pooler_output']
        hidden_layer = torch.relu(self.layer1(pooler_output))
        count_scores = self.layer2(hidden_layer)
        # shape: (batch_size, seqlen)
        head_output = {}
        head_output['count_scores'] = count_scores
        return head_output

    def loss_fun(self, head_output: Dict[str, torch.Tensor], instance: Dict[str, torch.LongTensor]) -> torch.Tensor:
        predict_score = head_output['count_scores']
        target = instance['target_numbers']
        # print(predict_score, target)
        # print(predict_score.shape, target.shape)
        loss = self.critierion(predict_score, target)
        return loss

    def predict(self, instance) -> Dict[str, any]:
        head_output=self.forward(instance)
        num=head_output['count_scores'].argmax(dim=1)
        decode_output = {}
        decode_output['answer_text'] = str(num.numpy()[0])
        return decode_output
        # return decode_output
