

from stream_net.model.heads.head import head
from typing import Dict, List, Union, Tuple, Any
import torch
import torch.nn as nn


class head_classifier_head(head):
    def __init__(self, input_dims=768, hidden_dims=768, head_type_num=4) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, head_type_num)
        self.criterion = torch.nn.MSELoss().cuda()

    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooler_output = head_input['pooler_output']
        hidden_layer = torch.relu(self.layer1(pooler_output))
        head_type_score = self.layer2(hidden_layer)
        # head_output=torch.softmax(head_output,dim=-1)
        head_output = {}
        head_output['head_type_score'] = head_type_score
        return head_output

    def loss_fun(self, head_output: Dict[str, torch.Tensor], instance: Dict[str, torch.Tensor]) -> torch.Tensor:
        head_type_score = head_output['head_type_score']
        ground_truth_label = instance['head_type'].float()
        head_type_score = torch.sigmoid(head_type_score)
        loss = self.criterion(head_type_score, ground_truth_label)
        return loss

    def predict(self, instance) -> Dict[str, any]:
        return self.forward(instance)
        # return decode_output
