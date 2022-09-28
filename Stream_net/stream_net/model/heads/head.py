

from typing import Dict, List, Union, Tuple, Any
import torch


class head(torch.nn.Module):
    # the loss_function is a function that takes in the output of the head and the target and return the loss
    def loss_fun(self, head_output: Dict[str, torch.Tensor], instance: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    # the predict function is used to generate the answer for the question
    def predict(self, instance: Dict[str, any]) -> Dict[str, any]:
        raise NotImplementedError
    
    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
