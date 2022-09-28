

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, List, Union, Tuple, Any
from stream_net.model.heads.head import head


class reinforce_model(nn.Module):
    def __init__(self, pretrained_model: str, head: head) -> None:
        super().__init__()
        # the pretrained model is a string, which is the name of the pretrained model
        self._transformers_model = AutoModel.from_pretrained(pretrained_model)
        self.head = head

    def forward(self,  instance: Dict[str, torch.LongTensor]) -> Dict[str, Any]:
        # Shape: (batch_size, seqlen, bert_dim)
        # with torch.no_grad():
        token_representations = self._transformers_model(
                instance['input_ids'], token_type_ids=instance['token_type_ids'], attention_mask=instance['attention_mask'])
        # Shape: (batch_size, seqlen, bert_dim)
        instance['token_representations'] = token_representations['last_hidden_state']
        instance['pooler_output'] = token_representations['pooler_output']
        head_output = self.head(instance)
        return head_output
    
    def predict(self, instance) -> Dict[str, Any]:
        self.forward(instance)
        return self.head.predict(instance)
