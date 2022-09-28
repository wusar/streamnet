

from turtle import forward

from matplotlib.pyplot import axis
from stream_net.model.heads.head import head
from stream_net.data.answer_generator.arithmetic_generator import get_number_from_word
from typing import Dict, List, Union, Tuple, Any
import torch
import torch.nn as nn


class arithmetic_head(head):
    def __init__(self, input_dims=768, hidden_dims=768) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dims*2, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, 3)
        # 由于1和100在实际问题中很常见，因此我们将其专门设置为一个特殊的数字，与文本的数字信息一起送入模型
        self.special_nums = [1, 100]
        special_embedding_dim = input_dims
        self.special_num_embeddings = torch.nn.Embedding(
            len(self.special_nums), special_embedding_dim)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, head_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_representations = head_input['token_representations']
        pooler_output = head_input['pooler_output']
        number_indices = head_input['number_indices']
        encoded_numbers = torch.gather(
            token_representations, 1, number_indices.unsqueeze(-1).expand(-1, -1, token_representations.size(-1)))
        special_numbers_embd = self.special_num_embeddings(torch.arange(
            len(self.special_nums), device=number_indices.device))
        special_numbers_embd = special_numbers_embd.expand(
            number_indices.shape[0], -1, -1)
        encoded_numbers = torch.cat([special_numbers_embd, encoded_numbers], 1)
        encoded_numbers = torch.cat(
            [encoded_numbers, pooler_output.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)
        # print("encoded_numbers:", encoded_numbers.shape)
        hidden_layer1 = torch.relu(self.layer1(encoded_numbers))
        predicted_expr_signs = self.layer2(hidden_layer1)
        output = {}
        output['predicted_expr_signs'] = predicted_expr_signs
        return output

    def loss_fun(self, head_output:Dict[str,torch.Tensor], head_input: Dict[str, torch.LongTensor]) -> torch.Tensor:
        predicted_expr_signs=head_output['predicted_expr_signs']
        mask = head_input['number_mask']
        normal_expr_signs = head_input['normal_expr_signs']
        special_expr_signs = head_input['special_expr_signs']
        expr_signs = torch.cat([special_expr_signs, normal_expr_signs], 1)
        log_probs = torch.log(torch.softmax(predicted_expr_signs, dim=-1))
        log_likelihoods = \
            torch.gather(log_probs, dim=-1,
                         index=expr_signs.unsqueeze(-1)).squeeze(-1)
        log_likelihoods = torch.masked_select(log_likelihoods, mask.bool())
        loss = -log_likelihoods.mean()

        return loss

    def predict(self, instance) -> Dict[str, any]:
        head_output = self.forward(instance)
        predicted_expr_signs = head_output['predicted_expr_signs']
        norm_values = instance['number_values']
        number_values = norm_values.cpu().numpy()
        predicted_expr = torch.argmax(predicted_expr_signs, dim=-1)
        predicted_expr = predicted_expr.cpu().numpy()
        decode_output = {}
        decode_output['predicted_expr'] = predicted_expr
        answer_text = []
        answer_num = 0
        has_answer = False
        for j in range(predicted_expr.shape[1]):
            if predicted_expr[0][j] == 0:
                continue
            elif predicted_expr[0][j] == 1:
                has_answer = True
                answer_num += number_values[0][j]
            elif predicted_expr[0][j] == 2:
                has_answer = True
                answer_num -= number_values[0][j]
        answer_text=str(answer_num)
        if has_answer:
            decode_output['has_answer'] = True
        else:
            decode_output['has_answer'] = False
        decode_output['answer_text'] = answer_text
        return decode_output
