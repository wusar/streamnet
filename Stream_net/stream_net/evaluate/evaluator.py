
from regex import P
import torch
from tqdm import tqdm
from stream_net.evaluate.drop_eval import get_metrics as drop_em_and_f1
from torch.utils.tensorboard import SummaryWriter
from stream_net.model.heads import *
from stream_net.model.reinforce_model import reinforce_model
from transformers import AutoTokenizer, DataCollatorWithPadding
from stream_net.data.drop_datasets import drop_datasets
import random
import sys
import io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")
#如果不加入上面这一行，会出现编解码错误的情况
def get_gold_answer_text(gold_answer_message, tokenizer):
    input_ids = gold_answer_message['input_ids']
    answer_as_span_starts = gold_answer_message['answer_as_span_starts'].numpy(
    )
    answer_as_span_ends = gold_answer_message['answer_as_span_ends'].numpy()
    answer_texts = []
    for i in range(input_ids.shape[0]):
        answer_texts.append(tokenizer.decode(
            input_ids[i][answer_as_span_starts[i]:answer_as_span_ends[i]+1]))
    return answer_texts

class assemble_model:
    """
    choose the best head type and get the predict result
    """
    def __init__(self, config,tokenizer=None):
        self.model_dir = config['model']['model_root_path']
        self.config = config
        self.pretrained_model = config['model']['pretrained_model']
        self._tokenizer = tokenizer
        self.id2head = ["arithmetic", "count", "single_span", "tagged_spans"]
        self.models = {}
        self.models['single_span'] = self.load_model('single_span')
        self.models['tagged_spans'] = self.load_model('tagged_spans')
        self.models['arithmetic'] = self.load_model('arithmetic')
        self.models['count'] = self.load_model('count')
        self.models['head_type'] = self.load_model('head_type')

    def load_model(self, head_type):
        state_dict = torch.load(self.model_dir+head_type+'_module.pth')
        hidden_dims = self.config['model']['hidden_dims']
        input_dims = self.config['model']['input_dims']
        if head_type == "single_span":
            head = single_span_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == "tagged_spans":
            head = tagged_span_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == "arithmetic":
            head = arithmetic_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == "count":
            head = count_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == "head_type":
            head = head_classifier_head(input_dims=input_dims, hidden_dims=hidden_dims)

        model = reinforce_model(self.pretrained_model, head)
        model.load_state_dict(state_dict)
        return model

    def predict(self, data_instance):
        with torch.no_grad():
            head_type = self.models['head_type'].predict(data_instance)['head_type_score']
            head_type_scores=head_type.numpy()[0]
            math_scores=head_type_scores[0]+head_type_scores[1]
            span_score=head_type_scores[2]+head_type_scores[3]
            if math_scores>span_score:
                predict_answer=self.models['arithmetic'].predict(data_instance)
                head_type='arithmetic'
                if predict_answer['has_answer']==False:
                    predict_answer=self.models['count'].predict(data_instance)
                    head_type='count'
                predict_text = predict_answer['answer_text']
            else:
                predict_answer=self.models['single_span'].predict(data_instance)
                head_type='single_span'
                if predict_answer['has_answer']==False:
                    predict_answer=self.models['tagged_spans'].predict(data_instance)
                    head_type='tagged_spans'
                predict_text = self._tokenizer.decode(predict_answer['answer_ids'])
                
            print('head_type',head_type)

            return predict_text,head_type


class Evaluator:
    """
    evaluate the model on the test dataset
    """
    def __init__(self, config):
        pretrained_model = config['model']['pretrained_model']
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        dev_datasets=config['datasets']['validation_datasets_path']
        self.dev_loader = drop_datasets(
            dev_datasets, self._tokenizer, 'evaluate')

        self.log_writer = SummaryWriter(config['evaluate']['log_root_path'])
        self.global_dev_step = 0
        self.assemble_model = assemble_model(
            config,self._tokenizer)

    def run(self):
        with tqdm(self.dev_loader) as process_bar:
            head_scores = {}
            for head_type in self.assemble_model.id2head:
                head_scores[head_type] = []
            for data_instance in process_bar:
                predict_text ,head_type= self.assemble_model.predict(data_instance)
                # for predict_answer in predict_answers:
                
                gold_answer_text=data_instance['answer_texts']
                # print(gold_answer_text)
                
                print('predict:',predict_text,'\tground truth',gold_answer_text)
                sys.stdout.flush()
                predict_text = predict_text.lower()
                gold_answer_text=gold_answer_text.lower()
                em, f1 = drop_em_and_f1(predict_text, gold_answer_text)
                print('em:',em,'\tf1:',f1)
                process_bar.set_postfix(avg_em=em,avg_f1=f1)
                head_scores[head_type].append((em,f1))
                self.log_writer.add_scalar(head_type+'_em_score', em, len(head_scores[head_type]))
                self.log_writer.add_scalar(head_type+'_f1_score', f1, len(head_scores[head_type]))
                # self.log_writer.add_scalar('loss', loss, self.global_dev_step)
                self.global_dev_step += 1
                if self.global_dev_step > 1000:
                    break
                
                    
