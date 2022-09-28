
from regex import P
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stream_net.model.reinforce_model import reinforce_model
from stream_net.model.heads import *
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from stream_net.data.drop_datasets import drop_datasets
from torch.utils.data import DataLoader


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


class Trainer:
    '''
    model : 一个模型
    head : 一个head
    train_loader : 训练集
    dev_loader : 验证集
    epoch : 迭代次数
    optimizer : 优化器
    device : 设备
    save_path : 保存路径
    '''

    def __init__(self, config, head_type):
        self.head_type = head_type
        pretrained_model = config['model']['pretrained_model']
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokenizer = tokenizer
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True)

        train_drop_data = drop_datasets(
            './drop_datasets/drop_dataset_train_standardized.json', tokenizer, head_type)
        self.train_loader = DataLoader(train_drop_data, batch_size=config['datasets']['batch_size'],
                                       shuffle=False, collate_fn=data_collator)

        dev_drop_data = drop_datasets(
            './drop_datasets/drop_dataset_dev_standardized.json', tokenizer, head_type)
        self.dev_loader = DataLoader(dev_drop_data, batch_size=config['datasets']['batch_size'],
                                     shuffle=False, collate_fn=data_collator)
        input_dims=config['model']['input_dims']
        hidden_dims=config['model']['hidden_dims']
        if head_type == 'single_span':
            head = single_span_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == 'tagged_spans':
            head = tagged_span_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == 'arithmetic':
            head = arithmetic_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == 'count':
            head = count_head(input_dims=input_dims, hidden_dims=hidden_dims)
        elif head_type == 'head_type':
            head = head_classifier_head(input_dims=input_dims, hidden_dims=hidden_dims)
        pretrained_model = config['model']['pretrained_model']
        model = reinforce_model(pretrained_model, head)
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']['learning_rate'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epoch = config['train']['epoch']
        self.save_path = config['model']['model_root_path'] + \
            head_type+'_module.pth'

        self.log_writer = SummaryWriter(config['train']['log_root_path'])
        self.global_train_step = 0
        self.global_dev_step = 0

    def train(self):
        # 训练模型
        self.model = self.model.to(self.device)
        self.model.train()
        with tqdm(self.train_loader) as process_bar:
            for instance in process_bar:
                instance = {k: v.to(self.device)
                                 for k, v in instance.items()}
                loss = self.train_step(
                    self.model, instance, self.optimizer)
                del instance
                self.log_writer.add_scalar(
                    self.head_type+'_train_loss', loss.item(), self.global_train_step)
                process_bar.set_postfix(loss=loss.item())
                self.global_train_step += 1
                #if self.global_train_step > 200:
                #    break
        # torch.save(self.model.state_dict(), self.save_path)

    def train_step(self, model, instance, optimizer):
        # 梯度归零
        optimizer.zero_grad()
        head_output = model(instance)
        loss = model.head.loss_fun(head_output, instance)
        # 计算损失
        loss.backward()
        optimizer.step()
        return loss

    def eval(self,epoch):
        # 在训练时验证模型的训练效果
        self.model = self.model.to(self.device)
        self.model.eval()
        losses = []
        with torch.no_grad():
            with tqdm(self.dev_loader) as process_bar:
                for instance in process_bar:
                    instance = {k: v.to(self.device)
                                     for k, v in instance.items()}
                    head_output = self.model(instance)
                    loss = self.model.head.loss_fun(head_output, instance)
                    del instance
                    self.log_writer.add_scalar(
                        self.head_type+'_dev_loss', loss.item(), self.global_dev_step)
                    self.global_dev_step += 1
                    process_bar.set_postfix(loss=loss.item())
                    losses.append(loss)

        self.log_writer.add_scalar(
            self.head_type+'_average_dev_Loss:', sum(losses)/len(losses),epoch)
        return sum(losses)/len(losses)

    def run(self):
        min_loss=1000000
        patience=0
        for i in range(self.epoch):
            print(self.head_type,"epoch",i)
            self.train()
            loss=self.eval(i)
            if loss<min_loss:
                min_loss=loss
                print('save model epoch: ',i,'at ',self.save_path)
                torch.save(self.model.state_dict(), self.save_path)
                patience=0
            #if patience>=5:
            #    break
            patience+=1
