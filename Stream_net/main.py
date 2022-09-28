

import sys


from stream_net.train.trainer import Trainer
from stream_net.evaluate.evaluator import Evaluator
import torch
import json
import argparse

parser = argparse.ArgumentParser(
    usage='python main.py -m <mode> -t <head_type> -c <config_path>', description='mode: train or eval\nhead_type: single_span, tagged_spans, arithmetic, count, head_type\nconfig_path: path to config file')

parser.add_argument('-m', '--mode', dest='mode', type=str,
                    default='train', help='train or eval', choices=['train', 'eval'])

parser.add_argument('-t', '--head_type', dest='head_type',
                    type=str, default='head_type', help='head_type: single_span, tagged_spans, arithmetic, count, head_type', choices=['single_span', 'tagged_spans', 'arithmetic', 'count', 'head_type'])

parser.add_argument('-c', '--config', dest='config',
                    type=str, default='./config.json', help='the path of a json config file')


args = parser.parse_args()
head_type = args.head_type
config_file = args.config
mode = args.mode
config = json.load(open(config_file))
if mode == 'train':
    trainer = Trainer(config, head_type)
    trainer.run()
if mode == 'eval':
    evaluator=Evaluator(config)
    evaluator.run()
