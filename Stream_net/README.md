本代码是对论文《StreamNet: A Combination Model For Answering Multi-task Reading Comprehension Problems》中的人工神经算法的实现，也是对项目["A Simple and Effective Model for Answering Multi-span Questions"](http://arxiv.org/abs/1909.13375)(https://github.com/eladsegal/tag-based-multi-span-extraction)的复现。


### Dependencies
*   Python 3.6+.
*   [Numpy 1.16]
*   [torch 1.11.0+cu113]
*   [tqdm]
*   [word2number 1.1]
*   [transformers 4.18.0]

## Setup

在安装完python后，使用命令：
pip install -r requirements.txt
就可以安装本项目所需要的库
### Commands
使用命令
python main.py --mode train --config config.json --head_type single_span
训练处理single_span问题的模型，head_type可以是single_span, tagged_spans, arithmetic, count, head_type
使用命令
python main.py --mode eval --config config.json
来测试模型在测试集上的评分
### Files
*   config.json: 配置文件
*   drop_datasets/: 数据文件夹
*   models/: 模型文件夹
*   stream_net/: 项目实现的代码文件夹
*   logs/: 训练日志文件夹

stream_net文件夹的目录结构如下：
├── data
│   ├── answer_generator            
│   │   ├── answer_generator.py                 : 答案生成器
│   │   ├── arithmetic_generator.py             : 算术答案生成器
│   │   ├── count_generator.py                  : 计数答案生成器
│   │   ├── evaluate_generator.py               : 评估模型时所用的答案生成器
│   │   ├── head_type_answer_generator.py       : 模型所用的输出头类型答案生成器
│   │   ├── single_span_answer_generator.py     : 单个span答案生成器
│   │   └── tagged_spans_answer_generator.py    : 标记span答案生成器
│   └── drop_datasets.py                        : 数据集
├── evaluate
|   ├── drop_eval.py                            : 测试集评估器
│   └── evaluator.py                            : 评估模型所用的评估器
├── model
│   ├── heads                                   : 模型的输出头
│   │   ├── arithmetic_head.py                  : 算术输出头
│   │   ├── count_head.py                       : 计数输出头
│   │   ├── head.py                             : 输出头
│   │   ├── head_classifier_head.py             : 输出头分类器
│   │   ├── single_span_head.py                 : 单个span输出头
│   │   └── tagged_span_head.py                 : 标记span输出头
│   └── reinforce_model.py                      : 模型
└── train
    └── trainer.py                              : 训练器