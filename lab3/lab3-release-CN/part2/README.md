# Lab 3 - part2

## 简介
在这个部分，您将调用Deepseek模型来帮助Pacman快速安全地完成游戏，总共占5分。您只需：

1. 按照[Deepseek API 官方文档](https://api-docs.deepseek.com/zh-cn/)创建API密钥，并将它填写在 **main.py** 中。
2. 完成 **LLM.py** 中标有 "TODO" 的部分（大语言模型提示词的编写）。它用来向大语言模型描述pacman游戏的规则。**提示词越完善、越全面，则模型的表现越好。**
3. 运行以下指令开始pacman游戏，以测试你的提示词的效果。
```bash
python main.py
```


## 环境安装
我们建议您创建一个新的conda环境以减小环境冲突的可能性。只需要运行下面的指令：
```bash
conda env create -f environment.yaml
conda activate pacman
```

如果出现错误，首先检查您已有的环境中是否已经存在名为“pacman”的环境，如果是则可以更改 **environment.yaml** 文件中的“name”条目解决此问题。

或者您可以选择手动安装：
```bash
conda create --name pacman python=3.11
pip install numpy pygame openai pycryptodome
conda activate pacman
```


## 评分
运行 `python autograder.py` 以获取你的成绩。我们使用带截断的线性评分方法。**90步**内完成游戏及格（**60/100 points**），**50步**内完成游戏满分**100/100 points**。请在评分前至少成功运行一次`python main.py`。

由于大语言模型的输出不可复现，我们使用日志进行评分，这意味着您运行的结果将记录在日志中用于评分。


## 注意
在这个部分中，为简单起见, **鬼将不会移动**。

我们推荐您使用Deepseek模型（v3和r1，可在 **main.py** 中更改），请注意：
1. v3的能力较弱（但完全足以完成这项任务），输出速度很快。
2. r1具有很强的推理能力，但需要更长的等待时间和更高的价格。

请自行抉择模型取舍。
也可以使用其他大语言模型。

当AI移动吃豆人到墙或鬼时，游戏将终止。此时，您应该修改提示词以避免类似的非法行为。

**不要试图自己改写日志，因为这很容易被我们发现。**
**不要在提示词中写除游戏规则和决策方法以外的任何信息。**


## 致谢
本部分由岳子轶、赵一鸣、刘旭设计实现。感谢2025年人工智能引论团队的所有教授和助教做出的宝贵贡献。
