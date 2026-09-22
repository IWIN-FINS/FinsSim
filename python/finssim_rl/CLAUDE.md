## 项目概述

本项目是单智能体强化学习进行水下潜器研究，使用的Unity ML-Agents接口。主要是从Unity搭建强化学习环境，然后导出binary build文件接入python的stable baselines 3算法进行学习。

最开始的脚本为scripts/single_scripts/下面的单文件定义模式，后面重建为CLI应用，支持不同模型的train eval等。

### 文件夹结构

scripts/ 运行的CLI脚本

src/underwater_rl/ python文件，采用src架构
    models/ 下面定义一些强化学习模型
    training/ 保存一些config

checkpoints/ 保存RL模型

config_unity/ 已经弃用。主要是调用原生Unity接口进行训练

docs/ 文档和使用说明

tests/ 测试脚本

## 个性化

使用英文思考，回答推荐使用中文

对于新写的单个CLI脚本，在脚本顶端写上文件注释和用法说明。没有明确要求的时候不添加md文档。