## 项目概述

本项目是对水下潜器多智能体强化学习进行研究。使用的Unity ML-Agents接口。主要是从Unity搭建强化学习环境，然后导出binary build文件接入别人开源实现的python的[cleanmarl](https://github.com/AmineAndam04/cleanmarl)算法进行多智能体强化学习。

目前实现的unity多智能体强化学习的环境有很多。预计需要研究：
- **3chase1 (herding)** 三追一问题，定义为一个Herder，两个Chaser/Netter去追一个逃方Prey。Herder如同渔民一个人赶网，两个Chaser手拉手拉网，将鱼进行捕获。
- **3chase1 (chasing)** 三追一，但是是三个人手拉手拉网，在一个立体水下空间对Prey进行追赶。
- **2chase1** 二追一问题，两个chaser拉网追一个Prey

目前初步的设想是让Herder/Chaser/Netter的最大速度比Prey慢，但是瞬时加速度比Prey快，能利用合作关系巧妙追上。

### 文件夹结构

src/finssim_marl/
  single_scripts/ 借助原先的 cleanmarl 定义的单脚本运行MAPPO相关算法
  algorithms/ 下面的mappo_multihead和mappo_multihead_position_controller是目前维护并开启训练了的

scripts/ 统一的train, eval脚本

docs/ 文档和使用说明

目前正在做第一个3chase1 (herding), 写了`cleanmarl/mappo_3chase1_unity.py`这个文件。

## 具体研究问题

### **3chase1 (herding)**
三追一问题，定义为一个Herder和两个Netter去追一个逃方Prey。**Herder和Netter统称为Chaser**（也就是同属于Chaser Team）。Herder如同渔民一个人赶网，两个Chaser手拉手拉网，将鱼进行捕获。

目前Chaser的动作空间为[-1,1]的八个实数值，直接对应Unity层的八个推进器的推力。

Prey的动作空间暂定为[-1,1]的三个实数值，在Unity中的定义为：

```csharp
float moveStepX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
float moveStepY = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
float moveStepZ = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
rb.linearVelocity = new Vector3(moveStepX, moveStepY, moveStepZ) * moveSpeed;
```

#### 一些Unity里面环境返回的常见数据格式：

这边的`obs_dict`为：
```
{'Netter?team=2?agent_id=0': array([ 1.        ,  1.9797949 , -0.02332479,  9.516841  , -0.11835279,
-0.01482341, -0.01912729,  7.5       , -0.02099239,  5.        ,
1.970356  , -0.0644888 ,  0.5098164 ], dtype=float32), 'Prey?team=3?agent_id=1': array([ 7.5000000e+00, -2.0992389e-02,  5.0000000e+00,  0.0000000e+00,
-9.2858024e-02,  0.0000000e+00,  1.9703560e+00, -6.4488798e-02,
5.0981641e-01,  1.9797949e+00, -2.3324788e-02,  9.5168409e+00,
1.4990288e+01, -1.1143848e-02,  5.0045681e+00], dtype=float32), 'Netter?team=1?agent_id=2': array([ 1.        ,  1.970356  , -0.0644888 ,  0.5098164 , -0.1546133 ,
-0.16090038,  0.11276507,  7.5       , -0.02099239,  5.        ,
1.9797949 , -0.02332479,  9.516841  ], dtype=float32), 'Herder?team=0?agent_id=3': array([ 0.0000000e+00,  1.4990288e+01, -1.1143848e-02,  5.0045681e+00,
2.0028202e-02,  2.7981058e-02, -4.7182608e-03,  7.5000000e+00,
-2.0992389e-02,  5.0000000e+00,  0.0000000e+00,  0.0000000e+00,
0.0000000e+00], dtype=float32)}
```
`reward_dict`的内容为：
```
{'Netter?team=2?agent_id=0': np.float32(-0.7140518), 'Prey?team=3?agent_id=1': np.float32(0.0), 'Netter?team=1?agent_id=2': np.float32(-0.71310765), 'Herder?team=0?agent_id=3': np.float32(0.01671427)}
```

## 个性化

使用英文思考，回答推荐使用中文

遵循良好的设计模式规范

本项目有python的虚拟环境！在`.venv`下面！需要使用python虚拟环境运行脚本！