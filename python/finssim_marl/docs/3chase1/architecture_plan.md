
## 整体思路

采用CTDE的想法（可能因为CTCE在实际运行的时候需要通信和大量的计算，不太方便？）
当然后面可以针对CTDE进行更多机器人的仿真扩展验证实用性，或者拿掉几个机器人验证鲁棒性，这是后话.

### 分层RL

#### Plan A（实机比较可行吗？？？）
Obs = Fish relative pos + dir (以上是用视觉摄像头单一跟踪) + 机器人靠近给出0/1 bit warning或者四个方向的0/1 bit warning以便可以确定紧急避障方向。

但是感觉这样避障的效果也不好，可能干脆不做Safe MARL了，或者就做简单的reward shaping实现安全

#### Plan B （仿真略微简单，实机也有一定的可能性）

就是实机的时候接一个外部的全局定位系统，给出上面声呐买不起等缺失的辅助信息。

Obs = Self linear velocity + rotation velocity + Fish relative pos + ont-hot(队友类型:herder/netter) + 队友的relative pos

关键是有一些问题：就是到底要不要加上Fish的direction/velocity？这个是直接传感器给/差分给，还是干脆算法内部使用LSTM？

需要的信息多了是不是要Transformer？

#### Plan C （仿真更加简单，但是需要的信息更多）

可以给出自体的position，他们的全局position，但是也有一个问题：给了position是不是就对position数值敏感了？比如开始的时候是(0,0,0)还是(100,100,100)？算法就缺失了泛化性了！

#### Plan D （直接模仿现有的一篇论文）

大概意思是：自体感知需要包括自己和target的信息，不需要考虑队友的信息。但是每台机器人有一个独有的robot ID。训练过程中使用全局的Critic先训练好，再CTDE执行，执行的时候就可以实现不碰撞。但是他们论文的三个机器人是一样的，我们的三个机器人是分成Herder和Netter的，具体问题描述参见 @CLAUDE.md ，角色不太一样。


最后讨论出的结果是：我个人还是觉得视觉方案很难，可能只是加分点吧，或者新的一篇论文的内容。所以先默认能够获取全局定位先把MARL做一版出来！而且关于安全性，也暂时只是使用reward shaping保证的

### 其它备选方案

#### Extend A

直接端到端学习尝试一下，或者运用经典的MAPPO，LSTM

#### Extend B

slot-structured MLP

slot-structured MLP 指的是另一层意思：

先按语义把 observation 拆槽
比如：
self 一槽
fish 一槽
teammate_1 一槽
teammate_2 一槽
然后每个槽各自编码
对两个 teammate 槽通常还会共享同一个 encoder
最后再把这些 embedding 拼起来做决策
它大概是这种感觉：

self_feat = SelfEncoder(self_state)
fish_feat = FishEncoder(fish_state)
mate1_feat = TeammateEncoder(mate1_state)
mate2_feat = TeammateEncoder(mate2_state)   # 共享权重
joint_feat = concat(self_feat, fish_feat, mate1_feat, mate2_feat, role_onehot)
action = PolicyHead(joint_feat)
它比平铺 MLP 多出来的价值是：

对“两个队友槽位”有更强的结构归纳偏置
更容易共享“怎么看一个 teammate”的规则
后面如果槽位数变化，或者要做 permutation/symmetry 处理，更容易扩展
但它不是必须的。对你现在这个任务，我会这么判断：

如果现在就是 2 个固定队友槽位，语义固定，维度也不大，
那当前 flat MLP / shared-feature MLP 是完全合理的起点。
真正更重要的是：
观测语义先设计对
坐标系先统一对
高低层接口先拆开
特征归一化先做好
换句话说：

shared_feature MLP 不是“不行”
它只是“没有显式利用 teammate slot 结构”
这会影响泛化上限，但未必影响你做出第一个可用版本
所以我对这件事的建议是：

v1 可以先继续用当前这种 shared-feature MLP
只改 observation layout 和 high/low-level interface
等你把 controller 通道理顺了，再决定要不要上 slot-structured encoder
一句话总结这两个问题：

自体 position：不要强塞回 actor obs，走单独的 controller_obs 通道。
主干网络：当前 shared-feature MLP 可以先用，slot-structured MLP 只是更有结构偏置，不是 v1 的前置条件。

## 一些未来需要做的

消融实验

## 未来可以做的

- 尝试Trasnformer？Encoder? Slot-mlp？
- 变数量的Netter？
- 为了方便实机，将observation弱化？
- safe marl


## 技术细节

如果需要比较复杂的传递Unity Env里面的其它信息，可以在Unity脚本里面搭载一个虚拟的环境Agent，由它来传递所需的额外信息和action