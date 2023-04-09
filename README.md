# 食物语“摆龙门”游戏AI代打

adapted from the [RLCard framework](https://github.com/datamllab/rlcard)

基于斗地主代码开发。

训练两个模型，分别用于先手和后手

卡牌编码：一共22张。颜色：灰G、蓝B、黄Y、彩C、红R，花色：梅M、兰L、竹Z、菊J。如“彩梅”为“CM”。雉羹与鹄羹分别为DW, XW。

动作编码：卡牌编码 + 区域（0-私有区域，1-公有区域）。比如，"CM1"表示将彩梅打入公开区域。
动作空间维度：22x2 = 44

状态空间维度：系统公有牌(22) + 己方公有牌 (22) + 己方私有牌 (22) + 对方公有牌 (22) + 己方手牌 (22) + 己方剩余公有空间(4) + 己方剩余私有空间(4) + 对方剩余公有空间(4) + 对方剩余私有空间(4) + 目前己方各花色数量 (6x4+3) + 目前己方各颜色数量 (5x5) + 目前对方各花色数量 (6x4+3) + 目前对方各颜色数量 (5x5) = 230

奖励函数：己方最终得分 - 对方最终得分


## Agents

### AI Agents

DMC agent: Deep Monte Carlo. 

Version 1.0: (--ai_agent dmc --model_path results/dmc_swy/bailongmen/model.tar)
- 对战random agent的情况（基于1000场的模拟对局）：先手胜率91.1%，平均每局净胜5.68分；后手时胜率91.3%，平均每局净胜5.77分。
- 对战rule-based agent的情况（基于1000场的模拟对局）：先手胜率62.2%，平均每局净胜1.84分；后手时胜率71.2%，平均每局净胜2.57分。

### Baseline agents

**Random agent**


**Rule-based agent**

Rule-based agent 对战 Random agent:基于1000场的模拟对局，Rule-based agent先手时胜率87.0%，平均每局净胜4.54分；后手时胜率88.3%，平均每局净胜4.80分。

