# 食物语“摆龙门”游戏AI代打

## 免责声明

本程序仅用于学习与交流，任何超出此范围的使用所导致的后果将全部由使用者本人承担。选择使用本程序，表明您已无条件同意接受以上条款。为避免本程序的恶意使用破坏游戏平衡，本站将不提供预训练模型权重等支持文件。

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
- 对战random agent的情况（基于5000场的模拟对局）：
    - 先手：胜91.1%，负5.6%，平3.3%。平均每局净胜5.75分。
    - 后手：胜92.1%，负4.8%，平3.1%。平均每局净胜5.92分。
- 对战rule-based agent的情况（基于5000场的模拟对局）：
    - 先手：胜63.3%，负28.2%，平8.5%。平均每局净胜1.882分。
    - 后手：胜68.1%，负22.8%，平9.9%。平均每局净胜2.465分。
- 对战游戏内官方AI（“鱼香肉丝”）的情况：手动10场对局中，我方8胜、1负、1平。

### Baseline agents

**Random agent**


**Rule-based agent**

- Rule-based agent 对战 Random agent的情况（基于1000场的模拟对局）：
    - 先手：胜81.5%，负12.6%，平5.9%。平均每局净胜4.56分。
    - 后手：胜84.7%，负11.2%，平4.1%。平均每局净胜4.94分。

