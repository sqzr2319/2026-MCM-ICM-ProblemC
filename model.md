## **一、问题简化与符号定义**

### **1.1 已知数据**
对于每季 $s=1,2,\dots,34$：
- 参赛者集合：$i = 1, \dots, N_s$
- 周次：$t = 1, \dots, T_{s,i}$（直到该选手被淘汰）
- $J_{s,i,t}$：第 $s$ 季第 $t$ 周专业评审给选手 $i$ 的平均打分（标准化到 0-1 之间）
- $F_{s,i}$：选手 $i$ 在赛季 $s$ 时的社交媒体粉丝数（取对数后标准化）
- $L_{s,i,t}$：第 $s$ 季选手 $i$ 是否在恰好在第 $t$ 周被淘汰（1 表示被淘汰，0 表示未被淘汰）

### **1.2 未知目标**
- $P_{s,i,t}$：选手 $i$ 在第 $t$ 周获得的**大众投票倾向分数**
- $V_{s,i,t}$：选手 $i$ 在第 $t$ 周获得的**大众投票百分比**
- 约束：$\sum_{i \in \text{active}_t} V_{s,i,t} = 1$（每周大众投票百分比总和为1）

---

## **二、核心建模思想**

### **2.1 基本假设**
1. 每周淘汰结果由 **（评委分，大众投票）** 按某种**已知规则**（排名法或百分比法）组合决定
2. 大众投票受两个因素影响：
   - **当前表现**：评委打分 $J_{s,i,t}$ 反映的客观质量
   - **固有流行度**：粉丝数 $F_{s,i}$ 反映的人气基础

### **2.2 数学模型框架**
我们采用**贝叶斯潜在变量模型**，结构如下：

$$
\begin{aligned}
\text{大众投票倾向} &: \quad P_{s,i,t} = \gamma_{s,i} + \delta_1 J_{s,i,t} + \delta_2 F_{s,i} \\
 	{大众投票百分比} &: \quad \tilde{P}_{s,i,t} = P_{s,i,t} - \min_{j \in \text{active}_t} P_{s,j,t} \\
 & \quad V_{s,i,t} = \frac{\tilde{P}_{s,i,t}}{\sum_{j \in \text{active}_t} \tilde{P}_{s,j,t}} \quad (\text{若分母为 }0, \text{则取均匀分配})
\end{aligned}
$$

其中：
- $\gamma_{s,i}$：选手个体随机效应
- $\delta_1$：评委打分对实力/投票的影响系数
- $\delta_2$：粉丝数对实力/投票的影响系数

---

## **三、利用淘汰结果的约束条件**

### **3.1 百分比法**

对于每周 $t$，对于使用**百分比法**的季节（Season 3-27），记组合分为 $C_{s,i,t}$：

$$
C_{s,i,t} = \underbrace{\frac{J_{s,i,t}}{\sum_j J_{s,j,t}}}_{\text{评委百分比}} + \underbrace{V_{s,i,t}}_{\text{大众投票百分比}}
$$

使用 **softmax** 将硬决策转化为软概率：

$$
\mathbb{P}_{s,i,t} = \frac{\exp(-\tau \cdot C_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(-\tau \cdot C_{s,j,t})}
$$

其中 $\tau \to \infty$ 时恢复硬决策。实际中取 $\tau=10$ 足够大。

淘汰者：若有 $k$ 名选手在第 $s$ 季第 $t$ 周被淘汰，则取 $\mathbb{P}_{s,i,t}$ 最高的 $k$ 名选手。

优化目标：$\min -\sum_{s,i,t} L_{s,i,t} \log(\mathbb{P}_{s,i,t})$。

### **3.2 排名法**

对于每周 $t$，对于使用**排名法**的季节（Season 1-2, 28-34），记组合分为 $R_{s,i,t}$：

定义评委分数排名为：
$$
\text{rank}_J(i) \approx 1 + \sum_{j \neq i} \sigma\left(\beta \cdot\frac{J_{s,j,t} - J_{s,i,t}}{\sum_{k \in \text{active}_t} J_{s,k,t}}\right)
$$

其中 $\sigma(x)$ 是sigmoid函数，$\sigma(x) \approx 1$ 如果 $x > 0$。

参数 $\beta$ 控制评委分差对最终淘汰决策的影响，选取 $\beta=1$。

类似地，大众投票排名为：
$$
\text{rank}_V(i) \approx 1 + \sum_{j \neq i} \sigma(\beta (V_{s,j,t} - V_{s,i,t}))
$$

组合排名：
$$
\tilde{R}_{s,i,t} = \text{rank}_J(i) + \text{rank}_V(i)
$$

使用 **softmax** 将硬决策转化为软概率：

$$
\mathbb{P}_{s,i,t} = \frac{\exp(\tau \cdot \tilde{R}_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(\tau \cdot \tilde{R}_{s,j,t})}
$$

优化目标： $\min -\sum_{s,i,t} L_{s,i,t} \log(\mathbb{P}_{s,i,t})$。

### **3.3 二次选择机制**

对于Season 28-34，存在二次选择机制，分四种情况：

1. 决赛周：无二次选择，直接使用上述排名法概率 $\mathbb{P}_{s,i,t}$。
2. 非决赛周且淘汰1人：先使用排名法概率 $\mathbb{P}_{s,i,t}$ 选出“Bottom2”，再从中选出淘汰者。

$$
\mathbb{P}^{\text{final}}_{s,i,t} = \sum_{j \neq i} \mathbb{P}(i,j \in \text{Bottom2}) \cdot \mathbb{P}(\text{评委淘汰} i | i,j \in \text{Bottom2})
$$

$$
\mathbb{P}(i,j \in \text{Bottom2}) = \mathbb{P}_{s,i,t} \cdot \frac{\mathbb{P}_{s,j,t}}{1-\mathbb{P}_{s,i,t}} + \mathbb{P}_{s,j,t} \cdot \frac{\mathbb{P}_{s,i,t}}{1-\mathbb{P}_{s,j,t}}
$$

$$
\mathbb{P}(\text{评委淘汰} i | i,j \in \text{Bottom2}) = \sigma\left(\beta \cdot \frac{J_{j,t} - J_{i,t}}{\sum_{k \in \text{active}_t} J_{s,k,t}}\right)
$$

优化目标： $\min -\sum_{s,i,t} L_{s,i,t} \log(\mathbb{P}^{\text{final}}_{s,i,t})$。

3. 非决赛周且淘汰2人：直接使用排名法概率 $\mathbb{P}_{s,i,t}$ 选出淘汰者。
4. 非决赛周且淘汰0人：不计入损失函数。

---

## **四、一致性指标**

将所有事件的“预测正确的淘汰者数”累加后除以“总淘汰者数”，即
$$
	{Accuracy}_{\text{weighted}} = \frac{\sum_{\text{events}} \#\text{预测正确的淘汰者}}{\sum_{\text{events}} k}.
$$

- 决赛加分规则：定义每季的“决赛”为该季最后一次发生淘汰的周。若该周的淘汰者全部预测正确，则视作也预测对了“冠军”，在分子额外加 1，并在分母额外加该季的决赛场次 1。故总体目标数从“仅淘汰者总数”扩展为“淘汰者总数 + 决赛场次数”，即总人数。

