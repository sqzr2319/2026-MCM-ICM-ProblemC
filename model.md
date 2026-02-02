## **一、问题简化与符号定义**

### **1.1 已知数据**
对于每季 $s=1,2,\dots,34$：
- 参赛者集合：$i = 1, \dots, N_s$
- 周次：$t = 1, \dots, T_{s,i}$（直到该选手被淘汰）
- $J_{s,i,t}$：第 $s$ 季第 $t$ 周专业评审给选手 $i$ 的平均打分
- $F_{s,i}$：选手 $i$ 在赛季 $s$ 时的 Google Trends 热度分数
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
   - **固有流行度**：Google Trends 热度分数 $F_{s,i}$ 反映的人气基础

### **2.2 数学模型框架**
我们采用**贝叶斯潜在变量模型**，结构如下：

$$
\begin{aligned}
\text{大众投票倾向} &: \quad P_{s,i,t} = \gamma_{s,i} + \delta_1 \frac{J_{s,i,t}}{\sum_{j \in \text{active}_t} J_{s,j,t}} + \delta_2 \frac{F_{s,i}}{\sum_{j \in \text{active}_t} F_{s,j}} \\
 	{大众投票百分比} &: \quad V_{s,i,t} = \frac{\exp(P_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(P_{s,j,t})} 
\end{aligned}
$$

其中：
- $\gamma_{s,i}$：选手个体随机效应
- $\delta_1$：评委打分对实力/投票的影响系数
- $\delta_2$：Google Trends 热度分数对实力/投票的影响系数

我们假设，$\delta_1,\delta_2$ 在采用相同规则的所有赛季中共享，但在排名法和百分比法的赛季中不共享，即 $\delta_1^{\text{rank}}, \delta_2^{\text{rank}}$ 与 $\delta_1^{\text{percent}}, \delta_2^{\text{percent}}$ 分别独立估计。

---

## **三、利用淘汰结果的约束条件**

### **3.1 百分比法**

对于每周 $t$，对于使用**百分比法**的季节（Season 3-27），记组合分为 $C_{s,i,t}$：

$$
C_{s,i,t} = \underbrace{\frac{J_{s,i,t}}{\sum_{j \in \text{active}_t} J_{s,j,t}}}_{\text{评委百分比}} + \underbrace{V_{s,i,t}}_{\text{大众投票百分比}}
$$

使用 **softmax** 将硬决策转化为软概率：

$$
\mathbb{P}_{s,i,t} = \frac{\exp(-\tau \cdot C_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(-\tau \cdot C_{s,j,t})}
$$

其中 $\tau \to \infty$ 时恢复硬决策。实际中取 $\tau=60$ 足够大。

淘汰者：若有 $k$ 名选手在第 $s$ 季第 $t$ 周被淘汰，则取 $\mathbb{P}_{s,i,t}$ 最高的 $k$ 名选手。

优化目标：$\min -\sum_{s,i,t} L_{s,i,t} \log(\mathbb{P}_{s,i,t})$。

### **3.2 排名法**

对于每周 $t$，对于使用**排名法**的季节（Season 1-2, 28-34），记组合分为 $R_{s,i,t}$：

定义评委分数排名为：
$$
\text{rank}_J(i) \approx 1 + \sum_{j \neq i} \sigma\left(\frac{J_{s,j,t} - J_{s,i,t}}{\sum_{k \in \text{active}_t} J_{s,k,t}}\right)
$$

其中 $\sigma(x)$ 是sigmoid函数，$\sigma(x) \approx 1$ 如果 $x > 0$。

类似地，大众投票排名为：
$$
\text{rank}_V(i) \approx 1 + \sum_{j \neq i} \sigma(V_{s,j,t} - V_{s,i,t})
$$

组合排名：
$$
\tilde{R}_{s,i,t} = \text{rank}_J(i) + \text{rank}_V(i)
$$

使用 **softmax** 将硬决策转化为软概率：

$$
\mathbb{P}_{s,i,t} = \frac{\exp(\tau \cdot \tilde{R}_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(\tau \cdot \tilde{R}_{s,j,t})}
$$

淘汰者：若有 $k$ 名选手在第 $s$ 季第 $t$ 周被淘汰，则取 $\mathbb{P}_{s,i,t}$ 最高的 $k$ 名选手。

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
\mathbb{P}(\text{评委淘汰} i | i,j \in \text{Bottom2}) = \sigma\left( \frac{J_{j,t} - J_{i,t}}{\sum_{k \in \text{active}_t} J_{s,k,t}}\right)
$$

淘汰者：取 $\mathbb{P}^{\text{final}}_{s,i,t}$ 最高的选手。

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

## **五、确定性指标**

使用训练好的模型参数，计算每个选手在每周的 $V_{s,i,t}$。我们采用 $V_{s,i,t}$ 的标准差作为不确定性指标，利用 **Delta 方法** 从参数向量出发计算标准差。

设参数向量为：
$$
\theta = \begin{bmatrix}
\gamma_{1,1} \\ \vdots \\ \gamma_{S,N_S} \\
\delta_1^{\text{rank}} \\
\delta_2^{\text{rank}} \\
\delta_1^{\text{percent}} \\
\delta_2^{\text{percent}}
\end{bmatrix}
\in \mathbb{R}^{M}
$$
其中 $M = \sum_{s=1}^{34} N_s + 4$。
$$
P_{s,i,t} = \gamma_{s,i} + \delta_1 \frac{J_{s,i,t}}{\sum_{j \in \text{active}_t} J_{s,j,t}} + \delta_2\frac{F_{s,i}}{\sum_{j \in \text{active}_t} F_{s,j}}
$$

梯度为：
$$
\nabla_\theta P_{s,i,t} = \begin{cases}
1 & \text{对应 } \gamma_{s,i} \\
\frac{J_{s,i,t}}{\sum_{j \in \text{active}_t} J_{s,j,t}} & \text{对应 } \delta_1 \\
\frac{F_{s,i}}{\sum_{j \in \text{active}_t} F_{s,j}} & \text{对应 } \delta_2
\end{cases}
$$

$$
V_{s,i,t} = \frac{\exp(P_{s,i,t})}{\sum_{j \in \text{active}_t} \exp(P_{s,j,t})} = \frac{\tilde{P}_{s,i,t}}{S_t}
$$

使用商法则：
$$
\frac{\partial V_{s,i,t}}{\partial \theta} = \frac{1}{S_t} \frac{\partial \tilde{P}_{s,i,t}}{\partial \theta} - \frac{\tilde{P}_{s,i,t}}{S_t^2} \frac{\partial S_t}{\partial \theta}
$$

其中：
$$
\frac{\partial S_t}{\partial \theta} = \sum_{j \in A_t} \frac{\partial \tilde{P}_{s,j,t}}{\partial \theta}
$$

接下来我们需要参数协方差矩阵 $\Sigma_\theta$：

$$
\Sigma_\theta = \left[-\nabla^2 \mathcal{L}(\hat{\theta})+\lambda I\right]^{-1}
$$

其中 $\nabla^2 \mathcal{L}(\hat{\theta})$ 是**对数似然函数在最优参数 $\hat{\theta}$ 处的Hessian矩阵**，$\lambda$ 是一个小的正数，用于数值稳定性。

对于你的模型，对数似然为：
$$
\mathcal{L}(\theta) = \sum_{s,i,t} L_{s,i,t} \log \mathbb{P}_{s,i,t}(\theta)
$$

其中 $\mathbb{P}_{s,i,t}(\theta)$ 根据季节类型不同（百分比法/排名法/二次选择）而不同。

**Hessian计算**：
$$
H = \nabla^2 \mathcal{L}(\theta) = \sum_{s,i,t} L_{s,i,t} \left[ \frac{\nabla^2 \mathbb{P}_{s,i,t}}{\mathbb{P}_{s,i,t}} - \frac{\nabla \mathbb{P}_{s,i,t} \otimes \nabla \mathbb{P}_{s,i,t}}{\mathbb{P}_{s,i,t}^2} \right]
$$

由于需要二阶导数，计算复杂。实际中常用经验Fisher信息矩阵近似：

$$
\hat{H} \approx -\sum_{s,i,t} \nabla \log \mathbb{P}_{s,i,t} \cdot \nabla \log \mathbb{P}_{s,i,t}^T
$$

这是Hessian的负定近似，更易计算。

然后使用 Delta 方法计算方差。$V_{s,i,t}$ 的方差为：
$$
\text{Var}(V_{s,i,t}) = \mathbf{g}_{V}^T \Sigma_\theta \mathbf{g}_{V}
$$
其中 $\mathbf{g}_{V} = \frac{\partial V_{s,i,t}}{\partial \theta} \in \mathbb{R}^M$ 是已推导的梯度向量。

标准差：
$$
\text{SD}(V_{s,i,t}) = \sqrt{\mathbf{g}_{V}^T \Sigma_\theta \mathbf{g}_{V}}
$$

## **六、检验**

使用模型预测的 $V_{s,i,t}$，对每个赛季每周分别应用百分比法和排名法，计算预测淘汰者。计算两种规则下预测结果的一致率。

为了比较哪种规则比另一种更倾向于粉丝投票，我们再采用一种新规则：只根据 $V_{s,i,t}$ 排名淘汰选手。计算这种“纯大众投票”规则与原规则（百分比法/排名法）预测结果的一致率，一致率更高的规则更倾向于粉丝投票。

接着我们检查特定"争议"名人（即评委与粉丝意见存在分歧的情况）：

- 第2季 - Jerry Rice，尽管有5周获得最低评委评分，但仍获得亚军。
- 第4季 - Billy Ray Cyrus，尽管有6周获得最后一名评委评分，但仍获得第5名。
- 第11季 - Bristol Palin，尽管12次获得最低评委评分，但仍获得第3名。
- 第27季 - Bobby Bones，尽管评委评分持续较低，但仍赢得冠军。

对这些选手，使用模型预测的 $V_{s,i,t}$，分别应用百分比法、排名法、带有二次选择的排名法，检查是否会导致相同的结果。计算三种方法中哪一种引起的争议最少（即与实际的争议结果一致率最低）。

## **七、名人特征的影响**

根据模型，$\gamma_{s,i}$ 就是和技术表现以及人气解耦之后的投票偏好，是名人特征对粉丝投票数影响的良好表征。我们使用训练好的模型参数 $\gamma_{s,i}$，对年龄绘制散点图，做回归分析；对行业、家乡国家地区、家乡州（美国选手）、专业舞者搭档绘制箱线图，做ANOVA分析，检验这些名人特征对 $\gamma_{s,i}$ 是否有显著影响。

类似地，我们使用 $J_{s,i} = \frac{1}{T_{s,i}} \sum_{t=1}^{T_{s,i}} J_{s,i,t}$ 作为选手 $i$ 在赛季 $s$ 的平均评审评分，对年龄、行业、家乡国家地区、家乡州（美国选手）、专业舞者搭档做同样的散点图、箱线图，进行同样的回归分析和ANOVA分析，检验这些名人特征对评审评分是否有显著影响。

## **八、敏感性分析**

我们对整个模型中唯一的固定参数 $\tau$ 进行敏感性分析。

选取不同的 $\tau$ 进行训练，绘制 $\tau$ 与 ${Accuracy}_{\text{weighted}}$、$V_{s,i,t}$ 标准差均值之间的关系曲线，观察模型性能和不确定性指标对 $\tau$ 的敏感性。