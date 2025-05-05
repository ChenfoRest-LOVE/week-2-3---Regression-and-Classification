# 吴恩达机器学习笔记（Week 2 & 3）

## Week 2：多元线性回归与梯度下降

### 1. 多元线性回归

#### 1.1 单变量线性回归回顾

- 模型：  
  $$
  f_w(x) = w_0 + w_1x
  $$
- 代价函数：  
  $$
  J(w_0, w_1) = \frac{1}{2m} \sum_{i=1}^{m} (f_w(x^{(i)}) - y^{(i)})^2
  $$

#### 1.2 多元线性回归扩展

- 模型：  
  $$
  f_{\mathbf{w}}(\mathbf{x}) = w_0 + w_1x_1 + \cdots + w_nx_n = \mathbf{w}^T \mathbf{x}
  $$
  其中：
  $$
  \mathbf{x} = [1, x_1, x_2, ..., x_n]^T,\quad \mathbf{w} = [w_0, w_1, ..., w_n]^T
  $$

#### 1.3 多元线性回归代价函数

- 非向量化：  
  $$
  J(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)})^2
  $$
- 向量化：  
  $$
  J(\mathbf{w}) = \frac{1}{2m} (\mathbf{Xw} - \mathbf{y})^T (\mathbf{Xw} - \mathbf{y})
  $$

---

### 2. 梯度下降

#### 2.1 更新公式

- 向量化表示：  
  $$
  \mathbf{w} := \mathbf{w} - \alpha \cdot \nabla J(\mathbf{w})
  $$
- 梯度公式（向量化）：  
  $$
  \nabla J(\mathbf{w}) = \frac{1}{m} \mathbf{X}^T(\mathbf{Xw} - \mathbf{y})
  $$

#### 2.2 特征缩放（Feature Scaling）

- 均值归一化：  
  $$
  x_j := \frac{x_j - \mu_j}{s_j}
  $$
- Z-score 标准化：  
  $$
  x_j := \frac{x_j - \mu_j}{\sigma_j}
  $$

#### 2.3 学习率选择

- 学习率过小：收敛慢  
- 学习率过大：可能发散  
- 通常尝试值：`0.001` 到 `1` 之间的指数级增量

---

### 3. 多项式回归

- 扩展输入变量：
  $$
  x_2 = x_1^2,\quad x_3 = x_1^3,\quad x_4 = x_1 \cdot x_2
  $$
- 有助于拟合非线性数据  
- 特征缩放尤其重要

---

### 4. 正规方程（Normal Equation）

- 无需迭代，解析求解最优参数：
  $$
  \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
  $$
- 缺点：计算逆矩阵在特征多时成本高（$O(n^3)$）

---

## Week 3：逻辑回归与正则化

### 1. 逻辑回归（Logistic Regression）

#### 1.1 模型

- 假设函数：
  $$
  h_{\mathbf{w}}(\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}
  $$
- 输出值 $h_{\mathbf{w}}(\mathbf{x})$ 表示属于类别 1 的概率

#### 1.2 代价函数（Log Loss）

- 对数损失函数：
  $$
  J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^m \left[-y^{(i)} \log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) - (1 - y^{(i)}) \log(1 - h_{\mathbf{w}}(\mathbf{x}^{(i)}))\right]
  $$

#### 1.3 梯度更新

- 梯度下降更新：
  $$
  w_j := w_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}
  $$

---

### 2. 多类分类（One-vs-All）

- 对于 $k$ 类问题：
  - 训练 $k$ 个逻辑回归模型，分别预测 “是否属于该类”
  - 预测时选择最大概率对应的类

---

### 3. 正则化（Regularization）

#### 3.1 L2 正则化（Ridge）

- 加入正则项后的代价函数（以线性回归为例）：
  $$
  J(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2
  $$
- 注：$w_0$ 不正则化

---

#### 3.2 正则化目的

- 减少模型过拟合  
- 保留关键特征，限制权重增长  
- 更强泛化能力

---

> 本笔记整理自吴恩达机器学习课程第二周与第三周内容，适合用于 Markdown 平台复习与复用。
