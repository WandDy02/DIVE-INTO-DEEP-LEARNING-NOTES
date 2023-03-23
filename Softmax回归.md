# Softmax回归

softmax回归适用于分类问题，将向量映射成概率

## 1.*独热编码*（one-hot encoding）

独热编码是一个向量，它的分量和类别一样多。 类别对应的分量设置为1，其他所有分量设置为0。 在我们的例子中，标签$$y$$将是一个三维向量， 其中(1,0,0)对应于“猫”、(0,1,0)对应于“鸡”、(0,0,1)对应于“狗”：

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## 2.网络架构

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。 为了解决线性模型的分类问题，我们需要和输出一样多的*仿射函数*（affine function）。每个输出对应于它自己的仿射函数。例如在猫鸡狗分类问题中，有4个特征和3个可能是输出类别，我们将需要12个标量来表示权重（带下标的$$w$$）， 3个标量来表示偏置（带下标的$$b$$）。 下面我们为每个输入计算三个*未规范化的预测*（logit）：$$o_1$$、$$o_2$$和$$o_3$$。

$$\begin{split}\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}\end{split}$$

 与线性回归一样，softmax回归也是一个单层神经网络。 由于计算每个输出$$o_1$$、$$o_2$$和$$o_3$$取决于 所有输入$$x_1$$、$$x_2$$、$$x_3$$和$$x_4$$， 所以softmax回归的输出层也是全连接层。

![../_images/softmaxreg.svg](https://zh-v2.d2l.ai/_images/softmaxreg.svg)

## 3.softmax运算

在分类问题中，我们希望模型的输出$$\hat{y}_j$$可以视为属于类$$j$$的概率，然后选择具有最大概率的类别$$\operatorname*{argmax}_j y_j$$作为我们的预测。但是将线性层的输出直接视为概率时存在一些问题：**一方面，我们没有限制这些输出数字的总和为1。 另一方面，根据输入的不同，它们可以为负值。**

**要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1。 **此外，我们需要一个训练的目标函数，来激励模型精准地估计概率。 例如， 在分类器输出0.5的所有样本中，我们希望这些样本是刚好有一半实际上属于预测的类别。 这个属性叫做*校准*（calibration）。

 **softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持 可导的性质。 为了完成这一目标，我们首先对每个未规范化的预测求幂，这样可以确保输出非负。 为了确保最终输出的概率值总和为1，我们再让每个求幂后的结果除以它们的总和。如下式**：

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$

softmax运算不会改变未规范化的预测$$o$$ 之间的大小次序，只会确定分配给每个类别的概率。 因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别。

$$\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.$$

## 4.损失函数

与线性回归一样，需要一个损失函数来度量预测的效果。对于softmax回归，使用最大似然估计。

### 4.1对数似然

softmax函数给出了一个向量$$\hat{\mathbf{y}}$$， 我们可以将其视为“对给定任意输入$$\mathbf{x}$$的每个类的条件概率”。 例如，$$\hat{y}_1$$=$$P(y=\text{猫} \mid \mathbf{x})$$。 假设整个数据集$$\{\mathbf{X}, \mathbf{Y}\}$$具有$$n$$个样本， 其中索引$$i$$的样本由特征向量$$\mathbf{x}^{(i)}$$和独热标签向量$$\mathbf{y}^{(i)}$$组成。 我们可以将估计值与实际值进行比较:

$$P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).$$

根据最大似然估计，我们最大化$$P(\mathbf{Y} \mid \mathbf{X})$$,相当于最小化负对数似然：$$-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),$$

其中，对于任何标签$$y$$和模型预测$$\hat{\mathbf{y}}$$，损失函数为：

$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.$$

这样的损失函数通常被称为*交叉熵损失*（cross-entropy loss）。

### 4.2损失函数的化简及其导数

将$$\hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$代入$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.$$中得到：

$$\begin{split}\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}\end{split}$$

化简后对$$o_j$$的偏导为：

$$\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.$$



**对于softmax模型同样可以采用随机梯度下降算法进行模型训练。训练过程与线性回归类似。**

## 5.模型预测和评估

在训练softmax回归模型后，给出任何样本特征，我们可以预测每个输出类别的概率。 通常我们使用预测概率最高的类别作为输出类别。 如果预测与实际类别（标签）一致，则预测是正确的。 在接下来的实验中，我们将使用**准确度（accuracy）**来评估模型的性能。 精度等于正确预测数与预测总数之间的比率。用公式表示为：

$$\\accuracy = \frac{\\TP+TN}{\\TP+FP+TN+FN}$$



- True Positive (TP): 把正样本成功预测为正。
- True Negative (TN)：把负样本成功预测为负。
- False Positive (FP)：把负样本错误地预测为正。
- False Negative (FN)：把正样本错误的预测为负。