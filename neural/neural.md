# 神经网络学习笔记

## 背景

想想人的大脑是个很神奇的东西，在不断的学习中会形成自己的逻辑判断能力可以找到同一类事物的共同特征并以此来进行简单和基础的判断或者分类。如果能有一种模型或者算法能模拟大脑那是不是同样可以通过学习来进行更复杂活着抽象的判断或者分类了。神经网络学习算法可以算是一个这方面的尝试(但是它的目的并不是取模仿人的大脑)。记得在[感知机](https://github.com/MemoryCrash/MachineLearningPractice/blob/master/perceptron/perceptron.md)这个学习笔记中，提到了，对于感知机这样的基础算法，有两个很巧的拓展方式去处理复杂的判断，一个是通过将高维数据映射到低维然后通过核函数去处理这个形成了SVM算法，一个是增加层数去实现复杂判断的处理这样形成了神经网络，它属于监督学习。注意这个笔记只是针对神经网络，并不是深度学习。

## 神经元模型
<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuron.png" width=50% height=50%/>

真实的神经元的样子比较接近上图。神经元接受传入的信号，当信号超过阀值神经元被激活。经过简单化处理我们可以将神经元模型简化为下图。其中w表示输入信号的权值，b代表阀值，最后利用激活函数得出输出y。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/perceptron.png"/>

简单讲我们希望通过0来表示神经元是抑制，使用1表示神经元是兴奋状态。这样看来似乎使用阶跃函数比较好。但是阶跃函数不连续，不利于后续的计算使用。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralStepFuction.jpg" width=30% height=30%/>

所以后面找到了sigmoid函数来作为激活函数，可以看出它在0附近快速的变化为0或者1并且也具有连续可导的特点。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmoid.png" width=30% height=30%/>

我们可以将神经元模型通过数学公式来表示：  

![pi](http://latex.codecogs.com/png.latex?\sigma&space;(z)=\frac{1}{1&plus;e^{-z}})     

这个公式其实就是sigmoid公式，但是光有这个还不够，我们还需要将公式中的z表示出来：    

![pi](http://latex.codecogs.com/png.latex?\sum_{j}w_{j}x_{j}&plus;b)    

## 神经网络

我们建立的一个神经元模型。这个对这个模型比较好的直接应用就是[感知机](https://github.com/MemoryCrash/MachineLearningPractice/blob/master/perceptron/perceptron.md)了但是通过感知机的笔记可以了解到，感知机能做的事情是非常有限的。通过组合多个神经元去构建一个多层的神经网络我们可以处理更加复杂的分类或者判断问题。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralNetwork.png" width=40% height=40%/>

这里我们构建了一个具有三层结构的神经网络，从左到右分别是输入层、隐藏层、输出层。从最开始input进入神经元的数据都会经过权重和偏移的影响，将值从左到右的传递下去直到最后output输出。也就是每一个输出的数据的变动经过这一系例的变化最后都将或多或少的影响到最后的输出结果。现在我们将权值和阀值标注到神经网络中去。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralWB.png" width=50% height=50%>

b2(2)是因为画图时不方便表示公式的妥协在这文中写做![b](http://latex.codecogs.com/png.latex?b_{2}^{2})代表第2层网络第2个神经元的偏移，w3(12)在文中写成![pi](http://latex.codecogs.com/png.latex?w_{12}^{3})表示第N-1层j个神经元到**第N层**的第i个神经元的权值，落这里就是第2层的第2个神经元到**第3层**的第1个神经元的权值。看到权值的标示方式可能会觉得很别扭，觉得下标的12应该从左到右进行标示才更合理，但是实际这样表示是为了后面进行计算矩阵运算方便。现在我们就是为了得到某种分类模型需要通过大量的训练数据得到权值和偏移的值。

## 求解权值和偏移

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralZA.png" width=50% height=50%/>

首先准备下进行算法解释的图像，可以看出来和上一节的图像的差别是多了![pi](http://latex.codecogs.com/png.latex?z_{3}^{2})和![pi](http://latex.codecogs.com/png.latex?a_{1}^{3})。其中z表示神经元接受到的信息包括带有权值的输入信号和偏移，a表示激活函数sigmoid根据z计算出来的激活值。

### 代价函数

按照我们训练一个模型的一般思路，我们需要找到一个代价函数，这样可以得到一个优化的目标函数。这里我们使用的代价函数是：

![pi](http://latex.codecogs.com/png.latex?C(w,b)=\frac{1}{2n}\sum_{x}\left&space;\|&space;y&space;(x)-a&space;\right&space;\|^{2})   

这里y(x)代表训练集x期望的输出，a代表x经过模型得到的输出，n代表训练样本的数量，w代表的是权值，b代表的是偏移，这里的1/2的作用是为了后续对代价函数求导方便。

![pi](http://latex.codecogs.com/png.latex?a_{j}^{l}=\sigma&space;(\sum_{k}w_{jk}^{l}a_{k}^{k-1}&space;&plus;&space;b_{j}^{l}))  

![pi](http://latex.codecogs.com/png.latex?z_{j}^{l}=\sum_{k}w_{jk}^{l}a_{k}^{k-1}&space;&plus;&space;b_{j}^{l}) 

对应可以写成向量的形式

![pi](http://latex.codecogs.com/png.latex?a^{l}=\sigma&space;(w^{l}a^{l-1}&plus;b^{l}))

![pi](http://latex.codecogs.com/png.latex?z^{l}=w^{l}a^{l-1}&plus;b^{l})

得到了这些后，接下来我们需要的就是求得使的代价函数获得最小值的w和b，但是想通过直接求导求解出w和b是很困难的。后面采用的是使用梯度下降的方式来不断逼近最优解。

### 梯度下降

[梯度下降法](https://zh.wikipedia.org/wiki/梯度下降法)的理解，可以通过一个三维的凹陷的面来理解如下图：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/Gd.png" width=50% height=50%/>

如果我们需要从上图中的一个任意点到达这个谷底应该如何做呢？一种考虑就是通过计算偏导数来获得运动的方向。再通过移动一小步，然后再去计算偏导数获得梯度再延着梯度移动一小步。就这样一步步的我们逐渐会逼近上图中的谷底，对应到函数也就是最优解。表达为数学公式为：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralGraW.png"/>

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralGraB.png"/>

这里的![pi](http://latex.codecogs.com/png.latex?\eta)表示的学习的速率，它越大移动的步子就越大，但是如果过大可能跨过最佳点，如果过小会造成移动的过慢学习的过程也将变的非常缓慢。对于我们的公式在进行梯度下降涉及到一个问题，具体的梯度下降的多少需要针对每一个训练的样本来计算一次梯度下降的多少然后求平均得到真正下降的值。为什么会这样呢？因为我们家谁的代价函数就是针对每个训练样本的的期望值和训练过程输出值之间的差并进行平均的处理。

设想如果训练的样本非常的大，那同样会极大的影响模型学习的时间。这个时候就带出了**随机梯度下降**算法，它和梯度下降的区别就是，随机梯度下降算法每次执行下降并不需要输入所有的训练样本。随机梯度下降算法会先随机选择固定少的多训练样本来执行梯度下降。这样整体的学习的时间就减少了很多，同时随机梯度算法依然会找到优化的最佳点。


### BP 反向传播算法

现在我们需要处理的问题进一步变成了如何求得在代价函数C中的w和b的偏导数。首先我们可以了解到求这个两个的偏导数的是可以得到当w或者b发生变化时代价函数改变的速度有多快。这里我们介绍[反向传播算法](http://www.tensorfly.cn/home/?p=76)来处理w和b的偏导数计算问题。

这里我们需要预先设定一些假设，其一是各个训练样本在带入代价函数后对w和b求导后对应每个训练样本的求导结果相加并求平均可以得到整个代价函数的梯度，其二可以将代价函数当成是激活层a的输出的一个函数。

在进行反向传播是我们先设定一个中间变量![pi](http://latex.codecogs.com/png.latex?\delta&space;_{j}^{l})它表示网络中第l层第j个神经元的误差。注意这里的误差只是一种表述，并不需要和一般的误差含义等同。反向传播可以通过![pi](http://latex.codecogs.com/png.latex?\delta&space;_{j}^{l})得到w和b的导数。

![pi](http://latex.codecogs.com/png.latex?\delta&space;_{j}^{l}=\frac{\partial&space;C}{\partial&space;z_{j}^{l}})   

#### 输出层误差

![pi](http://latex.codecogs.com/png.latex?\delta&space;_{L}^{l}=\frac{\partial&space;C}{\partial&space;a_{j}^{L}}{\sigma&space;}'(z_{j}^{L}))

注意这里的乘的关系是对应元素相乘。这个等式是通过链式求导法则得到的。

![pi](http://latex.codecogs.com/png.latex?\delta&space;_{L}^{l}=\frac{\partial&space;C}{\partial&space;a_{j}^{L}}\frac{\partial&space;a_{j}^{L}}{\partial&space;z_{j}^{L}})

其中

![pi](http://latex.codecogs.com/png.latex?{\sigma&space;}'(z_{j}^{L})=\frac{\partial&space;a_{j}^{L}}{\partial&space;z_{j}^{L}})

#### 非输出层误差

![pi](http://latex.codecogs.com/png.latex?\sigma_{l}=((w^{l&plus;1})^{T}\delta&space;^{l&plus;1})\odot&space;{\sigma&space;}'(z^{l}))

首先我们是希望这个误差出现传递的特点的。所以这个公式，我们可以通过这样的链式求导：

![pi](http://latex.codecogs.com/png.latex?\sigma_{j}^{l}=\sum_{k}\frac{\partial&space;C}{\partial&space;z_{k}^{l&plus;1}}\frac{\partial&space;z_{k}^{l&plus;1}}{\partial&space;z_{j}^{l}})

结合

![pi](http://latex.codecogs.com/png.latex?z_{k}^{l&plus;1}=\sum_{j}w_{kj}^{l&plus;1}a_{j}^{l}&plus;b_{k}^{l&plus;1}=\sum_{j}w_{kj}^{l&plus;1}\sigma&space;(z_{j}^{l})&plus;b_{k}^{l&plus;1})

可以得到开头的公式

#### 权值的梯度

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;w_{jk}^{l}}=a_{k}^{l-1}\delta&space;_{j}^{l})

依据链式求导法则

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;w_{jk}^{l}}=\frac{\partial&space;C}{\partial&space;z_{j}^{l}}\frac{\partial&space;z_{j}^{l}}{\partial&space;w_{jk}^{l}}=\delta&space;_{j}^{l}\frac{\partial&space;(w_{jk}^{l}a_{k}^{l-1}&plus;b_{j}^{l})}{\partial&space;w_{jk}^{l}}=a_{k}^{l-1}\delta&space;_{j}^{l})

#### 偏移的梯度

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;b_{l}^{j}}=\delta&space;_{j}^{l})

依据链式求导法则

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;b_{l}^{j}}=\frac{\partial&space;C}{\partial&space;z_{j}^{l}\frac{z_{j}^{l}}{\partial&space;b_{j}^{l}}}=\delta&space;_{j}^{l}\frac{\partial&space;(w_{jk}^{l}a_{k}^{l-1})}{\partial&space;b_{l}^{j}}=\delta&space;_{j}^{l})

### 算法描述

1. 输入训练样本集合

2. 对于每个训练样本x：设置对应的输入激活![pi](http://latex.codecogs.com/png.latex?a^{x,1})，执行以下步骤：

* 向前：对于每一层l=2，3，4....，L计算![pi](http://latex.codecogs.com/png.latex?z^{x,l}=w^{l}a^{x,l-1}&plus;b^{l})和![pi](http://latex.codecogs.com/png.latex?a^{x,l}=\sigma&space;(z^{x,l}))

* 输出层误差![pi](http://latex.codecogs.com/png.latex?\delta&space;^{x,L}):计算向量![pi](http://latex.codecogs.com/png.latex?\delta&space;^{x,L}=\triangledown&space;_{a}C_{x}\odot&space;{\sigma&space;}'(z^{x,L}))

* 后向传播误差：对于每一层l=L-1,L-2.....计算![pi](http://latex.codecogs.com/png.latex?\delta&space;^{x,L}=((w^{l&plus;1}^{T})\delta&space;^{x,l&plus;1})\odot&space;{\sigma&space;}'(z^{x,L}))   

3. 梯度下降：对于每一层l=L-1,L-2,.....,按照规则更新权值和偏移：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralGraW.png"/>

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralGraB.png"/>


## 参考数据  
[1]《机器学习》 周志华 著 

[2]《斯坦福大学公开课：机器学习课程 cs229 吴恩达 

[3]《coursera 机器学习课程》 吴恩达 

[4][《神经网络和深度学习》](http://www.tensorfly.cn/home/?p=80) 

[5][《反向传导算法》](http://deeplearning.stanford.edu/wiki/index.php/反向传导算法)   

