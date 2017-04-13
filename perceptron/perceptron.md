# 感知机学习笔记

## 背景
感知机是一种二分类的**线性分类模型**属于监督学习，只能用来对线性可分的数据进行分类，如果是面对线性不可分的数据就会出现震荡模型训练中无法收敛。要实现非线性数据的分类可以参考svm。当我们面对一堆数据的时候，期望通过一个超平面(对于二维数据超平面是一条直线)将数据划分为两类。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmChangeKernel.jpg" width=50% height=50%/>            

&emsp;&emsp;[图1]
## 感知机

感知的模型为如下这样：    
![](http://latex.codecogs.com/gif.latex?f(x)=sign(w\cdot&space;x&plus;b))   
这里f(x)是一个符号函数。sign(x)当x大于等于0的时候输出1，当x小于0的时候输出-1。其中w是权值，b代表的是阀值。w和x是做内积运算。我们可以通过下面这个图来进行理解：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/perceptron.png"/>      
&emsp;&emsp;[图2]

有趣的是如果按照图1来进行理解将感知机拓展到非线性可分数据就可以引入SVM(支持向量机)，如果按照图2将其拓展到可以支持非线性数据就可以引入神经网络。可见感知机是一个基础机器学习模型。

现在我们已经获知了感知的模型，现在的任务就是通过训练数据来获得模型中w和b的值。换个思路，怎么样的模型就是好的模型呢？至少在训练模型的阶段如果输入的训练样本得到的预测分类结果和实际分类结果一样就是比较好的模型，说是比较好，是考虑到可能出现的过拟合情况。所以我们可以定义一个函数来表示模型的预测结果和实际结果之间的差异的函数。这个函数我们一般称为代价函数。当这个代价函数我们可以取到最小的时候，那这样的模型也就是一个比较好的模型了。 

那问题现在转换为找到一个合适的代价函数。参考[1]中2章的描述，我们可以想到的方式是通过计算误分点的数量作为代价函数，但是这样的函数不可导难以计算。所以进一步考虑我们可以选择将误分点到分离平面的距离作为是代价函数，这样就方便多了。根据点到直线的距离公式可以得到一个误分点到分离超平面的距离是：

![pi](http://latex.codecogs.com/gif.latex?\frac{\left&space;|&space;w\cdot&space;x_{0}&plus;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|})

如何判断误分点，可以这样设想如果点被正确分类，那么这个点应该满足![pi](http://latex.codecogs.com/gif.latex?y_{0}(w\cdot&space;x_{0}&space;&plus;&space;b)>&space;0)对应如果是一个未分类正确的点，那么就应该是这样![pi](http://latex.codecogs.com/gif.latex?-y_{0}(w\cdot&space;x_{0}&space;&plus;&space;b)>&space;0)根据这些，进一步获得所有误分点的的距离和是：

![pi](http://latex.codecogs.com/gif.latex?-\frac{1}{\left&space;\|&space;w&space;\right&space;\|}\sum_{x_{i\in&space;M}}y_{i}(w\cdot&space;x_{i}&space;&plus;&space;b))

在这里我们不需要考虑![pi](http://latex.codecogs.com/gif.latex?\frac{1}{\left&space;\|&space;w&space;\right&space;\|})因为它在这里是固定的，最后得到的感知机损失函数为：

![pi](http://latex.codecogs.com/gif.latex?L(w,b)=-\sum_{x_{i}\in&space;M}y_{i}(w\cdot&space;x_{i}&plus;b))

### 最小化代价函数
这里我们采用随机梯度下降法来最小化代价函数。我们先找到代价函数涉及的两个变量的梯度：

![pi](http://latex.codecogs.com/gif.latex?\triangledown&space;_{w}L(w,b)=-\sum_{x_{i}\in&space;M}y_{i}x_{i}). 

![pi](http://latex.codecogs.com/gif.latex?\triangledown&space;_{b}L(w,b)=-\sum_{x_{i}\in&space;M}y_{i}).  

通过对关注的参数求导得到了梯度，但是这个还不是随机梯度。这个是所有训练数据计算后的梯度。随机梯度只需要去掉求和符号选择其中一组数据即可。这样我们的对w和b的更新策略是：

![pi](http://latex.codecogs.com/gif.latex?w\leftarrow&space;w-\eta&space;(-y_{i}x_{i})). 

![pi](http://latex.codecogs.com/gif.latex?b\leftarrow&space;b-\eta&space;(-y_{i})).   

这里的![pi](http://latex.codecogs.com/gif.latex?\eta)学习的速率。这样我们就获得了算法需要的内容。下面是具体的算法步骤：

## 参考书籍

[1]《统计学习方法》 李航 著   
[2]《机器学习》 周志华 著        
[3]《斯坦福大学公开课：机器学习课程 cs229 吴恩达     
