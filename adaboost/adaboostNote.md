# 集成学习学习笔记

## 背景

在我们决策一个事情的时候，特别是比较重要的事情时，一般来讲我们会去征求下多个人的意见然后综合各方意见再做出决定。本次讨论的集成学习中的adaBoost也是基于这样的考虑。我们目前的讨论是针对使用相同的“基础分类器”来进行分类的二分类情况。这种综合其它的情况将他们的输出作为一个最终决策的输出，给人的感觉和神经网络有点像。

## adaBoost

在学习的方法中有有些学习方法称为强可学习有些称为弱可学习，从名字可以看出来有些学习方法效果并不太好但是弱学习方法一般是比较容易实现的。这个时候其实是有方法将弱学习提升(boost)成为强学习，就像那句话一样“三个臭皮匠顶个诸葛亮”我们需要的就是找到方法让他们能顶上。adaBoost全称是adaptive boosting(自适应提升)，它就是这样的方法。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/adaBoost.png" width=50% height=50%/>

针对同样的一组训练样本，我们使用使弱学习方法去反复学习它然后每次学习都得到一个模型出来这样我们就拥有了多个决策者了。但是这个时候你也会发现如果只是这样的话那么为每次都是针对同样的训练数据去学习这样得到的训练模型也是大致相同的，并起不到集思广益的特点。所以我们不仅要得到多个模型还需要他们都有自己的想法。

这个时候adaBoost是这样去处理的。对于每次的学习得到的模型它实际的效果必然是有些分类正确有些分类失败。我们就把它分类失败的训练样本给一个更高的权值这样在下次学习的时候这部分分类失败的数据就会被特别关注。按照这样的情况我们循环下去直到分类好。然后我们就得到了多个决策者，考虑到有些决策者分类正确的情况比较多有些分类正确的情况比较少，他们的决策信息的可信度是不一致的。所以我们根据他们分类的正确情况给这些决策者赋予不同的权值，这里的权值判断并不是去选择权值最大的，是大家投票后统计票数来决定最终的分类结果。这就是adaBoost的基本思想。

### 算法实现

输入：训练数据集![pi](http://latex.codecogs.com/gif.latex?T&space;=&space;\{(x_{1},y_{1}),(x_{2},y_{2}),.....,(x_{N},y_{N})\})，其中![pi](http://latex.codecogs.com/gif.latex?x_{i}\in&space;\chi&space;\subseteq&space;R^{n})，![pi](http://latex.codecogs.com/gif.latex?y_{i}\in&space;\{-1,&plus;1\})；弱学习算法(可以是一个单层决策树)；

输出：最终分类器G(x)

* 1 初始化数据的权值分布
&emsp;![pi](http://latex.codecogs.com/gif.latex?D_{1}=(w_{11},...,w_{1i},...,w_{1N}),w_{1i}=\frac{1}{N},i=1,2,...,N)

* 2 对m=1,2,...,M

* 2.1 使用具有权值分布![pi](http://latex.codecogs.com/gif.latex?D_{m})的训练数据集学习，得到基本分类器
&emsp;![pi](http://latex.codecogs.com/gif.latex?G_{m}:\chi&space;\rightarrow&space;\{-1,&plus;1\})

* 2.2 计算![pi](http://latex.codecogs.com/gif.latex?G_{m}(x))在训练数据集上的分类误差率
&emsp;![pi](http://latex.codecogs.com/gif.latex?e_{m}=P(G_{m}(x_{i})\neq&space;y_{i})=\sum_{i=1}^{N}w_{mi}I(G_{m}(x_{i})\neq&space;y_{i}))

* 2.3 计算![pi](http://latex.codecogs.com/gif.latex?G_{m}(x))的系数，这里的对数是自然对数
&emsp;![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{m}=\frac{1}{2}log\frac{1-e_{m}}{e_{m}})

* 2.4 更新训练数据集的权值分布
&emsp;![pi](http://latex.codecogs.com/gif.latex?D_{m&plus;1}=(w_{m&plus;1,1},...,w_{m&plus;1,i},...,w_{m&plus;1,N)})
&emsp;![pi](http://latex.codecogs.com/gif.latex?w_{m&plus;1,i}=\frac{w_{mi}}{Z_{m}}exp(-\alpha&space;_{m}y_{i}G_{m}(x_{i})),i=1,2,...,N)

&emsp;这里，![pi](http://latex.codecogs.com/gif.latex?Z_{m})是规范化因子，它使的![pi](http://latex.codecogs.com/gif.latex?D_{m&plus;1})成为一个概率分布。

* 3 构建基本分类器的线性组合

&emsp;![pi](http://latex.codecogs.com/gif.latex?f\left&space;(&space;x&space;\right&space;)=\sum_{m=1}^{M}\alpha&space;_{m}G_{m}(x))

&emsp;得到最终分类器

&emsp;![pi](http://latex.codecogs.com/gif.latex?G_{x}=sign(f(x))=sign(\sum_{m=1}^{M}\alpha&space;_{m}G_{m}(x)))

可以通过下图来理解。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/adaBoostDetail.jpg" width=50% height = 50%/>

这里的第一排的图中黑色条代表我们的训练数据，而每一个基学习器对应训练数据可以看到有些长还有些短这就是改变训练数据权值的过程。最后每个基学习器对应了一个三角形表示这个学习器的表决权重。最后进行汇总输出结果。

## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著   
