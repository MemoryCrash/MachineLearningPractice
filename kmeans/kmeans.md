# K-means&EM&GMM 学习笔记

## 背景

我们的学习算法有些是监督学习算法，有些是非监督学习算法。像是logistic回归、支持向量机、决策树、贝叶斯分类器，都是监督学习。他们的特点是训练样本中包含了分类信息。但是实际生活中有些情况我们也不了解训练数据的分类标签，我们需要的就是学习算法可以帮助我们将训练样本进行分类当然这个过程中学习算法也并不知道这样分类的具体含义，然后我们根据学习算法分好的类给这些类赋予特定的标签。总的来说对没有分类标签的训练数据进行学习的就是非监督学习。

## K-means

在非监督学习中比较典型的就是聚类算法，就是将训练样本中关系密切的数据划分为一类。在聚类算法中就有k-means算法。在k-means算法中我们如何判断哪些训练样本是关系密切的呢？我们是这样判断的，先随机提供几个中心点出来也就是随机提供几个我们认为分出的类。然后计算各个样本点距离这些中心点的距离。样本点距离哪个中心的距离最近则将这个样本点标记为中心所属的类。各个样本点都分配好中心点以后，以类为颗粒计算样本点的均值。然后将这些均值赋值给对应类的中心。这样就算是更新中心的值了。按照这个节奏迭代下去直到更新值小于你设定的阀值或者不再发生变化。

这里可能会有疑问，随机产生的中心点数目靠谱不？因为分类本身是比较主观的事情。随意这样随机产生的类的数目还说比较靠谱的。

## EM

EM即是expectation-maximization，是一种迭代算法。分为两步，E步求期望；M步求极大值。EM的一个应用就是GMM(混合高斯模型)。我们在观察数据的时候可能存在隐藏数据，举个例子我们的数据由多个高斯分布组合成，但是我们并不知道具体哪个数据是属于那个高斯分布。我们所观察到的只是混合后所呈现出来的数据。但是我们最终是需要将生成数据的高斯分布都求解出来。EM 算法迭代的过程如下图示：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/em.jpg" width=50% height=50% />

### EM 算法的步骤(此处参考[2]中第9章的内容)

输入：观察变量数据Y，隐变量数据Z，联合分布![pi](http://latex.codecogs.com/png.latex?P(Y,Z|\theta&space;))，条件分布![pi](http://latex.codecogs.com/png.latex?P(Z|Y,\theta&space;))；

输出：模型参数![pi](http://latex.codecogs.com/png.latex?\theta)    

1. 选择参数的初值![pi](http://latex.codecogs.com/png.latex?\theta&space;^{0})，开始迭代；

2. E步：记![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i})为第i次迭代参数![pi](http://latex.codecogs.com/png.latex?\theta)的估计值，在第i+1次迭代的E步，计算

&emsp;![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i})=E_{z}[logP(Y,Z|\theta&space;)|Y,\theta&space;^{i}])

&emsp;&emsp;&emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?=\sum_{Z}logP(Y,Z|\theta&space;)P(Z|Y,\theta&space;^{i}))

这里，![pi](http://latex.codecogs.com/png.latex?P(Z|Y,\theta&space;^{i}))是在给定观测数据Y和当前的参数估计![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i})下隐变量数据Z的条件概率分布；

3. M步：求使![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i}))极大后的![pi](http://latex.codecogs.com/png.latex?\theta)，确定第i+1次迭代的参数的估计值![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i&plus;1})

&emsp;![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i&plus;1}=arg&space;max_{\theta&space;}Q(\theta&space;,\theta&space;^{i}))

4. 重复第2步和第3步，直达收敛。

### Q函数   

完全数据的对数似然函数![pi](http://latex.codecogs.com/png.latex?logP(Y,Z|\theta&space;))关于在给定观测数据Y和当前参数![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i})下对未观测数据Z的条件概率分布![pi](http://latex.codecogs.com/png.latex?P(Z|Y,\theta&space;^{i}))的期望称为Q函数，即

![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i})=E_{z}[logP(Y,Z|\theta&space;)|Y,\theta&space;^{i}])

下面关于EM算法作几点说明：    
步骤1 参数的初值可以任意选择，但是需注意EM算法对初值是敏感的.     
步骤2 E步求![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i}))。Q函数式中Z是未观测数据，Y是观测数据。注意，![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i}))的第1个变元表示要极大化的参数，第2个变元表示参数的当前估计值。每次迭代实际在求Q函数及其极大。       
步骤3 M步求![pi](http://latex.codecogs.com/png.latex?Q(\theta&space;,\theta&space;^{i}))的极大化，得到![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i&plus;1})，完成一次迭代![pi](http://latex.codecogs.com/png.latex?\theta&space;^{i}\rightarrow&space;\theta&space;^{i&plus;1})后面将证明每次迭代使似然函数增大或达到局部极值。     
步骤4 给出停止迭代的条件，一般是对较小的正数![pi](http://latex.codecogs.com/png.latex?\varepsilon&space;_{1},\varepsilon&space;_{2})若满足.      

![pi](http://latex.codecogs.com/png.latex?\left&space;\|&space;\theta&space;^{i&plus;1}-\theta&space;^{i}&space;\right&space;\|<\varepsilon&space;_{1})或者![pi](http://latex.codecogs.com/png.latex?\left&space;\|&space;Q(\theta&space;^{i&plus;1},\theta&space;^{i})&space;-Q(\theta&space;^{i},\theta&space;^{i})\right&space;\|<\varepsilon&space;_{2})

则停止迭代

## GMM

## 参考书籍

[1]《机器学习实战》 Peter Harrington 著 李锐 译    
[2]《统计学习方法》 李航 著   
[3]《机器学习》 周志华 著        
[4]《斯坦福大学公开课：机器学习课程 cs229 吴恩达     
[5][《斯坦福大学机器学习——EM算法求解高斯混合模型》](http://blog.csdn.net/linkin1005/article/details/41212085)     
[6][《EM算法与混合高斯模型》](http://blog.csdn.net/star_dragon/article/details/51058591#fnref:footnote)
