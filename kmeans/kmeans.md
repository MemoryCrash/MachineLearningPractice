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

## EM2(算法另外解释并不是新的算法参考[5])

这里会涉及到[jensen不等式](https://zh.wikipedia.org/wiki/延森不等式)可以根据下图来理解：     
<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/jensen.jpg" width=50% height=50%/>

在一个凸函数中f(E[X])<=E[f(X)]，而等号成立的条件是E[X]=X，也就是X是常数。在这里我们继续回到我们现在面对的情况。我们现在有些训练数据假设他们复合某种分布现在需要找到他们的分布的具体参数。我们可以通过极大似然法去寻找最能体现训练数据的分布参数。       
![pi](http://latex.codecogs.com/png.latex?l(\theta&space;)=\sum_{i=1}^{m}log&space;p(x,\theta&space;))           
这里我们使用的对数似然，这样是方便计算，并不会影响结果。刚才的情况我们设想了一种简单的情况并且得到了一个似然估计的式子，但在实际中我们可能面对这样的情况。训练数据可能是由多个分布混合而成，并不是一个单一的分布就可以表现，并且我们观察到的仅仅只是数据本身，至于隐藏在背后的这个训练数据是属于哪个分布就不得而知了。                    
这里我们假设存在一个隐藏的变量Z它表明了具体某个训练数据是属于哪个分布。这里时候我们的似然公式就可以更新如下：

![pi](http://latex.codecogs.com/png.latex?l(\theta&space;)=\sum_{i=1}^{m}log&space;\sum_{z}p(x,z;\theta&space;))

不过我们并不能直接根据这个公式来求解，因为这里的z是未知的。但是我们可以根据不停的迭代去逼近最佳的解(当然也极有可能掉入到一个局部最优解)，这里我们引入z的分布函数![pi](http://latex.codecogs.com/png.latex?Q_{i})，它是随意变量![pi](http://latex.codecogs.com/png.latex?z_{i})的分布函数。并且有：

![pi](http://latex.codecogs.com/png.latex?\sum_{z}Q_{i}(z)=1,Q_{i}(z)\geq&space;0)

接下来利用这个，引入到我们更新过的公式中得到：

![pi](http://latex.codecogs.com/png.latex?l(\theta&space;)=\sum_{i}log\sum_{z^{i}}Q_{i}(z^{i})\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})}\geq&space;\sum_{i}\sum_{z^{i}}Q_{i}(z^{i})log\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})})&emsp;(1)

后面的不等式是根据jensen不等式得到。方法是将![pi](http://latex.codecogs.com/png.latex?\sum_{z^{i}}Q_{i}(z^{i})\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})}) 看作是关于分布Q关于后面![pi](http://latex.codecogs.com/png.latex?\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})})的期望。f(x)就是log函数，是个凹函数，和我们刚才以凸函数为例介绍的jensen不等式结论刚好相反。变成了f(E[x])>=E[f(x)]，这样就得到了(1)式中的大于等于号右边的内容。为什么我们需要这个不等式？因为有了这个不等式我们就知道我们估计的似然函数的下界。并且在![pi](http://latex.codecogs.com/png.latex?\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})})等于常数的时候就取等号。

那么![pi](http://latex.codecogs.com/png.latex?Q_{i}(z^{i}))我们如何获取呢？    
我们根据jensen不等式的等号成立条件即真实的似然曲线和我们估计值相等的条件:

![pi](http://latex.codecogs.com/png.latex?\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})}=c)    
结合        
![pi](http://latex.codecogs.com/png.latex?\sum_{z}Q_{i}(z)=1)     
得到：   
![pi](http://latex.codecogs.com/png.latex?Q_{i}(z^{i})=\frac{p(x^{i},z^{i};\theta&space;)}{\sum_{z}p(x^{i},z;\theta&space;)})     
&emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?=\frac{p(x^{i},z^{i};\theta&space;)}{p(x^{i};\theta&space;)})      
&emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?=p(z^{i}|x^{i};\theta&space;))      

现在就可以具体执行EM算法    

1. 设置初始化值![pi](http://latex.codecogs.com/png.latex?\theta)，求似然Q的最大似然值，这个就是E(期望)。
2. 利用Q来更新![pi](http://latex.codecogs.com/png.latex?\theta)，这就是M(最大化)。
然后迭代下去。    
伪代码如下：   
Repeat until convergence{    
(E-step)For each i, set   
&emsp;&emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?Q_{i}(z^{i}):=p(z^{i}|x^{i};\theta&space;))   
(M-step)Set    
&emsp;&emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?\theta&space;:=arg&space;max_{\theta&space;}\sum_{i}\sum_{z^{i}}Q_{i}(z^{i})log\frac{p(x^{i},z^{i};\theta&space;)}{Q_{i}(z^{i})})   
}


<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/em.jpg" width=50% height=50%/>

## GMM

## 参考书籍

[1]《机器学习实战》 Peter Harrington 著 李锐 译    
[2]《统计学习方法》 李航 著   
[3]《机器学习》 周志华 著        
[4]《斯坦福大学公开课：机器学习课程 cs229 吴恩达     
[5][《斯坦福大学机器学习——EM算法求解高斯混合模型》](http://blog.csdn.net/linkin1005/article/details/41212085)     
[6][《EM算法与混合高斯模型》](http://blog.csdn.net/star_dragon/article/details/51058591#fnref:footnote)
