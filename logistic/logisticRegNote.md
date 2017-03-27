# **logistic 回归学习笔记**
## 背景
这里的 logistic 回归是属于监督学习，监督学习的特点是训练模型的数据中给定了标签，简单说就是我们知道训练的场景对应了什么样的结果。
## 线性回归

给定训练数据，学习训练数据的规律，然后通过拟合曲线去预测某个值，这个方式可以理解为线性回归。假设现在有一些数据，我们通过一个直线对这些数据进行拟合的过程就可以称为回归。

## logistic 回归

这里我们考虑的不是去拟合某个曲线，而是去在数据中划一条线出来以此来对数据进行分类。这个就可以称为逻辑回归了。利用 logist 回归进行分类的主要思想是:根据现有数据对分类边界线建立回归公式，以此进行分类。这里的“回归”一词源于最佳拟合，表示要找到最佳拟合参数集。

我们在进行预测练习的时候，先是进行2分类的练习，是或者不是，那么判断函数一般会联想到阶跃函数，但是在这里考虑到阶跃函数不可导的特点，所以并没有选择他。我们选择了sigmoid 函数，他能比较好满足我们判断0/1的要求同时又是可导的。sigmoid 函数的公式:![pi](http://latex.codecogs.com/png.latex?g&space;\left&space;(&space;z&space;\right&space;)=\textstyle\frac{1}{1&plus;e^{-z}})图形如下。

<img src ="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmoid.png" width = 50% height = 50%/>

可以看出来在任何大于0.5的数据被划分到1类，小于0.5的数据划分到0类。选择好了判断函数，接下来就需要对输入判断函数的数据进行处理。在这里我们认为输入的数据满足这样的公式![pi](http://latex.codecogs.com/png.latex?z=w_{0}x_{0}&plus;w_{1}x_{1}&plus;w_{2}x_{2}&plus;....&plus;w_{n}x_{n})用向量的形式表示为![pi](http://latex.codecogs.com/png.latex?z=w^{T}x)这里需要说明下这个**向量x**就代表了我们训练模型的**输入数据**，这里的**向量w**就代表了这些**数据的系数**我们的目的就是通过训练模型获取最优的w。怎么样算是最优呢，【可以更好的预测未知数据】是一个很关键的判定原则。

### 假设函数

![pi](http://latex.codecogs.com/png.latex?h_{w}\left&space;(&space;x&space;\right&space;)=\textstyle\frac{1}{1&plus;e^{-w^{T}x}})就是我们的假设函数。假设函数可以基于我们当前的系数w的内容输入训练数据然后输出一个结果来，我们再通过对比输出的结果和训练数据对应的标签来进行下一步的调整。我们在训练模型阶段会期望模型输出的值能和训练数据本身的标签尽可能的对应上。那如何对模型进行调整了这里调整的对象其实就是w。这个时候我们会很自然的构造一个代价函数。这里还有点![pi](http://latex.codecogs.com/png.latex?h_{w}(x))可以理解为给定x和w后的输出结果为正例的条件概率。这样可以得到如下公式:   
![pi](http://latex.codecogs.com/png.latex?P(y=1|x;w)=h_{w}(x))&emsp;&emsp;&emsp;&emsp;&emsp;(1)    
![pi](http://latex.codecogs.com/png.latex?P(y=0|x;w)=1-h_{w}(x))&emsp;&emsp;&emsp;(2)       
可以将公式(1)(2)合并为如下公式    
![pi](http://latex.codecogs.com/png.latex?P(y|x;w)=(h_{w}\left&space;(&space;x&space;\right&space;))^{y}(1-h_{w}\left&space;(&space;x&space;\right&space;))^{1-y})&emsp;&emsp;&emsp;(3)
### 代价函数

这里考虑公式(3)表达了输出值是正例或者反例的概率，那可以考虑到找到对应的w使的输出正例或者反例的概率最大。[最大似然估计](https://zh.wikipedia.org/wiki/最大似然估计)便是用来进行概率最大化的方法。我们似然函数是<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/CodeCogsEqn.png" width = 15% height = 15%/>m代表的是训练样本的数量，在这里有一种思路，现在我们面对的连乘实际是不好处理这个时候可以对等式两边求对数log这样就将连乘转换成了求和。这里考虑到我们是求取使的代价函数最优的w的值所以这样的求对数的过程并不会影响到w的取值。 

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/loglike.png" width = 12% height = 12%/>   
    
<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/loglikeexp.png" width = 35% height = 35%/> 

观察上面的等式实际上依据最大似然法的话我们应该是求给定 w 下 l(w) 的最大值，这个应该是求最大值的过程。
设 J(w) 就是我们用来衡量训练数据获得的输出结果与真实结果之间差异的损失函数![pi](http://latex.codecogs.com/png.latex?J\left&space;(&space;w&space;\right&space;)=-\frac{1}{m}l(w))既然是损失函数哪我们期望的就是损失最小化，所以就在原l(w)前加了一个符号，而对于1/m，在我自己理解这个项并不会影响最终的结果。那我们现在的问题就转换为求解使得 J(w) 最小的 w。对应的方法就是梯度下降法。
### 梯度下降

[梯度下降](https://zh.wikipedia.org/wiki/梯度下降法)的公式为：        
![pi](http://latex.codecogs.com/png.latex?w_{j}:=w_{j}-\alpha&space;\frac{\partial&space;}{\partial&space;w_{j}}J(w)&space;,&space;(j=0....n))      
这里需要注意n代表的是一个训练样本中拥有的特征数量，比如判断一个车的好坏就包括外观、马力、内饰等，刚才提到的m是训练样本的数量比如判断车好坏的例子我们会收集100条包含了外观、马力、内饰这些信息的数据。再回到梯度下降公式，现在我们看到的这个公式是针对单条训练数据来说，⍺ 代表的是步长需要你自己来设定取值过小梯度下降的会比较慢，设置大了可能会越过最佳的点。在梯度下降的过程中最关键的就是求解![pi](http://latex.codecogs.com/png.latex?\textstyle\frac{\partial&space;}{\partial&space;w_{j}}J(w))在求导的过程中会使用到[链式求导法则](https://zh.wikipedia.org/wiki/链式法则)，公式中的log代表了以e为底的对数，准确点以上公式中 log 应该写成 ln，而对 ln(x) 求导就是 1/x。对![pi](http://latex.codecogs.com/png.latex?e^{x})求导的结果依然是![pi](http://latex.codecogs.com/png.latex?e^{x})对梯度下降公式进行简化以后得到  
![pi](http://latex.codecogs.com/png.latex?w_{j}:=w_{j}-\alpha&space;\sum_{i=1}^{m}(h_{w}(x^{i})-y^{i})x_{j}^{i},(j=0...n))&emsp;&emsp;&emsp;(4) 

转换成向量形式     
![pi](http://latex.codecogs.com/png.latex?w:=w-\alpha&space;\cdot&space;x^{T}\cdot&space;(g(x\cdot&space;w)-y))     
注意这里进行的是![pi](http://latex.codecogs.com/png.latex?\cdot)代表是向量的内积。一般 logistic 回归的求解过程就是这样了。
## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著    
《斯坦福大学公开课：机器学习课程 cs229 吴恩达    
《coursera 机器学习课程》 吴恩达   
[【机器学习笔记1】Logistic回归总结](http://blog.csdn.net/dongtingzhizi/article/details/15962797)
