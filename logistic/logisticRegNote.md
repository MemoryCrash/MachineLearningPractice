# **logistic 回归学习笔记**

## 线性回归

给定训练数据，学习训练数据的规律，然后通过拟合曲线去预测某个值，这个方式可以理解为线性回归。假设现在有一些数据，我们通过一个直线对这些数据进行拟合的过程就可以称为回归。

## logistic 回归

还有一种情况我们考虑的不是去拟合某个曲线，而是去在数据中划一条线出来以此来对数据进行分类。这个就可以称为逻辑回归了。利用 logist 回归进行分类的主要思想是:根据现有数据对分类边界线建立回归公式，以此进行分类。这里的“回归”一词源于最佳拟合，表示要找到最佳拟合参数集。

我们在进行预测练习的时候，先是进行2分类的练习，是或者不是，那么判断函数一般会联想到阶跃函数，但是在这里考虑到阶跃函数不可导的特点，所以并没有选择他。我们选择了sigmoid 函数，他能比较好满足我们判断0/1的要求同时又是可导的。sigmoid 函数的公式:![pi](http://latex.codecogs.com/png.latex?\sigma&space;\left&space;(&space;z&space;\right&space;)=\textstyle\frac{1}{1&plus;e^{-z}})图形如下。

<img src ="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmoid.png" width = 50% height = 50%/>

可以看出来在任何大于0.5的数据被划分到1类，小于0.5的数据划分到0类。选择好了判断函数，接下来就需要对输入判断函数的数据进行处理。在这里我们认为输入的数据满足这样的公式![pi](http://latex.codecogs.com/gif.latex?z=w_{0}x_{0}&plus;w_{1}x_{1}&plus;w_{2}x_{2}&plus;....&plus;w_{n}x_{n})用向量的形式表示为![pi](http://latex.codecogs.com/gif.latex?z=w^{T}x)这里需要说明下这个<font color="red">向量x</font>就代表了我们训练模型的**输入数据**，这里的**向量w**就代表了这些**数据的系数**我们的目的就是通过训练模型获取最优的w。怎么样算是最优【可以更好的预测未知数据】是一个很关键的判定原则。
## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著    
《斯坦福大学公开课：机器学习课程 cs229 吴恩达    
《coursera 机器学习课程》 吴恩达
