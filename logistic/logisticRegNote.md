# **logistic 回归**

## 线性回归

给定训练数据，学习训练数据的规律，然后通过拟合曲线去预测某个值，这个方式可以理解为线性回归。假设现在有一些数据，我们通过一个直线对这些数据进行拟合的过程就可以称为回归。

还有一种情况我们考虑的不是去拟合某个曲线，而是去在数据中划一条线出来以此来对数据进行分类。这个就可以称为逻辑回归了。利用 logist 回归进行分类的主要思想是:根据现有数据对分类边界线建立回归公司，以此进行分类。这里的“回归”一词源于最佳拟合，表示要找到最佳拟合参数集。

我们在进行预测练习的时候，先是进行2分类的练习，是或者不是，那么判断的还是一般会很快的联想到阶跃函数，但是由于在这里考虑到阶跃函数不可导的特点，所以并没有选择他，我们选择了sigmod 函数，他的能比较好的满足我们判断0/1的要求同时又是可导的。sigmod 函数的公式:![pi](http://latex.codecogs.com/png.latex?\sigma&space;\left&space;(&space;z&space;\right&space;)=\textstyle\frac{1}{1&plus;e^{-z}})图形如下。

<img src ="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmod.png" width = 50% height = 50%/>
