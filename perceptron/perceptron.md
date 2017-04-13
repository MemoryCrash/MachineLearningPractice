# 感知机学习笔记

## 背景
感知机是一种二分类的**线性分类模型**，只能用来对线性可分的数据进行分类，如果是面对线性不可分的数据就会出现震荡模型训练中无法收敛。要实现非线性数据的分类可以参考svm。当我们面对一堆数据的时候，期望通过一个超平面(对于二维数据超平面是一条直线)将数据划分为两类。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmChangeKernel.jpg" width=50% height=50%/>.  

## 感知机

感知的模型为如下这样：    
![](http://latex.codecogs.com/gif.latex?f(x)=sign(w\cdot&space;x&plus;b))   
这里f(x)是一个符号函数。sign(x)当x大于等于0的时候输出1，当x小于0的时候输出-1。其中w是权值，b代表的是阀值。w和x是做内积运算。我们可以通过下面这个图来进行理解：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/perceptron.png"/>  


## 参考书籍

[1]《统计学习方法》 李航 著   
[2]《机器学习》 周志华 著        
[3]《斯坦福大学公开课：机器学习课程 cs229 吴恩达     
