# SVM 学习笔记    
## 背景     
[支持向量机(support vector machines)](https://zh.wikipedia.org/wiki/支持向量机)是一种二分类模型，基础的模型形式是根据间隔最大来进行分类的
线性分类器。根据特殊的映射函数(核函数)可以实现非线性分类。支持向量机是现成的最好的分类器，这里的“现成”指的是分类器不加修改即可直接使用(这是评价很好的
意思)。支持向量机的学习使用到部分线性代数的知识，在学习的过程中也是遵循由简单到复杂的过程。      
## SVM 支持向量机     

<img src ="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmGrapherWithText.jpg" width = 50% height = 50%/>
比如我们有如上形式的数据，在对它们进行分类的时候我们可以通过画一条线将上下的数据分隔开，但是如你所想的，可以将这两类数据划分开的直线有无数条。那这个时
候我们应该选择哪条直线来进行数据的划分。      
<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmGrapher3line.jpg" width = 50% height = 50%/>      
直观的看我们会倾向于选择红色的的直线来进行数据的划分。选择红色直线的理由是当这个时候有一个新的点需要被划分时，使用红色直线进行划分更有可能进行正确的划分因为它
给正例和反例都流出了最多的空间。       
所以在这里我们的寻找这个最优划分的直线的依据就是寻找间隔最大的直线。我们目前讨论的是在二维平面的划分所以是通过直线进行划分，如果是在训练的数据是在三维空间
则划分的就是一个平面，对于N维的空间中的数据我们可以通过N-1维来进行划分。这个N-1维用来划分的向量我们称它为分隔超平面(hyperplane)，分隔超平面可以写成这
样的形式:    

![pi](http://latex.codecogs.com/png.latex?w^{T}x&plus;b=0) 

也可以写成:

![pi](http://latex.codecogs.com/png.latex?w\cdot&space;x&plus;b=0)

这里![pi](http://latex.codecogs.com/png.latex?w\cdot&space;x)表示w和x的内积。如上公式表示w和x的内积也可以通过w的转置后和x相乘来求的。    

## 参考书籍
《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著        
《斯坦福大学公开课：机器学习课程 cs229 吴恩达      
《coursera 机器学习课程》 吴恩达     
《[SVM-tutorial](http://www.svm-tutorial.com)》 Alexandre KOWALCZYK    
