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

直观的看我们会倾向于选择红色的的直线来进行数据的划分。选择红色直线的理由是当这个时候有一个新的点需要被划分时，使用红色直线进行划分更有可能进行正确的划分因为它给正例和反例都流出了最多的空间。所以在这里我们的寻找这个最优划分的直线的依据就是寻找间隔最大的直线。我们目前讨论的是在二维平面的划分所以是通过直线进行划分，如果是在训练的数据是在三维空间则划分的就是一个平面，对于N维的空间中的数据我们可以通过N-1维来进行划分。这个N-1维用来划分的向量我们称它为分隔超平面(hyperplane)，分隔超平面可以写成这样的形式:    

![pi](http://latex.codecogs.com/png.latex?w^{T}x&plus;b=0) 

也可以写成:

![pi](http://latex.codecogs.com/png.latex?w\cdot&space;x&plus;b=0)

这里![pi](http://latex.codecogs.com/png.latex?w\cdot&space;x)表示w和x的内积。如上公式表示w和x的内积也可以通过w的转置后和x相乘来求的。这里可能有疑问为什么不是用y = ax+b这样的形式来表示。两种表示表达的意思的一致的，但是向量的表示法在高维下更方便，同时使用了向量的表示法后w代表的就是超平面的法向量这个对我们后续求解训练集中的点到超平面的距离非常有用。    
![pi](http://latex.codecogs.com/png.latex?y&space;-ax-b&space;=&space;0)    
w = [-1, -a]      
x = [x0, x1]        
![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;-b&space;=&space;0)      

这里是-b因为b代表的是一个偏置所以它的符号实际并不起到特别的作用。      

### 内积的几何含义      
两个向量的内积代表一个a向量在b向量上的投影长度和b向量范数(长度)的乘积。
<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmCdot.jpg" width = 50% height = 50%/>    
利用这个特点我们可以得到训练集中的点到超平面的距离就等于对应点的向量在超平面法向量上的投影和法向量的范数的乘积。我们知道法向量的长度是可以变化的。我们在计算中可以设置法向量的长度为单位长度这样训练点到超平面的距离就是固定的。单位法向量下得到的间隔就是“几何间隔”，非单位法向量得到的间隔就是“函数间隔”。

### 最大间隔      
<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmMargin.jpg" width = 50% height = 50%/>

在了解了如何在空间中求一个点到超平面之间的距离了，接下来就可以运用这个找到超平面，根据超平面的用来分隔两个类并拥有到两个类之间的最大距离的特点。我们寻找超平面等价于找到两个类之间的最大间隔。如上图示我们需要使得m获取最大值，同时我们假定除了超平面![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;&plus;b&space;=&space;0)外还有两个与他平行的平面分别是:        
![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;&plus;b&space;=&space;1)         
![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;&plus;b&space;=&space;-1)         

在这两个平面之间没有训练集合中的点，训练集合中的点要么落在他们上面或者落在他们后面去，训练集合在这两个平面上表现为:              
![pi](http://latex.codecogs.com/png.latex?y_{i}(w^{T}x_{i}&space;&plus;b)\geq&space;1&space;,(i=1.....N))      

这里y表示训练集中对应的分类，因为是2分类正类y为1反类y为-1而当训练集合的点落在上图的两个虚线后面时如果是正例为正值如果是反例为负值这样就可以统一为上面的公式。由图上有点A落在上部的虚线上则一定有一个根据法向量方向移动m距离的点会落在下面的虚线上。我们可以根据这个关系求解出m来。   

![pi](http://latex.codecogs.com/png.latex?x_{a'}=x_{a}&plus;\frac{w}{\left&space;\|&space;w&space;\right&space;\|}m)      

在根据得到的新点是可以落在下面的虚线，所以带入公式:

![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;&plus;b&space;=&space;-1)

得到      

![pi](http://latex.codecogs.com/png.latex?w^{T}(x_{a}-\frac{w}{\left&space;\|&space;w&space;\right&space;\|}m)&plus;b&space;=&space;-1)

求解以后可以得到:

![pi](http://latex.codecogs.com/png.latex?m=\frac{2}{\left&space;\|&space;w&space;\right&space;\|})

这个时候找到了m的表达，我们需要的就是在特定条件下最大化m。为了计算方便最大化m等价于

minimize &emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2})

subject to &emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?y_{i}(w^{T}x_{i}&space;&plus;b)\geq&space;1&space;,(i=1.....N))

## 参考书籍
《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著        
《斯坦福大学公开课：机器学习课程 cs229 吴恩达      
《coursera 机器学习课程》 吴恩达     
《[SVM-tutorial](http://www.svm-tutorial.com)》 Alexandre KOWALCZYK    
