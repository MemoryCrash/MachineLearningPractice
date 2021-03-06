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

在这两个平面之间没有训练集合中的点，训练集合中的点要么落在他们上面(这些使得等式成立的就叫做支持向量)或者落在他们后面去，训练集合在这两个平面上表现为:

![pi](http://latex.codecogs.com/png.latex?y_{i}(w^{T}x_{i}&space;&plus;b)\geq&space;1&space;,(i=1.....N))      

这里y表示训练集中对应的分类，因为是2分类正类y为1反类y为-1而当训练集合的点落在上图的两个虚线后面时如果是正例为正值如果是反例为负值这样就可以统一为上面的公式。由图上有点A落在上部的虚线上则一定有一个根据法向量方向移动m距离的点会落在下面的虚线上。我们可以根据这个关系求解出m来。   

![pi](http://latex.codecogs.com/png.latex?x_{a'}=x_{a}&plus;\frac{w}{\left&space;\|&space;w&space;\right&space;\|}m)      

在根据得到的新点是可以落在下面的虚线，所以带入公式:

![pi](http://latex.codecogs.com/png.latex?w^{T}x&space;&plus;b&space;=&space;-1)

得到      

![pi](http://latex.codecogs.com/png.latex?w^{T}(x_{a}-\frac{w}{\left&space;\|&space;w&space;\right&space;\|}m)&plus;b&space;=&space;-1)

求解以后可以得到(也可以根据平行线的距离公式得到):

![pi](http://latex.codecogs.com/png.latex?m=\frac{2}{\left&space;\|&space;w&space;\right&space;\|})

这个时候找到了m的表达，我们需要的就是在特定条件下最大化m。为了计算方便最大化m等价于

minimize &emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2})

subject to &emsp;&emsp;&emsp;![pi](http://latex.codecogs.com/png.latex?y_{i}(w^{T}x_{i}&space;&plus;b)\geq&space;1&space;,(i=1.....N))

### 对偶算法    
为了求解**线性可分支持向量机**的最优化问题，其中一个方法就是利用[拉格朗日乘子法](https://en.wikipedia.org/wiki/Lagrange_multiplier)整合约束信息后再根据对偶性求解对偶问题。因为一般来讲对偶问题会比原问题更容易求解。关于拉格朗日乘子法简述如下，假设![pi](http://latex.codecogs.com/png.latex?f(x))，![pi](http://latex.codecogs.com/png.latex?c_{i}(x))，![pi](http://latex.codecogs.com/png.latex?h_{j}(x))是定义在![pi](http://latex.codecogs.com/png.latex?R^{n})上的连续可微函数。考虑约束最优化问题:

![pi](http://latex.codecogs.com/png.latex?min&space;f(x),x\epsilon&space;R^{n})     

s.t.

&emsp;![pi](http://latex.codecogs.com/png.latex?c_{j}(x)\leq0&space;,i=1,2,....,k)    

&emsp;![pi](http://latex.codecogs.com/png.latex?h_{j}(x)=0,j=1,2,.....l)    

称此约束最优化问题为原始最优化问题或原始问题。这里引入广义拉格朗日函数:

![pi](http://latex.codecogs.com/png.latex?L(x,\alpha&space;,\beta&space;)=f(x)&plus;\sum_{i=1}^{k}\alpha&space;_{i}c_{i}(x)&plus;\sum_{j=1}^{l}\beta&space;_{j}h_{j}(x))

![pi](http://latex.codecogs.com/png.latex?\alpha&space;_{i})和![pi](http://latex.codecogs.com/png.latex?\beta&space;_{j})是拉格朗日乘子，其中![pi](http://latex.codecogs.com/png.latex?\alpha&space;_{i}\geq&space;0)。根据拉格朗日乘子法，我们将线性可分支持向量机的求解m的公式和其约束转换为一个公式如下:

![pi](http://latex.codecogs.com/png.latex?L(w,b,\alpha&space;)=\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}&plus;\sum_{i=1}^{N}\alpha&space;_{i}(1-y_{i}(w\cdot&space;x_{i}&plus;b)))&emsp;(1)

其中![pi](http://latex.codecogs.com/png.latex?\alpha&space;_{i}\geqslant&space;0)，![pi](http://latex.codecogs.com/png.latex?\alpha&space;=(\alpha&space;_{1},\alpha&space;_{2}.....,\alpha&space;_{N})^{T})为拉格朗日乘子向量。这是一个极小极大问题，如果我们固定了w和b这个时候在满足约束的情况下得到的公式就是![pi](http://latex.codecogs.com/png.latex?L(w,b,\alpha&space;)=\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2})这个时候就可以进一步进行![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2})的最小化。在这里我们不直接求解这个公式而是转而求解它的对偶问题先基于w和b最小化![pi](http://latex.codecogs.com/png.latex?L(w,b,\alpha&space;))再基于![pi](http://latex.codecogs.com/png.latex?\alpha)最大化![pi](http://latex.codecogs.com/png.latex?L(w,b,\alpha&space;))这个时候就将问题转化为了极大极小问题。现在我们距离计算这个极大极小问题。先通过对w和b分别求导并另其为0求的w和b，对![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2})求导，对应了一个多元函数对各个变量进行求导的过程最后得到的结果就是w:

![pi](http://latex.codecogs.com/png.latex?w=\sum_{i=1}^{N}\alpha&space;_{i}y_{i}x_{i})

![pi](http://latex.codecogs.com/png.latex?\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)

将这两个公式带入公式(1)得到:
基于![pi](http://latex.codecogs.com/png.latex?\alpha)最大化下面的公式

![pi](http://latex.codecogs.com/png.latex?-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}(x_{i}\cdot&space;x_{j})&plus;\sum_{i=1}^{N}\alpha&space;_{i})&emsp;(2)

将(2)式转化为最小化问题:

![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}(x_{i}\cdot&space;x_{j})-\sum_{i=1}^{N}\alpha&space;_{i})&emsp;(3)

s.t.

&emsp;![pi](http://latex.codecogs.com/png.latex?\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)

&emsp;![pi](http://latex.codecogs.com/png.latex?\alpha&space;_{i}\geq&space;0,(i=1,2,....N))

### SMO 优化算法        

#### 取值范围

考虑使用[SMO优化算法](https://zh.wikipedia.org/wiki/序列最小优化算法)来求解(3)的问题。在这个算法中我们需要寻找![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{i})的值来使得(3)出现最优解。这里我们先介绍一个叫做坐标上升的算法(对应有个坐标下降)，它的思想是当需要求解多变量的凸函数的最优解的时候，我们选择先对其中一个变量求极限值同时固定其他变量这样逐步去逼近最优解。在这里 SMO 算法也是借鉴了这样的算法。但是受到![pi](http://latex.codecogs.com/png.latex?\textstyle\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)约束的限制。在 SMO 算法中我们选择一次改变两个变量![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{1},\alpha&space;_{2})。根据约束条件:

![pi](http://latex.codecogs.com/gif.latex?0\leq&space;\alpha&space;_{i}\leq&space;C)

![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{1}y_{1}&plus;\alpha&space;_{2}y_{2}=-\sum_{i=3}^{N}\alpha&space;_{i}y_{i}=\xi)

可以画出如下的图像

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmRange.png"/>

这个图形表现了![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{1},\alpha&space;_{2})的取值范围。现在我们将其通过公式表达出来这里我们以求解
![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2})
在
![pi](http://latex.codecogs.com/gif.latex?y_{1}\neq&space;y_{2})
条件下取值范围为例子。根据上面的公式我们可以得出这样的结论:

![pi](http://latex.codecogs.com/gif.latex?\alpha_{1}^{old}-\alpha_{2}^{old}=\alpha_{1}^{new}-\alpha_{2}^{new}=k)

结合

![pi](http://latex.codecogs.com/gif.latex?0\leq&space;\alpha_{1}^{new}\leq&space;C)

得到

![pi](http://latex.codecogs.com/gif.latex?-k\leq&space;\alpha_{2}^{new}\leq&space;C-k)

将k替换掉得到

![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2}^{old}-\alpha&space;_{1}^{old}\leq&space;\alpha_{2}^{new}\leq&space;C&plus;\alpha&space;_{2}^{old}-\alpha&space;_{1}^{old})

![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{1}^{new})也可以同样这样得到。同时说明如果![pi](http://latex.codecogs.com/gif.latex?y_{1}=y_{2})时

![pi](http://latex.codecogs.com/gif.latex?\alpha_{1}^{old}+\alpha_{2}^{old}=\alpha_{2}^{new}+\alpha_{2}^{new}=k)

这个这样理解的如果y的相等则符号相同y不相等则符号相反而y作为输出只能取1或者-1。求得这个取值范围后的作用是什么呢？主要是提供给我们对优化后的![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{i})进行裁剪使用。

#### 更新公式

下面我们直接给出![pi](http://latex.codecogs.com/gif.latex?\alpha_{2}^{new})的更新公式:

![pi](http://latex.codecogs.com/gif.latex?\alpha_{2}^{new}=\alpha&space;_{2}^{old}&plus;\frac{y_{2}(E_{1}-E_{2})}{\eta&space;})

其中

![pi](http://latex.codecogs.com/gif.latex?\eta&space;=K_{11}&plus;K_{22}-2K_{12})

![pi](http://latex.codecogs.com/gif.latex?E_{i}=g_{x_{i}}-y_{i}=\left&space;\{&space;\sum_{j=1}^{N}\alpha&space;_{j}y_{j}K(x_{j},x_{i})&plus;b&space;\right&space;\}-y_{i},i=1,2...)

![pi](http://latex.codecogs.com/gif.latex?E_{i})其实就是模型输出的值和训练集本身的值之间的误差。
当获得了根据下面的判断条件对![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2}^{new})进行裁剪:
* ![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2}^{new}>&space;H)取H

* ![pi](http://latex.codecogs.com/gif.latex?L\leq&space;\alpha&space;_{2}^{new}\leq&space;H)取![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2}^{new})

* ![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{2}^{new}<&space;L)取L

另

![pi](http://latex.codecogs.com/gif.latex?\alpha&space;_{1}^{new}=&space;\alpha&space;_{1}^{old}&plus;y_{1}y_{2}(&space;\alpha&space;_{2}^{old}-\alpha&space;_{2}^{new}))

#### 变量获取

现在更新的公式有来我们接下来需要做的事获取一对变量来进行更新。首先检查在![pi](http://latex.codecogs.com/gif.latex?0<\alpha&space;_{i}<C)范围中的变量是否满足KKT条件的变量，有的话就选择违反最严重的作为第一个点。如果这个范围没有，就遍历整个训练集合来选择违反KKT条件的点。然后在选定第一个点后第二个待优化点根据![pi](http://latex.codecogs.com/gif.latex?\left&space;|&space;E_{1}-E_{2}&space;\right&space;|)最大化的原则来选择第二个待优化点。
这样选择待优化点的意图是让我们每次的优化和更新是更加有效的。

#### 计算阀值

在计算完两个优化变量以后我们都需要更新阀值:

![pi](http://latex.codecogs.com/gif.latex?b_{1}^{new}=-E_{1}-y_{1}K_{11}(\alpha&space;_{1}^{new}-\alpha&space;_{1}^{old})-y_{2}K_{21}(\alpha_{2}^{new}-\alpha&space;_{2}^{old})&plus;b^{old})

![pi](http://latex.codecogs.com/gif.latex?b_{2}^{new}=-E_{2}-y_{1}K_{12}(\alpha&space;_{1}^{new}-\alpha&space;_{1}^{old})-y_{2}K_{22}(\alpha_{2}^{new}-\alpha&space;_{2}^{old})&plus;b^{old})

当![pi](http://latex.codecogs.com/gif.latex?0<&space;\alpha&space;_{i}<&space;C)时![pi](http://latex.codecogs.com/gif.latex?b_{1}^{new})和![pi](http://latex.codecogs.com/gif.latex?b_{2}^{new})相等，这个好理解因为满足这个条件的是支持向量，而支持向量对应的分隔平面是同一个，所以这个平面对应的阀值也是一样的。当其他情况的我们选择两个阀值的平均值来作为新的阀值。

获取新的阀值基于此给出我们的![pi](http://latex.codecogs.com/gif.latex?E_{i})的更新函数:

![pi](http://latex.codecogs.com/gif.latex?E_{i}^{new}&space;=&space;\sum_{S}y_{j}\alpha&space;_{j}K(x_{i},x_{j})&plus;b^{new}-y_{i})

S代表所有支持向量的集合。

### 软间隔         
先前我们讨论的都是完全线性可分的数据，但是实际中的数据可能是线性不可分数据，通常情况是训练数据中有一些奇异点，将这些奇艺点去掉以后，剩下大部分集合是线性可分的。这就可以修改硬间隔最大化，使其成为软间隔最大化。   

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmSoftMargin.jpg" width=50% height = 50%/>  

如图所示如果出现了一个奇异点k，按照硬间隔的方式来查找最大间隔必然会得到红色虚线代表的分隔曲线，但是这个红色虚线并不是我们期望的。我们依然希望是以黑色的虚线来进行分类。后来是通过引入一个松弛变量![pi](http://latex.codecogs.com/png.latex?\xi&space;_{i}\geq&space;0)使得函数间隔加上了松弛变量后大于等于1。这样约束，条件就变为这样:  

![pi](http://latex.codecogs.com/png.latex?y_{i}(w\cdot&space;x_{i}&plus;b)\geq&space;1-\xi&space;_{i})

同时进行最优化的函数也随之变化为:

![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{N}\xi&space;_{i})   
这里的C>0称为惩罚参数，一般由应用问题决定，C值大时对误分类的惩罚增大，C值小时对误分类的惩罚减小。最小化目标函数的过程包含了使间隔尽可能大和使误分点尽可能少。根据硬间隔的方式寻找对偶问题后可以发现软间隔的对偶问题和硬间隔公式(3)非常相似只是约束条件发生了变化:

![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}(x_{i}\cdot&space;x_{j})-\sum_{i=1}^{N}\alpha&space;_{i})&emsp;(4)

s.t.

&emsp;![pi](http://latex.codecogs.com/png.latex?\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)

&emsp;![pi](http://latex.codecogs.com/png.latex?0\leq&space;\alpha&space;_{i}\leq&space;C,(i=1,2,....N))

### 非线性支持向量机

考虑这样的待分类数据我们是无法直接通过一条直线来进行分类的:

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmKernel.jpg" width = 50% height = 50%>

对于这样的问题我们可以将它转换为一个线性的问题来进行解决。在现实生活中我们也常遇到分类的问题，当我们根据当前特征对这个事物无法进行分类的时候，一般可以通过增加一个维度来进行分类。这里我们也是曲借鉴这样的想法，将数据进行转换后在进行分类。针对上图的数据可以通过对数据进行平方后转换为一个线性问题继续处理。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/svmChangeKernel.jpg" width = 50% height = 50%/>

将低维数据映射到高维后可以将待优化的公式表示成这样:

![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}(\phi&space;(x_{i})\cdot&space;\phi&space;(x_{j}))-\sum_{i=1}^{N}\alpha&space;_{i})&emsp;(5)

s.t.

&emsp;![pi](http://latex.codecogs.com/png.latex?\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)

&emsp;![pi](http://latex.codecogs.com/png.latex?0\leq&space;\alpha&space;_{i}\leq&space;C,(i=1,2,....N))

![pi](http://latex.codecogs.com/png.latex?\phi&space;(x)) 表示对数据映射到高维，对映射后的数据同样进行内积操作，但是存在一个映射到高维的数据计算量更大的情况。这里就通过一个核函数来进行处理。核函数输入的是映射前的数据，但是输出的是映射后的数据进行内积的结果。这样可以极大的减少计算量。这样我们进一步将公式(5)写为:

![pi](http://latex.codecogs.com/png.latex?\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}K(x_{i},&space;x_{j})-\sum_{i=1}^{N}\alpha&space;_{i})&emsp;(6)

s.t.

&emsp;![pi](http://latex.codecogs.com/png.latex?\sum_{i=1}^{N}\alpha&space;_{i}y_{i}=0)

&emsp;![pi](http://latex.codecogs.com/png.latex?0\leq&space;\alpha&space;_{i}\leq&space;C,(i=1,2,....N))

常用的核函数有：

* 多项式核函数

&emsp;![pi](http://latex.codecogs.com/png.latex?K(x,z)=(x\cdot&space;z&plus;1)^{p})

* 高斯核函数

&emsp;![pi](http://latex.codecogs.com/png.latex?K(x,z)=exp(-\frac{\left&space;\|&space;x-z&space;\right&space;\|^{2}}{2\sigma&space;^{2}}))

## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译           
《统计学习方法》 李航 著   
《机器学习》 周志华 著        
《斯坦福大学公开课：机器学习课程 cs229 吴恩达      
《coursera 机器学习课程》 吴恩达     
《[SVM-tutorial](http://www.svm-tutorial.com)》 Alexandre KOWALCZYK               
《[支持向量机通俗导论(理解SVM的三层境界)](http://blog.csdn.net/v_july_v/article/details/7624837)》 JULY

