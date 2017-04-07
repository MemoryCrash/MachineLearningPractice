# 决策树学习笔记     

## 背景
我们在生活中总是会面对很多需要决策或在叫作出判断的事情。一般人进行判断的时候都会根据一些特点(特征)来进行判断。比如电视剧里表现的侦探破案时会更加犯罪现场的各种情况层层筛选去判断信息(这个可以关注‘今日说法’节目)。我们嘛一般就是判断身高是高还是低，年收入是高中还是一般，是否是本地户口，是否有房子。嗯，对的这个是相亲的判断的问题。         
决策树这种监督学习算法也是这样的考虑，从一堆特征中去构造一个这样的决策树使其泛化性能最好(能更好的预测未知分类的数据)就像下面的图形示，内部节点表示一个一个的特征比如身高、户口、收入这些。叶子节点表示分类。我们这里举例是二分类所以我们的分类就是‘是’或者‘否’这样的内容。

<img src = 'https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/decisionTree.png' width=30% height=30%/>

## 决策树

在生成决策树的时候我们面临这样的问题就是选择什么的特征来进行划分。特征的选取是基于对训练数据具有分类能力的特征，这样可以提高决策树的学习效率。那怎么样的特征就是更具备分类能力的特征呢？我们需要找到一个选择特征的依据。我们依据的是信息增益。

### 信息增益

在信息论与概率统计中熵(entropy)是度量样本集合纯度的一个重要指标。假设集合D中第一k类样本所占的比例为![pi](http://latex.codecogs.com/gif.latex?p_{k})(k=1,2,3...,Y)，则D的信息熵定义为:

![pi](http://latex.codecogs.com/gif.latex?Ent(D)=-\sum_{k=1}^{Y}p_{k}log_{2}p_{k})

Ent(D)的值越小则，D的纯度越高。现在我们找到度量样本集合纯度的方法。现在需要的是找到一种划分效果最好的特征，按照这个思路来进行理解，就是使用某个特征进行划分后的样本集合的信息熵变小了也就是对应到样本集合的纯度在按照某个特征进行划分后得到了提高。当然我们的一个训练样本对应了肯定不止一个特征，所以我们就需要在这些特征中找到能使的样本集合的纯度提高的最多的那个作为分类划分的特征。我们用信息增益表示按照某个特征进行分类后信息纯度提高的量。   

假设集合D有特征a，其中a特征可以取的值有N个，同时![pi](http://latex.codecogs.com/gif.latex?\left&space;|&space;D^{n}&space;\right&space;|)表示a特征的某个取值对应的集合样本数量。![pi](http://latex.codecogs.com/gif.latex?\left&space;|&space;D\right&space;|)表示总的训练样本数量。那么使用特征a来进行分类的信息增益就是:

![pi](http://latex.codecogs.com/gif.latex?g(D,a)=Ent(D)-\sum_{n=1}^{N}\frac{\left&space;|&space;D^{n}&space;\right&space;|}{\left&space;|&space;D&space;\right&space;|}Ent(D^{n}))

### 生成决策树

我们依据信息增益来生成决策树具体流程如下:
输入:训练数据集合D，特征集A，阀值![pi](http://latex.codecogs.com/gif.latex?\varepsilon)
输出:决策树

* 1 若D中所有实例属于同一类![pi](http://latex.codecogs.com/gif.latex?C_{k})，则T为单结点树，并将类![pi](http://latex.codecogs.com/gif.latex?C_{k})作为该结点的类标记，返回T
* 2 若![pi](http://latex.codecogs.com/gif.latex?A=\phi)，则T为单结点树，并将D中实例树最大的类![pi](http://latex.codecogs.com/gif.latex?C_{k})作为该结点的类标记，返回T
* 3 否则，计算A中各个特征对D的信息增益，选择信息增益最大的特征![pi](http://latex.codecogs.com/gif.latex?A_{g})
* 4 如果![pi](http://latex.codecogs.com/gif.latex?A_{g})的信息增益小于阀值![pi](http://latex.codecogs.com/gif.latex?\varepsilon)，则置T为单节点树，并将D中实例数最大的类![pi](http://latex.codecogs.com/gif.latex?C_{k})作为该结点的类标记，返回T
* 5 否则，对![pi](http://latex.codecogs.com/gif.latex?A_{g})的每一可能值![pi](http://latex.codecogs.com/gif.latex?a_{i})，依![pi](http://latex.codecogs.com/gif.latex?A_{g}=a_{i})将D分割为若干非空子集![pi](http://latex.codecogs.com/gif.latex?D_{i})，将![pi](http://latex.codecogs.com/gif.latex?D_{i})中实例数最大的类作为标记，构建子结点，由结点及子结点构成树T，返回T
* 6 对第i个子结点，以![pi](http://latex.codecogs.com/gif.latex?D_{i})为训练集，以![pi](http://latex.codecogs.com/gif.latex?A-A_{g})为特征集，递归地调用步1～步5，得到子树![pi](http://latex.codecogs.com/gif.latex?T_{i})返回![pi](http://latex.codecogs.com/gif.latex?T_{i})

### 信息增益和信息增益率
当我们使用信息增益来为依据进行分类的时候会发现这种方法对于取值比较多的属性有所偏好，为了减少这种偏好引入了信息增益率:

![pi](http://latex.codecogs.com/gif.latex?g_{R}(D,A)=\frac{g(D,a)}{H_{a}(D)})

![pi](http://latex.codecogs.com/gif.latex?H_{a}(D)=-\sum_{i=1}^{n}\frac{\left&space;|&space;D_{i}&space;\right&space;|}{\left&space;|&space;D&space;\right&space;|}log_{2}\frac{\left&space;|&space;D_{i}&space;\right&space;|}{\left&space;|&space;D&space;\right&space;|})

这里我们可以看出分母是的取值是如果a的取值越多那么就越大，通过这样来惩罚取值比较多的属性。但是这样又可能产生对取值比较少的特征的偏好。所以在实际情况下会综合下各种情况，先从候选划分属性中找出**信息增益**高于平均水平的属性，再从这里面选择**信息增益率**比较高的特征出来。

## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著        
《斯坦福大学公开课：机器学习课程 cs229 吴恩达       
《coursera 机器学习课程》 吴恩达 
