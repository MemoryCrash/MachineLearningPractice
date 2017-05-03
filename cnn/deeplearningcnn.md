# 卷积神经网络简略笔记

## 卷积神经网络与神经网络的关系
神经网络的关系和卷积神经网络的关系可以理解过卷积神经网络是为了可以更好处理图片而对神经网络进行了相应的改造和特殊化。这里的以卷积来为这个新的神经网络进行
命名，这里主要是卷积神经网络的卷积层在对原始数据的处理上类似于卷积操作。如果本身对数学上的卷积操作不怎么了解，其实也不影响对卷积神经网络的具体理解。

## 神经网络处理图片的问题
我们可以使用神经网络来处理图片，对图片进行学习分类。但是神经网络可能对处理尺寸较小的，不复杂的图片处理更擅长。对于具有相反特点的图片神经网络的学
习就不理想了。其中的原因包括有神经网络会将图片的所有像素信息都传入到神经网络的输入层中，当图片过大或者需要进行更复杂的分类时会导致模型中出现大量的
参数。这样直接导致的是计算量变得极大最终难以计算。而且为了进行更复杂的判断新增的隐藏又可能导致梯度消失的情况。

## 卷积神经网络
为了减少上面问题的影响，发展出了卷积神经网络。最主要的特点是添加了卷积层来对图片进行处理。卷积神经网络模型的样子如下：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/simple_conv.png" width=50% height=50%/>

简单说下上图中convolutional layer就是卷积层，主要作用是通过过滤器来过滤出原始图片的特征，一般一个卷积层有多个过滤器所以也就能得到多个特征来。接
下来是pooling layer是池化层主要是将卷积层产生的特征信息进行采样以进一步减小图片的尺寸。接着将池化后的数据输入到全连接层，这里的全连接层就和神经网
络是一样的类，接着是输入到输出层得到分类结果。

### 激活函数
在神经网络中我们一般使用sigmod作为激活函数，它的图像如下

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmoid.png" width=50% height=50%/>

这里可以看出来在偏离图像的中心点0越远则图像的变换的趋势越加缓慢，如果在初始化w和b的过程可能某些神经元获得了一个更大的值。这就引起了梯度的下降变的缓慢
解决的方式可以选择换一个激活函数比如relu函数，它的图像如下：

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/relu.png" width=50% height=50%>

### 代价函数 sigmoid+交叉熵
在先前使用sigmod作为激活函数的神经网络中我们使用代价函数的模式一般是平方差的形式。代价函数的形式是会依赖对激活函数求导，我们知道求导对应到曲线可以理解
为曲线在某点的变化率。以下通过一个一个输入一个输出的单个神经元模型来进行解释。以下输入假设x是1，期待y输出为0可以看出求导的结果确实依赖了对sigmod的求
导，这样就会出现上面讨论的梯度下降缓慢的问题。

![p](http://latex.codecogs.com/png.latex?C=\frac{(y-a)^{2}}{2})

![p](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;w}=(a-y){\sigma&space;}'(z)x=a{\sigma&space;}'(z))

![p](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;b}=(a-y){\sigma&space;}'(z)=a{\sigma&space;}'(z))

通过使用交叉熵来作为代价函数：

![pi](http://latex.codecogs.com/png.latex?C=-\frac{1}{n}\sum_{x}[ylna&plus;(1-y)ln(1-a)])

可以对w和b分别求导发现结果是和激活函数的导数没有关系的。

### softmax+log-likelihood
除了交叉熵以外还可以通过softmax的方式来解决学习速度衰减的问题。我们仅将输出层从普通的sigmod作为激活函数的层替换为softmax层。softmax输出层同样
接受z=wx+b然后通过以下公式来计算输出结果

![pi](http://latex.codecogs.com/png.latex?a_{j}^{L}=\frac{e^{z_{j}^{L}}}{\sum_{k}e^{z_{k}^{L}}})

可以看出来这里得到的是某个值占总体的一个比例。配合softmax我们的代价函数需要替换成log-likelihood

![pi](http://latex.codecogs.com/png.latex?C\equiv&space;-lna_{y}^{L})

这里表示的是单个输入样本的代价，如果有多个样本的可以对他们的代价求均值，作为总的代价函数。通过代价函数对w和b求导得到公式

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;b_{j}^{L}}=a_{j}^{L}-y_{j})

![pi](http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;w_{jk}^{L}}=a_{k}^{L-1}(a_{k}^{L}-y_{j}))

### 正则化

一般有l1,l2,dropout的方式来对模型进行正则化，主要目的还是防止模型的过拟合，其中l2正则化方法是对所有的w进行如下处理并把这个部分添加到代价函数中去
需要注意的是l2正则化只需包括w不需要包括b。

![p](http://latex.codecogs.com/png.latex?C=-\frac{1}{n}\sum_{xj}[y_{j}lna_{j}^{L}&plus;(1-y_{j})ln(1-a_{j}^{L})]&plus;\frac{\lambda&space;}{2n}\sum_{w}w^{2})

这样起到的作用使的优化这个代价函数，会更倾向于获取一个w并不是那么复杂的模型。一般直观的来看过拟合的模型都是在训练集中表现过于好的函数，而表现的过好的
模型一般w都是比较复杂的。我们对于l1正则化不作解释。

接下还有一种方式是dropout，可以理解为丢弃。处理的思想类似我们以前看的集成学习法。这dropout中会随机的丢弃一些神经元(非输入和输出层)，也就是我们每次迭
代更新梯度只对模型中的部分神经元进行处理。在一个batch的样本训练并更新完w和b以后我们会重新再进行一次dropout。这样的感觉就像训练了多个子神经网络，最后
将他们组合成一个大的神经元来进行使用。是不是和集成学习思想类似？

### 参数初始化
在先前的神经网络中一般采用的是符合一个均值是0，标准差是1的高斯分布的随机数去初始化w和b，但是这个并不适合初始化隐藏层比较多的神经网络。一般是将标准差表示
为![pi](http://latex.codecogs.com/png.latex?1/\sqrt{n_{in}})这里的![pi](http://latex.codecogs.com/png.latex?n_{in})表示具有输入权
重的神经元个数。

### 卷积层
卷积层是卷积神经网络不同于神经网络的一个重要特点。首先我们输入到卷积神经网络的依然是一个图片，但是作为接受这个数据的卷积层并不是如果神经网络一样将图片的
所有像素信息都输入，而是使用一个比输入图片小的多的过滤器来扫描原始输入的图片数据。过滤器可以理解为将原始图片的一小部分作为输入并通过权值和偏移进行计算后
输入激活函数得到在新的数据。这个过滤器会按照设置每次滑动并重复刚才的过程。最终得到的就是对应一个过滤器得到的一个特征map。这里需要特别说明单独过滤器滑动
的任何区域计算使用的w和b都是一样的，一般的讲这个可以称为**参数共享**。这样的好处是识别某个特征将和这个特征在原始图片出现的位置无关。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/numerical_no_padding_no_strides.gif" width=50% height=50%/>

可以直观的看到经过过滤器处理以后的图像的大小变小了，同时多个过滤器也可以获得到图片的多种特征。同时可以看出来我们需要仔细设计过滤器的大小以及每次滑动的距
离以使的在原始图像中每个像素都可以涉及到。但是可能真就存在冲突的情况或者我希望过滤器处理后图片的大小不发生改变，这个时候可以适当的在待过滤的图片边缘添加
0来实现。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/numerical_padding_strides.gif" width=50% height=50%/>

这些处理得到特征map一般还需要通过池化来进行采样，其中一个目的就是进一步减小图片大小。一般的池化方法就是最大池化。比如使用2*2大小的框格在特制map上每次
移动2格，将2*2在特征map范围上的最大的数据返回。这样处理以后的数据可能是作为下一个卷积层的输入也可能作为全连接层的输入。


