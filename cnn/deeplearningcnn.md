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

### 代价函数
在先前使用sigmod作为激活函数的神经网络中我们使用代价函数的模式一般是平方差的形式。代价函数的形式是会依赖对激活函数求导，我们知道求导对应到曲线可以理解
为曲线在某点的变化率。以下通过一个一个输入一个输出的单个神经元模型来进行解释。以下输入假设x是1，期待y输出为0可以看出求导的结果确实依赖了对sigmod的求
导，这样就会出现上面讨论的梯度下降缓慢的问题。

![p](http://latex.codecogs.com/gif.latex?C=\frac{(y-a)^{2}}{2})

![p](http://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;w}=(a-y){\sigma&space;}'(z)x=a{\sigma&space;}'(z))

![p](http://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;b}=(a-y){\sigma&space;}'(z)=a{\sigma&space;}'(z))

通过使用交叉熵来作为代价函数：

![pi](http://latex.codecogs.com/gif.latex?C=-\frac{1}{n}\sum_{x}[ylna&plus;(1-y)ln(1-a)])

可以对w和b分别求导发现结果是和激活函数的导数没有关系的。
