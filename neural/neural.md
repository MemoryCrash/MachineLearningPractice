# 神经网络学习笔记

## 背景

想想人的大脑是个很神奇的东西，在不断的学习中会形成自己的逻辑判断能力可以找到同一类事物的共同特征并以此来进行简单和基础的判断或者分类。如果能有一种模型或者算法能模拟大脑那是不是同样可以通过学习来进行更复杂活着抽象的判断或者分类了。神经网络学习算法可以算是一个这方面的尝试(但是它的目的并不是取模仿人的大脑)。记得在[感知机](https://github.com/MemoryCrash/MachineLearningPractice/blob/master/perceptron/perceptron.md)这个学习笔记中，提到了，对于感知机这样的基础算法，有两个很巧的拓展方式去处理复杂的判断，一个是通过将高维数据映射到低维然后通过核函数去处理这个形成了SVM算法，一个是增加层数去实现复杂判断的处理这样形成了神经网络。注意这个笔记只是针对神经网络，并不是深度学习。

## 神经元模型
<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuron.png" width=50% height=50%/>

真实的神经元的样子比较接近上图。神经元接受传入的信号，当信号超过阀值神经元被激活。经过简单化处理我们可以将神经元模型简化为下图。其中w表示输入信号的权值，b代表阀值，最后利用激活函数得出输出y。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/perceptron.png"/>

简单讲我们希望通过0来表示神经元是抑制，使用1表示神经元是兴奋状态。这样看来似乎使用阶跃函数比较好。但是阶跃函数不连续，不利于后续的计算使用。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralStepFuction.jpg" width=30% height=30%/>

所以后面找到了sigmoid函数来作为激活函数，可以看出它在0附近快速的变化为0或者1并且也具有连续可导的特点。

<img src = "https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/sigmoid.png" width=30% height=30%/>

我们可以将神经元模型通过数学公式来表示：  

![pi](http://latex.codecogs.com/png.latex?\sigma&space;(z)=\frac{1}{1&plus;e^{-z}})  

这个公式其实就是sigmoid公式，但是光有这个还不够，我们还需要将公式中的z表示出来：

![pi](http://latex.codecogs.com/png.latex?\sum_{j}w_{j}x_{j}&plus;b)  

## 神经网络

我们建立的一个神经元模型。这个对这个模型比较好的直接应用就是[感知机](https://github.com/MemoryCrash/MachineLearningPractice/blob/master/perceptron/perceptron.md)了但是通过感知机的笔记可以了解到，感知机能做的事情是非常有限的。通过组合多个神经元去构建一个多层的神经网络我们可以处理更加复杂的分类或者判断问题。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/neuralNetwork.png" width=40% height=40%/>

## 参考数据  
[1]《机器学习》 周志华 著 
[2]《斯坦福大学公开课：机器学习课程 cs229 吴恩达 
[3]《coursera 机器学习课程》 吴恩达 
[4][《神经网络和深度学习》](http://www.tensorfly.cn/home/?p=80) 
[5][《反向传导算法》](http://deeplearning.stanford.edu/wiki/index.php/反向传导算法)   

