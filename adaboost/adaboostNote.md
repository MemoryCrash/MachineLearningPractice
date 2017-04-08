# 集成学习学习笔记

## 背景

在我们决策一个事情的时候，特别是比较重要的事情时，一般来讲我们会去征求下多个人的意见然后综合各方意见再做出决定。本次讨论的集成学习中的adaBoost也是基于这样的考虑。我们目前的讨论是针对使用相同的“基础分类器”来进行分类的二分类情况。这种综合其它的情况将他们的输出作为一个最终决策的输出，给人的感觉和神经网络有点像。

## adaBoost

在学习的方法中有有些学习方法称为强可学习有些称为弱可学习，从名字可以看出来有些学习方法效果并不太好但是弱学习方法一般是比较容易实现的。这个时候其实是有方法将弱学习提升(boost)成为强学习，就像那句话一样“三个臭皮匠顶个诸葛亮”我们需要的就是找到方法让他们能顶上。adaBoost全称是adaptive boosting(自适应提升)，它就是这样的方法。

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/adaBoost.png" width=50% height=50%/>

针对同样的一组训练样本，我们使用使弱学习方法去反复学习它然后每次学习都得到一个模型出来这样我们就拥有了多个决策者了。但是这个时候你也会发现如果只是这样的话那么为每次都是针对同样的训练数据去学习这样得到的训练模型也是大致相同的，并起不到集思广益的特点。所以我们不仅要得到多个模型还需要他们都有自己的想法。

这个时候adaBoost是这样去处理的。对于每次的学习得到的模型它实际的效果必然是有些分类正确有些分类失败。我们就把它分类失败的训练样本给一个更高的权值这样在下次学习的时候这部分分类失败的数据就会被特别关注。按照这样的情况我们循环下去直到分类好。然后我们就得到了多个决策者，考虑到有些决策者分类正确的情况比较多有些分类正确的情况比较少，他们的决策信息的可信度是不一致的。所以我们根据他们分类的正确情况给这些决策者赋予不同的权值，这里的权值判断并不是去选择权值最大的，是大家投票后统计票数来决定最终的分类结果。这就是adaBoost的基本思想。



## 参考书籍

《机器学习实战》 Peter Harrington 著 李锐 译    
《统计学习方法》 李航 著   
《机器学习》 周志华 著   