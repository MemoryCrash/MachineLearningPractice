#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import pickle
import gzip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool


def linear(z):
    return z

# 定义一个relu函数，可以作为新的激活函数来缓解梯度计算缓慢的问题
def ReLU(z):
    return T.maximun(0.0, z)


from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#GPU = True
GPU = False

if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify GPU flag to False.")
    try:
        theano.config.device = 'gpu'
    except:
        pass
    theano.config.floatX = 'float32'

else:
    print('Running with a CPU. If this is not desired,then the modify GPU flag to True.')


# 加载mnist数据
def load_data_shared(filename="../neural/dataSource/mnist.pkl.gz"):
    # borrow=Ture表示引用
    # 使用with打开文件可以更好处理文件的打开和关闭
    with gzip.open(filename, 'rb') as f:
        # 在python3中可以直接使用pickle，同时使用encoding主要是为了避免可能存在的编码问题
        trainning_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        def shared(data):
            shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
            shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
            # 将图片数据读入后放到shared里面去，同时将y的数据类型转换为int32
            return shared_x, T.cast(shared_y, "int32")
        return [shared(trainning_data), shared(validation_data), shared(test_data)]


class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        # 初始化神经网络中的第一个层
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # 初始化神经网络中非第一和最后一个的层
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            # 本层的输入数据是上一层的输出。这里的数据有两种一种是output或者使用了dropout对数据稀疏处理的输出
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        # 初始化神经网络中的最后一层
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    # 随机梯度下降实现 eta 用来作为学习速率，lmbda 用来作为正则化中使用
    def SGD(self, trainning_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        trainning_x, trainning_y = trainning_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # 计算在使用梯度下降的时候，根据总的数据和一次下降需要的数据，计算一个epochs梯度下降的次数
        num_training_batches = int(size(trainning_data) / mini_batch_size)
        num_validation_batches = int(size(validation_data) / mini_batch_size)
        num_test_batches = int(size(test_data) / mini_batch_size)

        # 使用l2范数根据w来生成一个正则项
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        # 代价函数
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        # 求代价函数对params的导数
        grads = T.grad(cost, self.params)
        # 使用param-eta*grad来更新param，执行梯度的更新
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        #定义一个64位的变量i
        i = T.lscalar()
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
            # 取一个索引下的练习数据和练习数据的标签对x和y的值进行更新,在givens中x作为一个字典的键，冒号后是值
            # 当在function被调用的时候，存在键的地方会被对应替换为值。通过i来根据批次来更新x和y在cpu或者gpu中
            # 的值，这样可以避免通过传值的形式在cpu和gpu之间进行数据拷贝的操作
            self.x : trainning_x[i * self.mini_batch_size : (i + 1) * self.mini_batch_size],
            self.y : trainning_y[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            })
        # 计算验证数据测得的正确率 这里的self.y会在函数调用的时候被替换为值
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
            self.x : validation_x[i * self.mini_batch_size : (i + 1) * self.mini_batch_size],
            self.y : validation_y[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            })
        # 计算测试数据测得的正确率
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
            self.x : test_x[i * self.mini_batch_size : (i + 1) * self.mini_batch_size],
            self.y : test_y[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            })
        # 获得最后层数据的模型输出
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
            self.x : test_x[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            })

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if(iteration + 1) % num_training_batches == 0:
                    # 计算验证集合得到的正确率以便分析模型训练情况
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    # 获取最好的验证率
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        # 验证集跑完了以后使用测试集再跑一次做出结果记录
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                            print("The corresponding test accuracy is {0:.2%}".format(test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class ConvPoolLayer(object):


    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        # prod 求对应元素的积，filter_shape[0]是过滤层的个数，filter_shape[2:]每个过滤层的大小
        # np.prod(poolsize)池化大小
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        # 针对每个过滤层的每个位置生成w的正态分布随机数，这里通过除w的个数来所有方差，使的正态分布更尖锐。
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape), dtype=theano.config.floatX),
            borrow=True)
        # 针对每个滤层生成正态分布的随机数字来初始化b
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX),
            borrow=True)

        self.params = [self.w, self.b]


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # self.image_shape 的参数分别是 batch size,input channels,input rows,input columns
        self.inpt = inpt.reshape(self.image_shape)
        # 传入整形后的输入数据，以及对应w..等数据
        conv_out = conv.conv2d(
            input = self.inpt, filters=self.w, filter_shape=self.filter_shape, image_shape=self.image_shape)
        # 在ignore_border=True时才能进行pading
        pooled_out = pool.pool_2d(input=conv_out, ws=self.poolsize, ignore_border=True)
        # dimshuffle对数组的某一行取出来拼接成新的数组，只有shared才能调用这个方法‘x’表示增加一维
        # dimshuffle('x', 0, 'x', 'x')，举个例子对于a=[1 2 3]如果a.dimshuffle['x' 0]就是[[1 2 3]]
        # 如果a.dimshuffle[0 'x']就是
        # [[1]
        #  [2]
        #  [3]]
        # 调整维度以后可以与其它维度的张量相加。
        # 注意pooled_out是(batch_size,num_filter,out_width,out_height),b是num_filter的向量。我们需要
        # 通过broadcasting让所有的pooled_out都加上一个bias,所以我们需要用dimshuffle函数把b变成(1,num_filter,1,1)
        # 的tensor。dimshuffle的参数'x'表示增加一个维度。0表示原来这个tensor的第0维。
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output


class FullyConnectedLayer(object):


    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # 初始化全连接层的w，size(n_in,n_out)
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)), 
            dtype=theano.config.floatX),
            name='w', borrow=True
            )
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
            dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # reshape的作用是将conv层输出的(batch_size,num_filter,width,height)转换成(batch_size,num_filter*width*height)
        # 在构造全连接层的时候我们也指定来n_in=num_filter*width*height这样就对应上了
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        # 激活函数得到值需要乘1 - self.p_dropout这样得到的是真的输出
        self.output = self.activation_fn((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        # 生成一个dropout后的神经网络的层
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout, self.w) + self.b)

    # 计算正确率
    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # softmaxlayer的层可以使用0来初始化
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        # axis=1是对行操作，argmax获取最大值的位置信息并返回。因为我们一次输入了多个训练样本，每个样本都会
        # 得到一个关于10个分类的输出的概率信息向量，我们需要的是针对每个样本的输出获取到分类概率最大的一个
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    # 此处计算输出的代价，一个批次有n个样本，每个输入的样本会在10个分类中作出决策，我们就根据输入的样本的y来查看分类以后此
    # 位置是否符合预期，对应到代价函数就是先求log再加起来并求平均值。
    def cost(self, net):
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    # 计算正确率
    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


def size(data):
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    # randomstate是伪随机数的种子生成小于999999的数
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    # binomial 是二项式分布（多次抛硬币的结果）
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    # cast是进行数据类型转换
    return layer * T.cast(mask, theano.config.floatX)


if __name__ == '__main__':

    trainning_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10

    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    net.SGD(trainning_data, 10, mini_batch_size, 0.1, validation_data, test_data)




