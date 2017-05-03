## theano 学习笔记

### theano 变量类型
theano是python的一个库，我们可以通过pip3 install theano 安装它。    
theano的变量类型包括
* **byte**: bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4, btensor5
* **16-bit integers**: wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4, wtensor5
* **32-bit integers**: iscalar, ivector, imatrix, irow, icol, itensor3, itensor4, itensor5
* **64-bit integers**: lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4, ltensor5
* **float**: fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4, ftensor5
* **double**: dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4, dtensor5
* **complex**: cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4, ctensor5

这里scalar代表标量，dscalar代表是double类型的标量，对应matrix是矩阵。
~~~ python
#导入theano
import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
~~~
在上面的代码中我们在theano中定义了'x'和'y'的变量。这里的'x'和'y'是可以被theano感知到的符号。接着我们定义了一个运算z=x+y，这里需要注意这个运算也
只是一个符号表示一种运算规则，这个时候的z中并不是就等于x+y。后面我们使用function定一个函数，它使用[x,y]表示输入的变量，用z表示因变量。当我们调用f
的时候返回的就是x和y相加以后的值，从这里看出来这个些运算和我们平时的python代码还是不一样的。

### theano共享变量
~~~ python
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state,state+inc)])
~~~
使用shared(0)会返回一个初始值为0的共享变量，共享变量可以在多个函数直接共享。这里的function带有一个updates参数，作用每次调用function后将state+inc
更新到state中去。可以通过get_value()和set_value()来获取和设置共享变量。

在function中常常还有一个参数叫givens，可能是这样使用givens={x:i}这样就是在运行中使用i来替换x。

### theano随机变量
theano获取随机变量的方式是通过引入shared_randomstreams，设定随机数的种子以及选择随机数的分布来生成随机数。
~~~ python
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomSteams(seed=234)
rv_n = srng.normal((2,2))
f = function([],rv_n)
~~~

### dimshuffle
dimshuffle用来调整张量的维度，因为我们在计算中有时候需要让两个张量维度能对应上才能进一步的进行加法减法等计算。
~~~ python
a.dimshuffle('x',0)
~~~
这里'x'表示增加的维度，0表示原来张量的0维。如果a是N那么经过dimshuffle计算后就是1*N.

### theano 图结构
在theano维护的变量和计算中都会整理成计算图然后theano会进行优化，之后进行计算。

~~~ python
import theano.tensor as T

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
~~~
对应计算图如下

<img src="https://github.com/MemoryCrash/MachineLearningPractice/blob/master/image/theanoapply.png" width=30% height=30%/>

这里的op引用的是我们定义的加法操作，输入数据引用的是x和y，x和y进一步引用了matrix类型，输出的数据z和apply是互相依赖的关系。这里的apply可以理解为一次具体的算法的实施或者是function函数的一次调用。
