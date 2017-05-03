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
~~~python
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
