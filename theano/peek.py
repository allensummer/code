# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/wq/.spyder2/.temp.py
theano example:
 sneak peek
"""
import theano
from theano import tensor

#什么两个典型的浮点标量
a = tensor.dscalar()
b = tensor.dscalar()

#创建一个很简单的表达式
c = a + b

#表达式转换为一个可调用对象，以（A，B）
#将参数传入，计算c的值
f = theano.function([a, b], c)

#将a赋值1.5,b赋值2.5
assert 4.0 == f(1.5, 2.5)