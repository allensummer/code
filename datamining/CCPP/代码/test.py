# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:39:54 2015

@author: wq
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import math
from sklearn import linear_model
data = xlrd.open_workbook('/home/wq/code/datamining/CCPP/ccpp.xls')

#读取数据，将xls中的数据读取出来
sheet1 = data.sheet_by_name('Sheet1')
#x = sheet1.col_values(0)
#y = sheet1.col_values(4)
#plt.plot(x, y, 'o')

sheet2 = data.sheet_by_name('Sheet2')
sheet3 = data.sheet_by_name('Sheet3')
sheet4 = data.sheet_by_name('Sheet4')
sheet5 = data.sheet_by_name('Sheet5')

#训练集
AT_tran = sheet1.col_values(0) + sheet2.col_values(0) + sheet3.col_values(0) + sheet4.col_values(0)
V_tran = sheet1.col_values(1) + sheet2.col_values(1) + sheet3.col_values(1) + sheet4.col_values(1)
AP_tran = sheet1.col_values(2) + sheet2.col_values(2) + sheet3.col_values(2) + sheet4.col_values(2)
RH_tran = sheet1.col_values(3) + sheet2.col_values(3) + sheet3.col_values(3) + sheet4.col_values(3)
PE_tran = sheet1.col_values(4) + sheet2.col_values(4) + sheet3.col_values(4) + sheet4.col_values(4)

#测试集
AT_test = sheet5.col_values(0)
V_test = sheet5.col_values(1)
AP_test = sheet5.col_values(2)
RH_test = sheet5.col_values(3)
PE_test = sheet5.col_values(4)


n = len(AT_tran)    #数据个数
N = len(sheet1.col_values(1))

datas = {'AT':AT_tran, 'V':V_tran, 'AP':AP_tran, 'RH':RH_tran, 'PE':PE_tran}
#散点矩阵图

data = {0:AT_tran,1:V_tran, 2:AP_tran, 3:RH_tran, 4:PE_tran }
fig, axes = plt.subplots(5,5, subplot_kw = {'xticks':[], 'yticks':[]})
title = [ 'AT', 'V', 'AP', 'RH', 'PE']
i = 0;


#X = np.random.normal(0,1,n)
#Y = np.random.normal(0,1,n)
#T = np.arctan2(Y,X)

for ax in axes.flat:
    k = int(i/5)
    j = 4 - ( i % 5)
    if k == 0:
        ax.set_title(title[j])
    if j == 4:
        ax.set_ylabel(title[k])
    ax.plot(sheet1.col_values(k), sheet1.col_values(j), 'o', alpha = 0.3, markersize = 2)
   
    #ax.plot(data[k], data[j], 'o', alpha = 0.015, markersize = 2)
    i += 1


#AT-PE散点图

#X = np.reshape(sheet1.col_values(0),(N, 1)) 
#Y = np.reshape( sheet1.col_values(4), (N, 1))
X = np.reshape(AT_tran, (n, 1))
Y = np.reshape(PE_tran, (n, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)

plt.plot(X, Y, 'o', color = 'blue', alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')
print 'AT-PE'
print clf.coef_, clf.intercept_

plt.figure()

#V-PE散点图

X = np.reshape(sheet1.col_values(1),(N, 1)) 
Y = np.reshape( sheet1.col_values(4), (N, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)
print 'V-PE'
print clf.coef_, clf.intercept_

plt.plot(X, Y, 'o', color = 'blue', alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')

plt.figure()

#AP-PE散点图

X = np.reshape(sheet1.col_values(2),(N, 1)) 
Y = np.reshape( sheet1.col_values(4), (N, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)
print 'AP-PE'
print clf.coef_, clf.intercept_

plt.plot(X, Y, 'o', color = 'blue', alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')

plt.figure()

#RH-PE散点图

X = np.reshape(sheet1.col_values(3),(N, 1)) 
Y = np.reshape( sheet1.col_values(4), (N, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)
print 'RH-PE'
print clf.coef_, clf.intercept_

plt.plot(X, Y, 'o', color = 'blue', alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')

plt.figure()


X = np.reshape(AT_tran, (n, 1))
Y = np.reshape(PE_tran, (n, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)

plt.plot(X, Y, 'o', color = 'blue', alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')

plt.show()