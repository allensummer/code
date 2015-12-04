# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:02:30 2015

@author: wq
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:39:54 2015

@author: wq
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from sklearn import linear_model
import pandas as pd

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

#将训练集数据转换为数据框
tran = pd.DataFrame({'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran})
n_tran = tran.count()[1]   #数据个数

#讲测试集数据转换为数据框
test = pd.DataFrame({'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test})
n_test = tran.count()[1]   #数据个数tran,

name = ['AT','V', 'RH', 'PE']
#线性拟合
X = np.reshape(tran[name[1]], (n_tran, 1))
Y = np.reshape(tran['PE'], (n_tran, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)
a =  clf.coef_[0][0]
b = clf.intercept_[0]
textRegression = 'y = ' + str(round( a,2)) +"x + " + str(round( b,2))
plt.plot(X, Y, 'o', color = 'blue', mec = 'b',alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')
    
plt.text(mean(X),mean(Y),textRegression)
    
plt.savefig(i,dpi=75)

'''
X = np.reshape(tran[name[2]], (n_tran, 1))
Y = np.reshape(tran['PE'], (n_tran, 1))
clf = linear_model.LinearRegression()
clf.fit( X ,Y)
a =  clf.coef_[0][0]
b = clf.intercept_[0]
textRegression = 'y = ' + str(round( a,2)) +"x + " + str(round( b,2))
plt.plot(X, Y, 'o', color = 'blue', mec = 'b',alpha = 0.2, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')
    
plt.text(mean(X),mean(Y),textRegression)
    
plt.savefig(i,dpi=75)
'''