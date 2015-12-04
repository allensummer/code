# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:13:49 2015

@author: wq
"""

#import numpy as np
#import matplotlib.pyplot as plt
import xlrd
from sklearn import linear_model
#import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#==============================================================================
# 
# def get_error(tranData, testData, name, func,evaluation):  
#     error = {}
#     
#     n_tran = len(tranData)
#     n_test = len(testData)
#     print name
#     Y1 = np.reshape(testData['PE'], (n_test, 1))
#     Y = np.reshape(tranData['PE'], (n_tran, 1))
#     for i in name:
#             X = np.reshape(tranData[i], (n_tran, 1))
#             X1 = np.reshape(testData[i],(n_test, 1))
# 
#             clf = func()
#             clf.fit( X ,Y)
#             
#             error[i+"绝对训练误差"]=mean_absolute_error(Y,clf.predict(X))
#             print i+"绝对训练误差:", str(error[i+"绝对训练误差"])
#             error[i+"绝对测试误差"]= mean_absolute_error(Y1, clf.predict(X1))
#             print i+"绝对测试误差:", error[i+"绝对测试误差"]

#       return error
#==============================================================================
            
def get_error(tranData, testData, name, func,evaluation, alpha):  
    error = {}
    clf = func(alpha)
#    print name
    Y = tranData['PE']
    Y1 =testData['PE']
    for i in name:
        
            X = tranData[i]
            X1 = testData[i]

            clf.fit( X ,Y)
            
            error[i+"训练误差"]=evaluation(Y,clf.predict(X))
            print i+"训练误差:", str(error[i+"训练误差"])
            error[i+"测试误差"]= evaluation(Y1, clf.predict(X1))
            print i+"测试误差:", error[i+"测试误差"]
            
    return error
       
def get_error2(tranData, testData, name, func,evaluation):  
    error = {}
    
#    print name
    Y = tranData['PE']
    Y1 =testData['PE']
    for i in name:
        
            X = tranData[i]
            X1 = testData[i]
            clf = func()
            clf.fit( X ,Y)
            print i, clf.coef_
            print i, clf.intercept_
            error[i+"训练误差"]=evaluation(Y,clf.predict(X))
            print i+"训练误差:", str(error[i+"训练误差"])
            error[i+"测试误差"]= evaluation(Y1, clf.predict(X1))
            print i+"测试误差:", error[i+"测试误差"]
            
    return error
        
def getData():
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
    AT_tran = sheet1.col_values(0) + sheet2.col_values(0) + sheet3.col_values(0)  
    V_tran = sheet1.col_values(1) + sheet2.col_values(1) + sheet3.col_values(1) 
    AP_tran = sheet1.col_values(2) + sheet2.col_values(2) + sheet3.col_values(2) 
    RH_tran = sheet1.col_values(3) + sheet2.col_values(3) + sheet3.col_values(3)  
    PE_tran = sheet1.col_values(4) + sheet2.col_values(4) + sheet3.col_values(4) 
    
    #测试集
    AT_test = sheet4.col_values(0) + sheet5.col_values(0)
    V_test = sheet4.col_values(1) + sheet5.col_values(1)
    AP_test =  sheet4.col_values(2) + sheet5.col_values(2)
    RH_test = sheet4.col_values(3) + sheet5.col_values(3)
    PE_test = sheet4.col_values(4) +  sheet5.col_values(4)
    
    tran = {'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran}
    test = {'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test}
    
    return tran, test
    
#==============================================================================
# #数据子集有一个属性
# #将训练集数据转换为数据框
# tran = pd.DataFrame({'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran})
# #将测试集数据转换为数据框
# test = pd.DataFrame({'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test})
#==============================================================================

#拟合函数有参数
def getResultError(tran, test, regFunc, errFunc, parameter):
        
    name1 = ['AT', 'V', 'AP', 'RH']
    tran1 = {'AT':[[x] for x in tran['AT']], 'V':[[x] for x in tran['V']], 'AP':[[x] for x in tran['AP']],'RH':[[x] for x in tran['RH']], 'PE':[[x] for x in tran['PE']]}
    test1 = {'AT':[[x] for x in test['AT']], 'V':[[x] for x in test['V']], 'AP':[[x] for x in test['AP']],'RH':[[x] for x in test['RH']], 'PE':[[x] for x in test['PE']]}
    error1 = get_error(tran1,  test1, name1, regFunc, errFunc, parameter)
    #clf = regFunc()
    #clf.fit(np.reshape( AT_tran, (len(AT_tran), 1)) ,np.reshape(PE_tran, (len(PE_tran), 1)) )
    
    #数据子集有两个属性
    tran2 = {'AT-V':[[x, y] for x, y in zip(tran['AT'], tran['V'])],'AT-AP':[[x, y] for x, y in zip(tran['AT'], tran['AP'])],  'AT-RH':[[x, y] for x, y in zip(tran['AT'], tran['RH'])], 'V-AP':[[x, y] for x, y in zip(tran['V'], tran['AP'])], 'V-RH':[[x, y] for x, y in zip(tran['V'], tran['AP'])] ,'PE':[[x] for x in tran['PE']]}
    test2 = {'AT-V': [[x, y] for x, y in zip(test['AT'],test['V'])],'AT-AP':[[x, y] for x, y in  zip(test['AT'], test['AP'])],  'AT-RH': [[x, y] for x, y in zip(test['AT'], test['RH'])], 'V-AP':[[x, y ] for x, y in zip(test['V'], test['AP'])], 'V-RH':[[x, y] for x, y in zip(test['V'], test['AP'])],'PE': [[x] for x in test['PE']]}
    
    name2 = [ 'AT-V', 'AT-AP', 'AT-RH', 'V-AP', 'V-RH']
    error2 = get_error(tran2,  test2, name2, regFunc, errFunc, parameter)
    
    #数据子集有三个属性
    tran3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'] , tran['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'], tran['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(tran['V'], tran['AP'], tran['RH'])], 'PE':[[x] for x in tran['PE']]}
    test3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(test['AT'],test['V'] , test['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(test['AT'],test['V'], test['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(test['V'], test['AP'], test['RH'])], 'PE':[[x] for x in test['PE']]}
    name3 = [ 'AT-V-AP', 'AT-V-RH', 'V-AP-RH' ]
    error3 = get_error(tran3,  test3, name3, regFunc, errFunc, parameter)
    
    #数据子集有四个属性
    tran4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'], tran['V'] , tran['AP'], tran['RH'])], 'PE':[[x] for x in tran['PE']]}
    test4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'],test['V'] , test['AP'], test['RH'])], 'PE':[[x] for x in test['PE']]}
    name4 = [ 'AT-V-AP-RH' ]
    error4 = get_error(tran4,  test4, name4, regFunc, errFunc, parameter)

    finalError = {1:error1, 2:error2, 3:error3, 4:error4}
    return finalError
    
def getResultError2(tran, test, regFunc, errFunc):
        
    name1 = ['AT', 'V', 'AP', 'RH']
    tran1 = {'AT':[[x] for x in tran['AT']], 'V':[[x] for x in tran['V']], 'AP':[[x] for x in tran['AP']],'RH':[[x] for x in tran['RH']], 'PE':[[x] for x in tran['PE']]}
    test1 = {'AT':[[x] for x in test['AT']], 'V':[[x] for x in test['V']], 'AP':[[x] for x in test['AP']],'RH':[[x] for x in test['RH']], 'PE':[[x] for x in test['PE']]}
    error1 = get_error2(tran1,  test1, name1, regFunc, errFunc)
    #clf = regFunc()
    #clf.fit(np.reshape( AT_tran, (len(AT_tran), 1)) ,np.reshape(PE_tran, (len(PE_tran), 1)) )
    
    #数据子集有两个属性
    tran2 = {'AT-V':[[x, y] for x, y in zip(tran['AT'], tran['V'])],'AT-AP':[[x, y] for x, y in zip(tran['AT'], tran['AP'])],  'AT-RH':[[x, y] for x, y in zip(tran['AT'], tran['RH'])], 'V-AP':[[x, y] for x, y in zip(tran['V'], tran['AP'])], 'V-RH':[[x, y] for x, y in zip(tran['V'], tran['AP'])] ,'PE':[[x] for x in tran['PE']]}
    test2 = {'AT-V': [[x, y] for x, y in zip(test['AT'],test['V'])],'AT-AP':[[x, y] for x, y in  zip(test['AT'], test['AP'])],  'AT-RH': [[x, y] for x, y in zip(test['AT'], test['RH'])], 'V-AP':[[x, y ] for x, y in zip(test['V'], test['AP'])], 'V-RH':[[x, y] for x, y in zip(test['V'], test['AP'])],'PE': [[x] for x in test['PE']]}
    
    name2 = [ 'AT-V', 'AT-AP', 'AT-RH', 'V-AP', 'V-RH']
    error2 = get_error2(tran2,  test2, name2, regFunc, errFunc)
    
    
    #××××××××××××××××××××××错误记录×××××××××××××××××××××××××××
    #测试输入数据的格式
    #这里我一开始使用的是数据框，但是使用np.reshape进行整形的时候，总是不成功。最后我就尝试自己把数据
    #组合成规定的格式，二维的格式为：[[1,2], [3,4], [5, 6]].用数据框图也能调成这个样子，可是总是报错。
    #AT-V = []
    #AT-V.append([x,y] for x,y in zip(AT_tran, tran['V']))
    #ATV = [[x, y] for x, y in zip(tran['AT'], tran['V'])]
    #PE = [[x] for x in tran['PE']]
    #clf = regFunc()
    #clf.fit( ATV, PE)
    #××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    
    #数据子集有三个属性
    tran3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'] , tran['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'], tran['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(tran['V'], tran['AP'], tran['RH'])], 'PE':[[x] for x in tran['PE']]}
    test3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(test['AT'],test['V'] , test['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(test['AT'],test['V'], test['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(test['V'], test['AP'], test['RH'])], 'PE':[[x] for x in test['PE']]}
    name3 = [ 'AT-V-AP', 'AT-V-RH', 'V-AP-RH' ]
    error3 = get_error2(tran3,  test3, name3, regFunc, errFunc)
    
    #数据子集有四个属性
    tran4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'], tran['V'] , tran['AP'], tran['RH'])], 'PE':[[x] for x in tran['PE']]}
    test4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'],test['V'] , test['AP'], test['RH'])], 'PE':[[x] for x in test['PE']]}
    name4 = [ 'AT-V-AP-RH' ]
    error4 = get_error2(tran4,  test4, name4, regFunc, errFunc)

    finalError = {1:error1, 2:error2, 3:error3, 4:error4}
    return finalError

if __name__ == '__main__':
        
    tran, test = getData()
    
    print "================最小二乘发回归================="
    print "××××××××××××××××平均绝对偏差*****************"
    linearAbsoluteError = getResultError2(tran, test, linear_model.LinearRegression , mean_absolute_error)
    print "******************均方差********************"
    linearSquareError = getResultError2(tran, test, linear_model.LinearRegression , mean_squared_error)
    print "================最小二乘发回归================="
    
    print "================Lasso回归================="
    print "××××××××××××××××平均绝对偏差*****************"
    linearAbsoluteError = getResultError(tran, test, linear_model.Lasso , mean_absolute_error, 0.1)
    print "******************均方差********************"
    linearSquareError = getResultError(tran, test, linear_model.Lasso  , mean_squared_error, 0.1)
    print "================Lasso回归================="
    
    print "================脊回归================="
    print "××××××××××××××××平均绝对偏差*****************"
    linearAbsoluteError = getResultError2(tran, test, linear_model.Ridge , mean_absolute_error)
    print "******************均方差********************"
    linearSquareError = getResultError2(tran, test, linear_model.Ridge , mean_squared_error)
    print "================脊回归================="
#==============================================================================
    
#测试函数使用
'''a = [1, 2, 3]
b = [2, 3, 4]
A = a #sheet1.col_values(0)
B = b #sheet1.col_values(4)
clf.fit(np.reshape( A, (len(A), 1)) ,np.reshape(B, (len(B), 1)) )

X = np.reshape( A, (len(A), 1))
Y = np.reshape(B, (len(B), 1)) 
plt.plot(X, Y, 'o', color = 'blue', mec = 'b',alpha = 1, markersize = 2)
plt.plot(X,clf.predict(X), color = 'red')
    
plt.show()
'''
'''for i in name:

    clf = linear_model.LinearRegression()
    clf.fit( X ,Y)
    error[i+"绝对训练误差"]=mean_absolute_error(Y,clf.predict(X))
    print i+"绝对训练误差:", str(error[i+"绝对训练误差"])
    error[i+"绝对误差"]= mean_absolute_error(Y1, clf.predict(X1))
    print i+"绝对测试误差:", error[i+"绝对测试误差"]
   ''' 
