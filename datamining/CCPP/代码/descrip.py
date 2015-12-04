# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:50:42 2015

@author: wq
"""

import xlrd
import pandas as pd

def getData():
    data = xlrd.open_workbook('/home/wq/code/datamining/CCPP/ccpp.xls')
    
    sheet1 = data.sheet_by_name('Sheet1')
    sheet2 = data.sheet_by_name('Sheet2')
    sheet3 = data.sheet_by_name('Sheet3')
    sheet4 = data.sheet_by_name('Sheet4')
    sheet5 = data.sheet_by_name('Sheet5')
    
    AT_tran = sheet1.col_values(0) + sheet2.col_values(0) + sheet3.col_values(0)  + sheet4.col_values(0) + sheet5.col_values(0)
    V_tran = sheet1.col_values(1) + sheet2.col_values(1) + sheet3.col_values(1) + sheet4.col_values(1) + sheet5.col_values(1)
    AP_tran = sheet1.col_values(2) + sheet2.col_values(2) + sheet3.col_values(2) + sheet4.col_values(2) + sheet5.col_values(2)
    RH_tran = sheet1.col_values(3) + sheet2.col_values(3) + sheet3.col_values(3)  + sheet4.col_values(3) + sheet5.col_values(3)
    PE_tran = sheet1.col_values(4) + sheet2.col_values(4) + sheet3.col_values(4)  + sheet4.col_values(4) +  sheet5.col_values(4)
    
#==============================================================================
#     AT_test = sheet4.col_values(0) + sheet5.col_values(0)
#     V_test = sheet4.col_values(1) + sheet5.col_values(1)
#     AP_test =  sheet4.col_values(2) + sheet5.col_values(2)
#     RH_test = sheet4.col_values(3) + sheet5.col_values(3)
#     PE_test = sheet4.col_values(4) +  sheet5.col_values(4)
#==============================================================================
    
    tran = {'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran}
#==============================================================================
#     test = {'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test}
#==============================================================================
    
    return tran#, test

tran  = getData()

df2 = pd.DataFrame({'AT':tran['AT'], 'V':tran['V'], 'AP':tran['AP'], 'RH':tran['RH'], 'PE':tran['PE']})
print "前五个数据"
print df2.head()

print "简单统计描述"
print df2.describe()

print "协方差矩阵"
print df2.cov()

print "相关系数矩阵"
print df2.corr()