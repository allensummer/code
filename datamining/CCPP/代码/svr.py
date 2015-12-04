# -*- coding: utf-8 -*-

import xlrd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

rng = np.random.RandomState(0)

def getData():
    data = xlrd.open_workbook('/home/wq/code/datamining/CCPP/ccpp.xls')
    
    #读取数据，将xls中的数据读取出来
    sheet1 = data.sheet_by_name('Sheet1')
    #x = sheet1.col_values(0)
    #y = sheet1.col_values(4)
    #plt.plot(x, y, 'o')
    #sheet2 = data.sheet_by_name('Sheet2')
   # sheet3 = data.sheet_by_name('Sheet3')
    sheet4 = data.sheet_by_name('Sheet4')
   # sheet5 = data.sheet_by_name('Sheet5')
    
    #训练集
    AT_tran = sheet1.col_values(0)[1:1000]# + sheet2.col_values(0) + sheet3.col_values(0)  
    V_tran = sheet1.col_values(1)[1:1000] #+ sheet2.col_values(1) + sheet3.col_values(1) 
    AP_tran = sheet1.col_values(2)[1:1000] #+ sheet2.col_values(2) + sheet3.col_values(2) 
    RH_tran = sheet1.col_values(3)[1:1000] #+ sheet2.col_values(3) + sheet3.col_values(3)  
    PE_tran = sheet1.col_values(4)[1:1000] #+ sheet2.col_values(4) + sheet3.col_values(4) 
    
    #测试集
    AT_test = sheet4.col_values(0)[1:1000] #+ sheet5.col_values(0)
    V_test = sheet4.col_values(1)[1:1000] #+ sheet5.col_values(1)
    AP_test =  sheet4.col_values(2)[1:1000] #+ sheet5.col_values(2)
    RH_test = sheet4.col_values(3)[1:1000] #+ sheet5.col_values(3)
    PE_test = sheet4.col_values(4)[1:1000] #+  sheet5.col_values(4)
    
    tran = {'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran}
    test = {'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test}
    
    return tran, test


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
    V_tran = sheet1.col_values(1)+ sheet2.col_values(1) + sheet3.col_values(1) 
    AP_tran = sheet1.col_values(2)+ sheet2.col_values(2) + sheet3.col_values(2) 
    RH_tran = sheet1.col_values(3)+ sheet2.col_values(3) + sheet3.col_values(3)  
    PE_tran = sheet1.col_values(4) + sheet2.col_values(4) + sheet3.col_values(4) 
    
    #测试集
    AT_test = sheet4.col_values(0)+ sheet5.col_values(0)
    V_test = sheet4.col_values(1)+ sheet5.col_values(1)
    AP_test =  sheet4.col_values(2) + sheet5.col_values(2)
    RH_test = sheet4.col_values(3)+ sheet5.col_values(3)
    PE_test = sheet4.col_values(4)+  sheet5.col_values(4)
    
    tran = {'AT':AT_tran,'V':V_tran,'AP':AP_tran,'RH':RH_tran,'PE':PE_tran}
    test = {'AT':AT_test,'V':V_test,'AP':AP_test,'RH':RH_test,'PE':PE_test}
    
    return tran, test
    
def get_kernelRidge_error(tranData, testData, name,evaluation):  
    error = {}
    f = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
#    print name
    Y = tranData['PE']
    Y1 =testData['PE']
    for i in name:
        
            X = tranData[i]
            X1 = testData[i]

            f.fit(X, Y)
            
            error[i+"训练误差"]=evaluation(Y,f.predict(X))
            print i+"训练误差:", str(error[i+"训练误差"])
            error[i+"测试误差"]= evaluation(Y1, f.predict(X1))
            print i+"测试误差:", error[i+"测试误差"]
            
    return error
    
def get_svr_error(tranData, testData, name,evaluation):
    error = {}
    f = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
#    print name
    Y = tranData['PE']
    Y1 =testData['PE']
    for i in name:
        
            X = tranData[i]
            X1 = testData[i]

            f.fit(X, Y)
            
            error[i+"训练误差"]=evaluation(Y,f.predict(X))
            print i+"训练误差:", str(error[i+"训练误差"])
            error[i+"测试误差"]= evaluation(Y1, f.predict(X1))
            print i+"测试误差:", error[i+"测试误差"]
            
    return error

def getResultError(tran, test, get_error , errFunc):
        
    name1 = ['AT', 'V', 'AP', 'RH']
    tran1 = {'AT':[[x] for x in tran['AT']], 'V':[[x] for x in tran['V']], 'AP':[[x] for x in tran['AP']],'RH':[[x] for x in tran['RH']], 'PE':[x for x in tran['PE']]}
    test1 = {'AT':[[x] for x in test['AT']], 'V':[[x] for x in test['V']], 'AP':[[x] for x in test['AP']],'RH':[[x] for x in test['RH']], 'PE':[x for x in test['PE']]}
    error1 = get_error(tran1,  test1, name1,  errFunc)
    #clf = regFunc()
    #clf.fit(np.reshape( AT_tran, (len(AT_tran), 1)) ,np.reshape(PE_tran, (len(PE_tran), 1)) )
    
    #数据子集有两个属性
    tran2 = {'AT-V':[[x, y] for x, y in zip(tran['AT'], tran['V'])],'AT-AP':[[x, y] for x, y in zip(tran['AT'], tran['AP'])],  'AT-RH':[[x, y] for x, y in zip(tran['AT'], tran['RH'])], 'V-AP':[[x, y] for x, y in zip(tran['V'], tran['AP'])], 'V-RH':[[x, y] for x, y in zip(tran['V'], tran['AP'])] ,'PE':[x for x in tran['PE']]}
    test2 = {'AT-V': [[x, y] for x, y in zip(test['AT'],test['V'])],'AT-AP':[[x, y] for x, y in  zip(test['AT'], test['AP'])],  'AT-RH': [[x, y] for x, y in zip(test['AT'], test['RH'])], 'V-AP':[[x, y ] for x, y in zip(test['V'], test['AP'])], 'V-RH':[[x, y] for x, y in zip(test['V'], test['AP'])],'PE': [x for x in test['PE']]}
    
    name2 = [ 'AT-V', 'AT-AP', 'AT-RH', 'V-AP', 'V-RH']
    error2 = get_error(tran2,  test2, name2,errFunc)
    
    #数据子集有三个属性
    tran3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'] , tran['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(tran['AT'], tran['V'], tran['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(tran['V'], tran['AP'], tran['RH'])], 'PE':[x for x in tran['PE']]}
    test3 = {'AT-V-AP':[[x, y, z] for x, y, z in zip(test['AT'],test['V'] , test['AP'])],'AT-V-RH':[[x, y, z] for x, y, z in zip(test['AT'],test['V'], test['RH'])],  'V-AP-RH':[[x, y, z] for x, y, z in zip(test['V'], test['AP'], test['RH'])], 'PE':[x for x in test['PE']]}
    name3 = [ 'AT-V-AP', 'AT-V-RH', 'V-AP-RH' ]
    error3 = get_error(tran3,  test3, name3, errFunc)
    
    #数据子集有四个属性
    tran4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'], tran['V'] , tran['AP'], tran['RH'])], 'PE':[x for x in tran['PE']]}
    test4 = {'AT-V-AP-RH':[[x, y, z, w] for x, y, z, w in zip(tran['AT'],test['V'] , test['AP'], test['RH'])], 'PE':[x for x in test['PE']]}
    name4 = [ 'AT-V-AP-RH' ]
    error4 = get_error(tran4,  test4, name4,errFunc)

    finalError = {1:error1, 2:error2, 3:error3, 4:error4}
    return finalError
    
if __name__ == '__main__':
    
    tran, test = getData()
    
    print "================核脊回归================="
    print "××××××××××××××××平均绝对偏差*****************"
    linearAbsoluteError = getResultError(tran, test, get_kernelRidge_error, mean_absolute_error)
    print "******************均方差********************"
    linearSquareError = getResultError(tran, test,get_kernelRidge_error, mean_squared_error)
    print "================核脊回归================="
    
    print "================svr回归================="
    print "××××××××××××××××平均绝对偏差*****************"
    linearAbsoluteError = getResultError(tran, test, get_svr_error, mean_absolute_error)
    print "******************均方差********************"
    linearSquareError = getResultError(tran, test,get_svr_error, mean_squared_error)
    print "================svr回归=================" 
#==============================================================================
#     
#     # 拟合函数
#     svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                    "gamma": np.logspace(-2, 2, 5)})
#     
#     kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#                       param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                                   "gamma": np.logspace(-2, 2, 5)})
#     
#     svr.fit(X[:train_size], y[:train_size])
#     svr_fit = time.time() - t0
#     print("SVR complexity and bandwidth selected and model fitted in %.3f s"
#           % svr_fit)
#     
#     t0 = time.time()
#     kr.fit(X[:train_size], y[:train_size])
#     kr_fit = time.time() - t0
#     print("KRR complexity and bandwidth selected and model fitted in %.3f s"
#           % kr_fit)
#     
#     sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
#     print("Support vector ratio: %.3f" % sv_ratio)
#     
#     t0 = time.time()
#     y_svr = svr.predict(X_plot)
#     svr_predict = time.time() - t0
#     print("SVR prediction for %d inputs in %.3f s"
#           % (X_plot.shape[0], svr_predict))
#     
#     t0 = time.time()
#     y_kr = kr.predict(X_plot)
#     kr_predict = time.time() - t0
#     print("KRR prediction for %d inputs in %.3f s"
#           % (X_plot.shape[0], kr_predict))
#     
#     
#     #############################################################################
#     # look at the results
#     sv_ind = svr.best_estimator_.support_
#     plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors')
#     plt.scatter(X[:100], y[:100], c='k', label='data')
#     plt.hold('on')
#     plt.plot(X_plot, y_svr, c='r',
#              label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
#     plt.plot(X_plot, y_kr, c='g',
#              label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
#     plt.xlabel('data')
#     plt.ylabel('target')
#     plt.title('SVR versus Kernel Ridge')
#     plt.legend()
#     
#     # Visualize training and prediction time
#     plt.figure()
#     
#     # Generate sample data
#     X = 5 * rng.rand(10000, 1)
#     y = np.sin(X).ravel()
#     y[::5] += 3 * (0.5 - rng.rand(X.shape[0]/5))
#     sizes = np.logspace(1, 4, 7)
#     for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
#                                                gamma=10),
#                             "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
#         train_time = []
#         test_time = []
#         for train_test_size in sizes:
#             t0 = time.time()
#             estimator.fit(X[:train_test_size], y[:train_test_size])
#             train_time.append(time.time() - t0)
#     
#             t0 = time.time()
#             estimator.predict(X_plot[:1000])
#             test_time.append(time.time() - t0)
#     
#         plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
#                  label="%s (train)" % name)
#         plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
#                  label="%s (test)" % name)
#     
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("Train size")
#     plt.ylabel("Time (seconds)")
#     plt.title('Execution Time')
#     plt.legend(loc="best")
#     
#     # Visualize learning curves
#     plt.figure()
#     
#     svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
#     kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
#     train_sizes, train_scores_svr, test_scores_svr = \
#         learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
#                        scoring="mean_squared_error", cv=10)
#     train_sizes_abs, train_scores_kr, test_scores_kr = \
#         learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
#                        scoring="mean_squared_error", cv=10)
#     
#     plt.plot(train_sizes, test_scores_svr.mean(1), 'o-', color="r",
#              label="SVR")
#     plt.plot(train_sizes, test_scores_kr.mean(1), 'o-', color="g",
#==============================================================================
#==============================================================================
#              label="KRR")
#     plt.xlabel("Train size")
#     plt.ylabel("Mean Squared Error")
#     plt.title('Learning curves')
#     plt.legend(loc="best")
#     
#     plt.show()
#==============================================================================
