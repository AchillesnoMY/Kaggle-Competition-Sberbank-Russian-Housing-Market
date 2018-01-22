import sys
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from pandas import Series,DataFrame
import lightgbm as lgm
import xgboost as xgb
from sklearn.model_selection import train_test_split

train= pd.read_csv('newTrain.csv')
#train_new=pd.read_csv('train_lessFeatures.csv')
test = pd.read_csv('newTest.csv')
#test=pd.read_csv('test_lessFeatures.csv')
train_y_log= np.log(train['price_doc'])
train_x = train.drop(['id', 'price_doc'], axis=1)
#train_x=train_new

xgb_param = {'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1}
# cross_validation_evaluation(train_x,train_y,xgb_param)
dtrain = xgb.DMatrix(train_x, train_y_log)
#with no new features: RMSE:0.250707, LB:0.31658 0.31457, rounds=398
#with new features: RMSE 0.250122, LB: 0.31578 0.31252, rounds=400
#with removing insignificant features: RMSE:0.25049 round=370 LB:0.31460 0.31240

#find best number_iterations
cv=xgb.cv(xgb_param,dtrain,num_boost_round=1000,early_stopping_rounds=20,verbose_eval=50,show_stdv=False,nfold=5,seed=1)
num_boost_rounds=len(cv)+1
print(num_boost_rounds)

#train the model
model=xgb.train(dict(xgb_param,silent=0),dtrain,num_boost_round=num_boost_rounds)
id_test=test.id
test=test.drop(['id'],axis=1)
test_d=xgb.DMatrix(test,feature_names=test.columns.values)
#predict
predictions=np.exp(model.predict(test_d))
output=pd.DataFrame({'id':id_test,'price_doc':predictions})
#output.to_csv('output_xgb.csv',index=False)
output.to_csv('output_xgb_newFeatures.csv',index=False)
print('xgboost predictions',predictions)


# use local cross validation to select best parameters
def parameterSelection(train_x, train_y):


   # if  __name__ == '__main__':

    parameter_test1={'max_depth': [3,4,5],
                     'min_child_weight':[1,2,3]}
    #gsearch1=GridSearchCV(estimator=XGBRegressor(learning_rate=0.1,n_estimators=200,objective='reg:linear',max_depth=5,subsample=0.8,
    #                                            seed=27,colsample_bytree=0.8,gamma=0.1),scoring=rmse,param_grid=parameter_test1,iid=False,cv=5,verbose=1,n_jobs=1)
    #gsearch1.fit(train_x,train_y)
    #print(gsearch1.grid_scores_)
    #print(gsearch1.best_params_,gsearch1.best_score_) #colsample_bytree:0.8 max_depth=5

    # Split the data set with 30 percent of test dataset
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    xgb_param = {'eta': 0.05,
                 'max_depth': 8,
                 'gamma': 0.1,
                 'subsample': 0.8,
                 'colsample_bytree': 0.8,
                 'objective': 'reg:linear',
                 'eval_metric': 'rmse',
                 'seed': 27,
                 'silent': 1,
                 'min_child_weight': 1
                 }

    num_boost_round = 1000
    # Find the number of boost_rounds
    # model=xgb.train(xgb_param,dtrain,num_boost_round=num_boost_round,evals=[(dtest,'Test')],early_stopping_rounds=10) #RMSE:2.56209e+06, 135 rounds
    # cv=xgb.cv(xgb_param,dtrain,seed=27,nfold=5,metrics={'rmse'},early_stopping_rounds=10)# test-rmse-mean:5749561.5    9rounds
    # print(cv)
    # print(cv['test-rmse-mean'].min())

    '''First tunning max_depth and min_child_weight. Those parameters can be used to control the complexity of the trees. It is important to tune
    them together in order to find a good tradeoff between model bias and variance'''

    gridseach_params = {'max_depth': [5, 6, 7, 8],
                        'min_child_weight': [1, 2, 3, 4]}
    min_rmse = float(np.inf)
    best_params = None
    for max_depth in gridseach_params['max_depth']:
        for min_child_weight in gridseach_params['min_child_weight']:
            print('CV with max_depth={}, min_child_weight={}'.format(max_depth, min_child_weight))
            xgb_param['max_depth'] = max_depth
            xgb_param['min_child_weight'] = min_child_weight

            cv_results = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, seed=27, nfold=5, metrics={'rmse'},
                                early_stopping_rounds=10)
            mean_rmse = cv_results['test-rmse-mean'].min()
            boost_rounds = cv_results['test-rmse-mean'].argmin()
            print('\tRMSE {} for {} rounds'.format(mean_rmse, boost_rounds))
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = (max_depth, min_child_weight)
    print('Best params: {} {}, RMSE: {}'.format(best_params[0], best_params[1], min_rmse))  # 5,2, RMSE:2675685.65

    xgb_param['max_depth'] = 5
    xgb_param['min_child_weight'] = 2

    '''Next, we tunne subsample and colsample_bytree
    '''

    gridsearch_params = {'subsample': [0.6, 0.7, 0.8],
                         'colsample_bytree': [0.6, 0.7, 0.8]}

    min_rmse = float(np.inf)
    best_params = None
    for subsample in gridsearch_params['subsample']:
        for colsample in gridsearch_params['colsample_bytree']:
            xgb_param['subsample'] = subsample
            xgb_param['colsample_bytree'] = colsample
            cv_results = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, seed=27, nfold=5, metrics={'rmse'},
                                early_stopping_rounds=10)
            mean_rmse = cv_results['test-rmse-mean'].min()
            boost_rounds = cv_results['test-rmse-mean'].argmin()
            print('\tRMSE {} for {} rounds'.format(mean_rmse, boost_rounds))
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = (subsample, colsample)
    print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))  # 0.8,0.8 RMSE:2675685.65

    '''Next, we tune the eta or learning rate'''

    min_rmse = float(np.inf)
    best_params = None
    num_boost_round = 1500
    for eta in [0.2, 0.1, 0.05, 0.02, 0.01]:
        xgb_param['eta'] = eta
        cv_results = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, seed=27, nfold=5, metrics={'rmse'},
                            early_stopping_rounds=10)
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print('\tRMSE {} for {} rounds'.format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta
    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))  # 0.01, RMSE=2674069.95

    num_boost_round = 2000
    xgb_param['eta'] = 0.05  # The best 'eta' is 0.01, since 0.01 takes long time, so we change back to 0.05
    # xgb.train(xgb_param,dtrain,num_boost_round=num_boost_round,evals=[(dtest,'Test')],early_stopping_rounds=10)#num_rounds=1055, Test_RMSE=2524740
    model = xgb.train(xgb_param, dtrain, num_boost_round=num_boost_round, evals=[(dtest, 'Test')],
                      early_stopping_rounds=10)
    best_num_rounds = model.best_iteration + 1
    best_model = xgb.train(xgb_param, dtrain, num_boost_round=best_num_rounds, evals=[(dtest, 'Test')],
                           early_stopping_rounds=10)
    return best_model


# best_model=parameterSelection(train_x, train_y)


'''Model evaluation'''
'''
#method1
rmse=make_scorer(RMSE,greater_is_better=False)
improved_xgb=XGBRegressor(max_depth=5,min_child_weight=2,learning_rate=0.05,n_estimators=314,objective='reg:linear',seed=27,subsample=0.8,gamma=0.1,
                        eval_metric='rmse',silent=1,colsample_bytree=0.8)
naive_xgb=XGBRegressor(max_depth=8,learning_rate=0.05,n_estimators=209,objective='reg:linear',seed=27,subsample=0.7,
                        eval_metric='rmse',silent=1,colsample_bytree=0.7)
results_naive=cross_val_score(estimator=naive_xgb,X=train_x,y=train_y,cv=5,scoring=rmse)
results_improved=cross_val_score(estimator=improved_xgb,X=train_x,y=train_y,cv=5,scoring=rmse)
print('The mean RMSE for improved XGB is ',results_improved.mean()) #-2729713.72595
print('The mean RMSE for naive XGB is ',results_naive.mean())#-2714936.77238
#method2
params_naive={'eta':0.05,
           'max_depth':8,
           'subsample':0.7,
           'colsample_bytree':0.7,
           'objective': 'reg:linear',
           'eval_metric': 'rmse',
           'seed':27,
           'silent': 1}
dtrain=xgb.DMatrix(train_x,train_y)
results_naive=xgb.cv(params_naive,dtrain,num_boost_round=1000,early_stopping_rounds=10,verbose_eval=50,show_stdv=False,nfold=5)
print(results_naive)
params_improved={'eta':0.05,
           'max_depth':5,
           'gamma':0.1,
           'subsample':0.8,
           'colsample_bytree':0.8,
           'objective': 'reg:linear',
           'eval_metric': 'rmse',
           'seed':27,
           'silent': 1,
           'min_child_weight':0.2}
results_improved=xgb.cv(params_improved,dtrain,num_boost_round=1000,early_stopping_rounds=10,verbose_eval=50,show_stdv=False,nfold=5)
print(results_improved)
'''


