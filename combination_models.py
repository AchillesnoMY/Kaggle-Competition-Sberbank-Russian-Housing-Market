
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from pandas import Series,DataFrame
import lightgbm as lgb
import xgboost as xgb

train=pd.read_csv('newTrain.csv')
test=pd.read_csv('newTest.csv')


#print('test size:',np.shape(test))
#print(test.ix[:,['product_type_Investment','product_type_OwnerOccupier']])
id_test=test.id

#extract investment dataset
investment_index_train=train.loc[:,'product_type_Investment']==1
investment_index_test=test.loc[:,'product_type_Investment']==1
#print(investment_index_test.sum())
#print(investment_index)
train_investment=train.loc[investment_index_train,:]
test_investment=test.loc[investment_index_test,:]
#print(test_investment.loc[:,['product_type_Investment','product_type_OwnerOccupier']])
#print(np.shape(test_investment))
label_investment=np.log(train_investment['price_doc'])
train_investment=train_investment.drop(['price_doc','id'],axis=1,inplace=False)
test_investment=test_investment.drop(['id'],axis=1,inplace=False)

#extract owner_occupier dataset
owner_index_train=train.loc[:,'product_type_OwnerOccupier']==1
owner_index_test=test.loc[:,'product_type_OwnerOccupier']==1
#print(owner_index_test.sum())
train_owner=train.loc[owner_index_train,:]
test_owner=test.loc[owner_index_test,:]
#print(test_owner.loc[:,['product_type_Investment','product_type_OwnerOccupier']])
#print(np.shape(test_owner))
label_owner=np.log(train_owner['price_doc'])
train_owner=train_owner.drop(['price_doc','id'],axis=1,inplace=False)
test_owner=test_owner.drop(['id'],axis=1,inplace=False)
#print(np.shape(train_owner))

output_combination=np.zeros(test.shape[0])
investment_pred_index=test.loc[:,'product_type_Investment']==1
owner_pred_index=test.loc[:,'product_type_OwnerOccupier']==1
#print(np.shape(investment_pred_index))
#print(np.shape(owner_pred_index))
#print(investment_pred_index)
#print(owner_pred_index)

#xgboost to predict investment price
'''
params_investment={'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1}
train_d_investment=xgb.DMatrix(train_investment,label_investment)
#train-rmse:1.30709e+06	test-rmse:2.51824e+06
#cv=xgb.cv(params_investment,train_d_investment,num_boost_round=1000,early_stopping_rounds=20,verbose_eval=50,show_stdv=False,nfold=5,seed=1)
#best_iterations=len(cv)+1
#print(best_iterations)
model_investment=xgb.train(params_investment,train_d_investment,num_boost_round=206)
test_d_investment=xgb.DMatrix(test_investment)
predict_investment=np.exp(model_investment.predict(test_d_investment))


#print(np.shape(predict_investment))
output_combination[investment_pred_index]=predict_investment
print(output_combination)

#xgboost to predict owner price
params_owner={'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1}
train_d_owner=xgb.DMatrix(train_owner,label_owner)
#0.111941
#cv=xgb.cv(params_owner,train_d_owner,num_boost_round=1000,early_stopping_rounds=20,verbose_eval=50,show_stdv=False,nfold=5,seed=1)
#best_iterations=len(cv)+1
model_owner=xgb.train(params_owner,train_d_owner,num_boost_round=901)
test_d_owner=xgb.DMatrix(test_owner)
predict_owner=np.exp(model_owner.predict(test_d_owner))
'''

#lgb to predict investment price
params_lgb={'objective':'regression',
       'metric':'rmse',
        'boosting':'gbdt',
        'learning_rate':0.01,
        'verbose':0,
        'bagging_seed':1,
        'max_bin':100,
        'max_depth':7,
        'bagging_freq':1,
        'bagging_fraction':0.95,
        'min_data_in_leaf':20,
        'num_leaves':2**5,
        'feature_fraction':0.7,
        'feature_fraction_seed':1}

train_d_investment=lgb.Dataset(train_investment,label_investment)
model_investment=lgb.train(params_lgb,train_d_investment,num_boost_round=1500)
predict_investment=np.exp(model_investment.predict(test_investment))
output_combination[investment_pred_index]=predict_investment
print(output_combination)

train_d_owner=lgb.Dataset(train_owner,label_owner)
model_owner=lgb.train(params_lgb,train_d_owner,num_boost_round=1500)
predict_owner=np.exp(model_owner.predict(test_owner))
output_combination[owner_pred_index]=predict_owner
#print(predict_owner)

print('final output:',output_combination)
print((output_combination==0).sum())
print(np.shape(output_combination))
output_combination=pd.DataFrame({'id':id_test,'price_doc':output_combination})
output_combination.to_csv('output_combination.csv',index=False)



