
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from pandas import Series,DataFrame
import lightgbm as lgb



train=pd.read_csv('newTrain.csv')
label=np.log(train['price_doc'])
train.drop(['id','price_doc'],axis=1,inplace=True)
train_lgb=lgb.Dataset(train,label)
test=pd.read_csv('newTest.csv')
test_id=test.id
test.drop(['id'],axis=1,inplace=True)


randomSeed=1
params={'objective':'regression',
       'metric':'rmse',
        'boosting':'gbdt',
        'learning_rate':0.01,
        'verbose':0,
        'bagging_seed':randomSeed,
        'max_bin':100,
        'max_depth':7,
        'bagging_freq':1,
        'bagging_fraction':0.95,
        'min_data_in_leaf':20,
        'num_leaves':2**5,
        'feature_fraction':0.7,
        'feature_fraction_seed':randomSeed}


train=pd.read_csv('newTrain.csv')
test=pd.read_csv('newTest.csv')
id_test=test.id
label=np.log(train.price_doc)
train.drop(['price_doc','id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)
train_lgb=lgb.Dataset(train,label)
#without new features: RMSE :0.247139  rounds:1500 LB:0.31339 0.30998
#with new features: RMSE: 0.246907 rounds:1500 LB:0.31382, 0.31063
cv = lgb.cv(params, train_lgb, num_boost_round=1500,nfold=5,early_stopping_rounds=50,show_stdv=False,verbose_eval=50,stratified=False)
model=lgb.train(params,train_lgb,num_boost_round=1500)
predictions=np.exp(model.predict(test))
print('lgb predictions:',predictions)
output=pd.DataFrame({'id':id_test,'price_doc':predictions})
output.to_csv('output_lgb.csv',index=False)
#output.to_csv('output)lgb_newFeatures.csv',index=False)



'''
train=pd.read_csv('x_pd.csv').values
test=pd.read_csv('t_pd.csv').values
label=pd.read_csv('y_pd.csv').values.flatten()
print(label)
print(train.shape)
print(label.shape)
data_light=lgb.Dataset(train,label)
model=lgb.train(params,data_light,num_boost_round=1500)
predictions=model.predict(test)
print(predictions)
'''
'''
#0.247165
#cv = lgb.cv(params, train_lgb, num_boost_round=1500,nfold=5,early_stopping_rounds=50,show_stdv=False,verbose_eval=50,stratified=False)
model=lgb.train(params,train_lgb,num_boost_round=1500)
prediction=np.exp(model.predict(test))
outcome=pd.DataFrame({'id':test_id,'price_doc':prediction})
outcome.to_csv('outcome.csv',index=False)
'''
#min_error=np.inf
#for item1 in grid_search_params['num_leaves']:
#    for item2 in grid_search_params['min_data_in_leaf']:
#        for item3 in grid_search_params['bagging_fraction']:
#                #print('Current parameters:{} {} {} {}'.format(best_params[0],best_params[1],best_params[2],best_params[3]))
#                params['num_leaves'],params['min_data_in_leaf'],params['bagging_fraction']=item1,item2,item3
#                cv=lgb.cv(params,train_lgb,1500,nfold=5,early_stopping_rounds=20,verbose_eval=50,show_stdv=False)
#                error=cv['rmse-mean'][-1]
#                print('Current parameters:{} {} {}, Error: {}'.format(item1, item2, item3,error))
#               if error<min_error:
#                    min_error=error
#                    best_params = (item1, item2,item3)
#print('Best params:{} {} {}, RMSE: {}'.format(best_params[0],best_params[1],best_params[2],min_error))


'''
def process(train, test):
    RS = 1
    np.random.seed(RS)
    ROUNDS = 1500  # 1300,1400 all works fine
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.01,  # small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 2 ** 5,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': RS,
        'feature_fraction': 0.7,
        'feature_fraction_seed': RS,
        'max_bin': 100,
        'max_depth': 7,
        'num_rounds': ROUNDS,
    }
    
    # Remove the bad prices as suggested by Radar
    train = train[(train.price_doc > 1e6) & (train.price_doc != 2e6) & (train.price_doc != 3e6)]
    train.loc[(train.product_type == 'Investment') & (train.build_year < 2000), 'price_doc'] *= 0.9
    train.loc[train.product_type != 'Investment', 'price_doc'] *= 0.969  # Louis/Andy's magic number
    test = pd.read_csv('test.csv', parse_dates=['timestamp'])

    id_test = test.id
    times = pd.concat([train.timestamp, test.timestamp])
    num_train = train.shape[0]
    y_train = train["price_doc"]
    print(y_train)
    train.drop(['price_doc'], inplace=True, axis=1)
    da = pd.concat([train, test])
    da['na_count'] = da.isnull().sum(axis=1)
    df_cat = None
    to_remove = []
    for c in da.columns:
        if da[c].dtype == 'object':
            oh = pd.get_dummies(da[c], prefix=c)
            if df_cat is None:
                df_cat = oh
            else:
                df_cat = pd.concat([df_cat, oh], axis=1)
            to_remove.append(c)
    da.drop(to_remove, inplace=True, axis=1)

    # Remove rare features,prevent overfitting
    to_remove = []
    if df_cat is not None:
        sums = df_cat.sum(axis=0)
        to_remove = sums[sums < 200].index.values
        df_cat = df_cat.loc[:, df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    x_train = da[:num_train].drop(['timestamp', 'id'], axis=1)
    #x_train.to_csv('train_lgb.csv',index=False)
    x_test = da[num_train:].drop(['timestamp', 'id'], axis=1)
    #x_test.to_csv('test_lgb.csv',index=False)
    # Log transformation, boxcox works better.
    y_train = np.log(y_train)
    #y_train.to_csv('label_lgb.csv',index=False,header=['price_doc'])
    train_lgb = lgb.Dataset(x_train, y_train)
    #0.248542
    cv = lgb.cv(params, train_lgb, num_boost_round=ROUNDS,nfold=5,early_stopping_rounds=50,show_stdv=False,verbose_eval=50,stratified=False )
    model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)
    predict = model.predict(x_test)
    predict = np.exp(predict)
    return predict, id_test
'''




