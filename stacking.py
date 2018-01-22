import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import lightgbm as lgb
from mlens.visualization import corrmat
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

class XGboost(object):
    def __init__(self,params):
        self.params=params

    def fit(self,train,label):
        self.train=train
        self.label=label
        data_train=xgb.DMatrix(train,label)
        cv=xgb.cv(self.params,data_train,num_boost_round=1000,early_stopping_rounds=20,verbose_eval=50,show_stdv=False,nfold=5,seed=1)
        best_iterations=len(cv)+1
        model=xgb.train(self.params,data_train,num_boost_round=best_iterations)
        self.model=model

    def predict(self,test):
        test_D=xgb.DMatrix(test)
        return self.model.predict(test_D)

class Lightgbm(object):
    def __init__(self,params):
        self.params=params

    def fit(self,train,label):
        self.train=train
        self.label=label
        data_train=lgb.Dataset(train,label)
        model=lgb.train(self.params,data_train,num_boost_round=1500)
        self.model=model

    def predict(self,test):
        return self.model.predict(test)

class Ensemble:

    def __init__(self,n_folds,stacker,base_models):
        self.n_folds=n_folds
        self.stacker=stacker
        self.base_models=base_models

    def fit_predict(self,train,test):
        numModels=len(self.base_models)
        numTrain=train.shape[0]
        numTest=test.shape[0]
        x=train.drop(['price_doc','id'],axis=1).values
        y=np.log(train['price_doc'].values.flatten())
        t=test.drop(['id'],axis=1).values

        x_fillNa=train.drop(['price_doc'],axis=1).fillna(-999).values
        t_fillNa=test.fillna(-999).values

        folds=list(KFold(len(y),n_folds=self.n_folds,shuffle=True))
        stacker_train=np.zeros((x.shape[0],numModels))
        stacker_test=np.zeros((t.shape[0],numModels))

        for i,clf in enumerate(self.base_models):
            print('Train base model'+str(i+1)+'....')
            output=np.zeros((t.shape[0],self.n_folds))
            for j,(train_index,test_index) in enumerate(folds):
                print('Training round '+str(j+1)+'....')
                if clf not in [model_lgb,model_xgb]:
                    x=x_fillNa
                    t=t_fillNa
                if clf in [model_lgb]:
                    print('here')
                    x=pd.read_csv('train_lgb.csv')
                    x=x.values
                    t=pd.read_csv('test_lgb.csv')
                    t=t.values
                    y=pd.read_csv('label_lgb.csv')
                    y=y.values.flatten()
                    print(np.shape(y))
                x_train=x[train_index]
                y_train=y[train_index]
                x_test=x[test_index]
                clf.fit(x_train,y_train)
                prediction=clf.predict(x_test)
                print('prediction:',prediction)
                stacker_train[test_index,i]=prediction
                output[:,j]=clf.predict(t)
            stacker_test[:,i]=output.mean(axis=1)
        #self.corr=pd.DataFrame('xgb')
        self.corr = pd.DataFrame(stacker_train,columns=['lgb','xgb','rf']).corr()
        corrmat(self.corr,inflate=False)
        plt.show()
        #cv
        #ss=ShuffleSplit(n_splits=5,test_size=0.3,random_state=0)
        #stacking_score=cross_val_score(estimator=self.stacker,X=stacker_train,y=y,cv=ss,scoring='neg_mean_squared_error')
        #print('cross validation result:',np.sqrt(stacking_score.mean()))
        self.stacker.fit(stacker_train,y)
        final_predictions=self.stacker.predict(stacker_test)
        return np.exp(final_predictions)


def encoding(data):
    for i in data.columns:
        if data[i].dtype == 'object':
            le = preprocessing.LabelEncoder()
            # print(i)
            # print('Missing',data.ix[:,i].isnull().sum())
            le.fit(np.array(data[i].astype(str)))
            data[i] = le.transform(np.array(data[i].astype(str)))
    return data

train=pd.read_csv('newTrain.csv')
test=pd.read_csv('newTest.csv')

train=encoding(train)
test=encoding(test)
test_id=test.id


xgb_param = {'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1}

lgb_param={'objective':'regression',
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

linearLgb_param = {'booster': 'gblinear', 'alpha': 0,  # for gblinear, delete this line if change back to gbtree
           'eta': 0.1, 'max_depth': 2, 'subsample': 1, 'colsample_bytree': 1, 'min_child_weight': 1,
           'gamma': 0, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse'}

model_lgb=Lightgbm(lgb_param)
model_xgb=XGboost(xgb_param)
model_linear=XGboost(linearLgb_param)
RF = RandomForestRegressor(n_estimators=500, max_features=17)
ETR = ExtraTreesRegressor(n_estimators=500, max_features=17, max_depth=None)
#Ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=200)
GBR = GradientBoostingRegressor(n_estimators=200,max_depth=5,max_features=0.5)

E=Ensemble(5,model_linear,[model_lgb,model_xgb,RF])
output=E.fit_predict(train,test)
print('final output',output)
output=pd.DataFrame({'id':test_id,'price_doc':output})
output.to_csv('output_stacking.csv',index=False)











