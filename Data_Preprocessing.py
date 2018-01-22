import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
train['isTrain'] = 1
label = train['price_doc']

test = pd.read_csv('test.csv')

# bad_address_fix=pd.read_excel('D:/Year4 Sem1/FYP/bank/BAD_ADDRESS_FIX.xlsx').set_index('id')
test['isTrain'] = 0
macro = pd.read_csv('D:/Year4 Sem1/FYP/bank/macro.csv')
print(np.shape(train))
print(np.shape(test))
print(np.shape(macro))

data_all = pd.concat([train, test], ignore_index=True)
#count for each build year, how many products for each type
#s=data_all.groupby(['build_year','product_type']).size()

def one_hot_encoding(data):
    to_remove=[]
    cat_var=None
    #one_hot_encoding
    for i in data.columns:
        if data[i].dtype=='object':
            new_dummies=pd.get_dummies(data[i],prefix=i)

            if cat_var is None:
                cat_var=new_dummies
            else:
                cat_var=pd.concat([cat_var,new_dummies],axis=1)
            to_remove.append(i)
    data.drop(to_remove,inplace=True,axis=1)
    #Remove the nominal columns with insignificant number of entries to prevent overfitting
    to_remove=[]
    #print(cat_var.columns)
    if cat_var is not None:
        sum_list=cat_var.sum(axis=0)
        to_remove=sum_list[sum_list<200].index.values
        cat_var=cat_var.ix[:,list(set(cat_var.columns)-set(to_remove))]
        data=pd.concat([data,cat_var],axis=1)
    return data

#use xgboost to find the feature importance
def FeatureImportance(data, label):
    data=one_hot_encoding(data)

    xgb_param = {'eta': 0.05,
                 'max_depth': 5,
                 'gamma': 0.1,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 'objective': 'reg:linear',
                 'eval_metric': 'rmse',
                 'silent': 1}
    dtrain = xgb.DMatrix(data, label)
    ##cv = xgb.cv(xgb_param, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=50, show_stdv=False,
     #           nfold=5, seed=1)
    #num_boost_rounds = len(cv) + 1
    model = xgb.train(dict(xgb_param, silent=0), dtrain, num_boost_round=451)
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=40, height=0.8, ax=ax)
    plt.savefig('feature_importance.png')
    plt.show()
    featureImportance = model.get_fscore()
    features = pd.DataFrame()
    features['features'] = featureImportance.keys()
    features['importance'] = featureImportance.values()
    features.sort_values(by=['importance'], ascending=False, inplace=True)
    return features
#s=FeatureImportance(train,label)
#remove insignificant features
def removeFeatures():
    newtrain = pd.read_csv('newTrain.csv')
    newtest=pd.read_csv('newTest.csv')
    label = newtrain['price_doc']
    newtrain.drop(['id', 'price_doc'], axis=1, inplace=True)
    features = FeatureImportance(newtrain, label)
    # new_features=features.ix[features.importance>20,:]
    remove_list = [features['features'].values[i] for i in range(np.shape(features)[0]) if
                   features['importance'].values[i] < 20]
    print('remove_list:',remove_list)
    newtrain.drop(remove_list, axis=1, inplace=True)
    newtest.drop(remove_list,axis=1,inplace=True)
    newtrain.to_csv('train_lessFeatures.csv', index=False)
    newtest.to_csv('test_lessFeatures.csv',index=False)


#plot correlation matrix
def plotCorrelationMatrix(features):
    topFeatures = features['features'].tolist()[:15]
    topFeatures.append('price_doc')
    corrMatt = train[topFeatures].corr()
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.heatmap(corrMatt, mask=mask, vmax=0.8, square=True, annot=True)
    plt.show()

#plot missing values
def plotMissingColumns(data):
    # Look at the missing values
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['feature', 'counts']
    missing_df.sort_values(['counts'], ascending=True, inplace=True)
    missing_df = missing_df.ix[missing_df.counts != 0, :]
    # print(missing_df)

    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(14, 20))
    rects = ax.barh(ind, missing_df.counts.values / len(data) * 100, color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.feature.values, rotation='horizontal')
    ax.set_xlabel("Percent of missing values %")
    ax.set_title("Percentage of missing values in each column")
    plt.savefig('Missing Values.png')
    plt.show()
#plotMissingColumns(data_all)

#create date relevant features
def createDateVariable(data, time,model_type):
    data[time] = pd.to_datetime(data[time])
    data['year'] = data[time].dt.year
    data['month'] = data[time].dt.month
    ''''''''''''''''''''''''''''''''''''''''''''
    if model_type=='xgboost' or model_type=='combination':
        data['day']=data[time].dt.day
        data['day_week']=data[time].dt.dayofweek
        #data['week_year']=data[time].dt.weekofyear
        data['quarter']=data[time].dt.quarter
        # data['day_month']=data[time].dt.days_in_month


#xgboost works better for dataset with new added features
#lightgbm works better for dataset without new added features
def dataPreprocessing(data,model_type=None):
    data_new = data.copy()
    # create the date variables
    createDateVariable(data_new, 'timestamp',model_type)

    ''' --------------------Detect outliers------------------------------'''

    data_new.ix[data_new.build_year==20052009,'build_year']=2005
    data_new.ix[data_new.build_year>2018,'build_year']=np.nan
    data_new.ix[data_new.build_year<1500,'build_year']=np.nan

    data_new.ix[data_new.num_room==0,'num_room']=np.nan

    data_new.ix[data_new.state==33,'state']=np.nan

    data_new.ix[data_new.full_sq>2000,'full_sq']=np.nan
    data_new.ix[data_new.full_sq<5,'full_sq']=np.nan

    data_new.ix[data_new.life_sq>200,'life_sq']=np.nan
    data_new.ix[data_new.life_sq<5,'life_sq']=np.nan

    sq_wrong_index=(data_new.life_sq>data_new.full_sq)
    #print(sq_wrong_index.sum())
    data_new.ix[sq_wrong_index,'life_sq']=np.nan

    data_new.ix[data_new.floor==0,'floor']=np.nan
    data_new.ix[data_new.max_floor==0,'max_floor']=np.nan

    floor_wrong_index=data_new.floor>data_new.max_floor
    #print(floor_wrong_index.sum())
    data_new.ix[floor_wrong_index,'max_floor']=np.nan

    data_new.ix[data_new.preschool_quota==0,'preschool_quota']=np.nan

    data_new.ix[data_new.kitch_sq>=data_new.life_sq,'kitch_sq']=np.nan
    data_new.ix[data_new.kitch_sq==0,'kitch_sq']=np.nan

    # Remove the extreme prices
    bad_index=data_new[(data_new.price_doc<=1e6)|(data_new.price_doc==2e6)|(data_new.price_doc==3e6)].index
    data_new.drop(bad_index,0,inplace=True)
    data_new.loc[(data_new.product_type == 'Investment') & (data_new.build_year < 2000), 'price_doc'] *= 0.9
    data_new.loc[data_new.product_type != 'Investment', 'price_doc'] *= 0.969  # Louis/Andy's magic number


    '''-----------------------------Feature Engineering----------------------'''
    if model_type=='xgboost' or model_type=='combination':
        #create housing age
        data_new['house_age']=data_new['year']-data_new['build_year']
        bad_index=data_new['house_age']<0
        data_new.ix[bad_index,'house_age']=np.nan
        #averge room area
        data_new['room_sq']=(data_new['life_sq']-data_new['kitch_sq'])/data_new['num_room']
        #age proportions in subarea
        data_new['male_prop']=data_new['male_f']/data_new['full_all']
        data_new['female_prop']=data_new['female_f']/data_new['full_all']
        data_new['young_prop']=data_new['young_all']/data_new['full_all']
        data_new['elder_prop']=data_new['ekder_all']/data_new['full_all']
        data_new['working_rate']=data_new['work_all']/data_new['full_all']
        data_new['retirement_rate']=data_new['ekder_all']/data_new['full_all']

        #inner house features
        data_new['floor_ratio'] = data_new['floor'] / data_new['max_floor']
        data_new['life_sq_prop']=data_new['life_sq']/data_new['full_sq']
        data_new['kitch_sq_prop']=data_new['kitch_sq']/data_new['full_sq']

    '''---------------------Handle Missing values---------------------'''
    '''
    data_new.life_sq.fillna(data_new.groupby(['sub_area'])['life_sq'].transform("median"), inplace=True)
    data_new.full_sq.fillna(data_new.groupby(['sub_area'])['full_sq'].transform("median"), inplace=True)
    data_new.kitch_sq.fillna(data_new.groupby(['sub_area'])['kitch_sq'].transform("median"), inplace=True)
    data_new.num_room.fillna(data_new.groupby(['sub_area'])['num_room'].transform("median"), inplace=True)

    data_new.floor.fillna(1, inplace=True)
    data_new.max_floor.fillna(1, inplace=True)
    wrong_max_floor_index = ((data_new['max_floor'] - data_new['floor']).fillna(-1)) < 0
    data_new['max_floor'][wrong_max_floor_index] = data_new['floor'][wrong_max_floor_index]
    data_new['max_floor'].fillna(1, inplace=True)

    data_new.material.fillna(data_new.groupby(['sub_area'])['material'].transform("median"), inplace=True)
    state_missing_map={33:3,None:0}
    data_new.state=data_new.state.replace(state_missing_map)
    data_new.build_year.fillna(data_new.groupby(['state'])['build_year'].transform('median'), inplace=True)


    #note that some life_sq are greater than the full_sq, some kitch_sq are greater than life_sq, which needed to be fixed
    wrong_life_index=data_new.life_sq>data_new.full_sq
    data_new.ix[wrong_life_index,'life_sq']=data_new.ix[wrong_life_index,'full_sq']*(3/5)
    wrong_kitch_index=data_new.kitch_sq>data_new.life_sq
    data_new.ix[wrong_kitch_index,'kitch_sq']=data_new.ix[wrong_kitch_index,'life_sq']*(1/3)

    #Neighbourhood features
    data_new['hospital_beds_raion'].fillna(0, inplace=True)

    data_new['cafe_avg_price_500'].fillna(data_new.cafe_avg_price_500.median(), inplace=True)
    data_new['cafe_sum_500_min_price_avg'].fillna(data_new.cafe_sum_500_min_price_avg.median(), inplace=True)
    data_new['cafe_sum_500_max_price_avg'].fillna(data_new.cafe_sum_500_max_price_avg.median(), inplace=True)    

    data_new['cafe_avg_price_1000'].fillna(data_new.cafe_avg_price_1000.median(), inplace=True)
    data_new['cafe_sum_1000_min_price_avg'].fillna(data_new.cafe_sum_1000_min_price_avg.median(), inplace=True)
    data_new['cafe_sum_1000_max_price_avg'].fillna(data_new.cafe_sum_1000_max_price_avg.median(), inplace=True)   

    data_new['cafe_avg_price_1500'].fillna(data_new.cafe_avg_price_1500.median(), inplace=True)
    data_new['cafe_sum_1500_min_price_avg'].fillna(data_new.cafe_sum_1500_min_price_avg.median(), inplace=True)
    data_new['cafe_sum_1500_max_price_avg'].fillna(data_new.cafe_sum_1500_max_price_avg.median(), inplace=True)

    data_new['cafe_avg_price_2000'].fillna(data_new.cafe_avg_price_2000.median(), inplace=True)
    data_new['cafe_sum_2000_min_price_avg'].fillna(data_new.cafe_sum_2000_min_price_avg.median(), inplace=True)
    data_new['cafe_sum_2000_max_price_avg'].fillna(data_new.cafe_sum_2000_max_price_avg.median(), inplace=True)   

    data_new['cafe_avg_price_3000'].fillna(data_new.cafe_avg_price_3000.median(), inplace=True)
    data_new['cafe_sum_3000_max_price_avg'].fillna(data_new.cafe_sum_3000_max_price_avg.median(), inplace=True)
    data_new['cafe_sum_3000_min_price_avg'].fillna(data_new.cafe_sum_3000_min_price_avg.median(), inplace=True)

    data_new['cafe_avg_price_5000'].fillna(data_new.cafe_avg_price_5000.median(), inplace=True)
    data_new['cafe_sum_5000_max_price_avg'].fillna(data_new.cafe_sum_5000_max_price_avg.median(), inplace=True)
    data_new['cafe_sum_5000_min_price_avg'].fillna(data_new.cafe_sum_5000_min_price_avg.median(), inplace=True)

    data_new['preschool_quota'].fillna(0, inplace=True)
    data_new['school_quota'].fillna(0, inplace=True)

    data_new['build_count_frame'].fillna(0, inplace=True)
    data_new['build_count_block'].fillna(data_new.build_count_block.median(), inplace=True)
    data_new['build_count_after_1995'].fillna(data_new.build_count_after_1995.median(), inplace=True)
    data_new['build_count_before_1920'].fillna(data_new.build_count_before_1920.median(), inplace=True)
    data_new['build_count_wood'].fillna(0, inplace=True)
    data_new['build_count_mix'].fillna(0, inplace=True)
    data_new['build_count_brick'].fillna(data_new.build_count_brick.median(), inplace=True)
    data_new['build_count_foam'].fillna(0, inplace=True)
    data_new['build_count_1921-1945'].fillna(0, inplace=True)
    data_new['build_count_monolith'].fillna(data_new.build_count_monolith.median(), inplace=True)
    data_new['build_count_panel'].fillna(data_new.build_count_panel.median(), inplace=True)
    data_new['build_count_slag'].fillna(0, inplace=True)
    data_new['raion_build_count_with_material_info'].fillna(data_new.raion_build_count_with_material_info.median(), inplace=True)
    data_new['raion_build_count_with_builddate_info'].fillna(data_new.raion_build_count_with_builddate_info.median(), inplace=True)
    data_new['build_count_1946-1970'].fillna(0, inplace=True)
    data_new['build_count_1971-1995'].fillna(0, inplace=True)

    data_new['prom_part_5000'].fillna(data_new['prom_part_5000'].median(), inplace=True)
    data_new['metro_km_walk'].fillna(data_new['metro_km_walk'].median(), inplace=True)
    data_new['metro_min_walk'].fillna(data_new['metro_min_walk'].median(), inplace=True)
    data_new['ID_railroad_station_walk'].fillna(data_new['ID_railroad_station_walk'].median(), inplace=True)
    data_new['railroad_station_walk_min'].fillna(data_new['railroad_station_walk_min'].median(), inplace=True)
    data_new['railroad_station_walk_km'].fillna(data_new['railroad_station_walk_km'].median(), inplace=True)
    data_new['green_part_2000'].fillna(data_new['green_part_2000'].median(), inplace=True)
    '''
    if model_type=='combination':
        for i in range(data_new.shape[0]):
            if pd.isnull(data_new.loc[:,'product_type'].values[i]):


                if data_new.loc[:,'build_year'].values[i]>2012:
                    data_new.loc[:,'product_type'].values[i]='OwnerOccupier'
                else:
                    data_new.loc[:,'product_type'].values[i]='Investment'

    data_new=one_hot_encoding(data_new)
    #print(data_new.loc[:,['product_type_Investment','product_type_OwnerOccupier']])
    data_new.drop(['year', 'timestamp'], 1, inplace=True)


    return data_new



#Split the dataset into train and test sets.
def splitDataSet(dataSet):
    train = dataSet.ix[dataSet.isTrain == 1, :].drop(['isTrain'], axis=1)
    test = dataSet.ix[dataSet.isTrain == 0, :].drop(['isTrain', 'price_doc'], axis=1)
    #print(np.shape(test))
    #print(test.ix[:,['product_type_Investment','product_type_OwnerOccupier']])
    train.to_csv('newTrain.csv', index=False)
    test.to_csv('newTest.csv', index=False)



new_data = dataPreprocessing(data_all,'xgboost')
splitDataSet(new_data)
#removeFeatures()
