import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
target=train.price_doc
#train['timestamp']=pd.to_datetime(train.timestamp)
price_info=train.ix[(train.build_year>1691)&(train.build_year<2019),:]#remove outliers in price

#look at the price variable
def plotPriceVariable(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    plt.scatter(x=np.arange(price_info.shape[0]),y=price_info['price_doc'])
    plt.xlabel('index',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.title('Selling Prices')
    plt.savefig('Price Distribution.png')
    plt.show()

#look at the increasing price distribution
def plotIncreasingPriceVariable(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    plt.scatter(x=np.arange(price_info.shape[0]),y=np.sort(price_info['price_doc'].values))
    plt.xlabel('index',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.title('Selling Prices')
    plt.savefig('Increasing Price Distribution.png')
    plt.show()

#we bin the price and look at it
def price_bins(price_info):
    plt.figure(figsize=(6,6))
    sns.distplot(price_info['price_doc'].values,bins=50,kde=True)
    plt.xlabel('price',fontsize=12)
    plt.title('Price Distribution')
    plt.savefig('Bins_Price_Distribution.png')
    plt.show()

#we bin the log of price
def log_price_bins(price_info):
    plt.figure(figsize=(6,6))
    sns.distplot(np.log(price_info['price_doc'].values),bins=50,kde=True)
    plt.xlabel('log-price',fontsize=12)
    plt.title('Log-Price Distribution')
    plt.savefig('Bins_Log_Price_Distrubtion.png')
    plt.show()

#look at the correlation among internal house features and prices
def plotInternalCorrelation(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='white')
    internal_data=price_info.loc[:,['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room',
                    'kitch_sq', 'state', 'price_doc']]
    corr_internal_mat=internal_data.corr()
    #mask = np.zeros_like(corr_internal_mat, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(10, 220, sep=80, n=7)
    sns.heatmap(corr_internal_mat,vmax=.3, center=0,cmap=cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('internal_correlation.png')
    plt.show()

#look at the relationship between the full_sq and price
def fullSquare_Price(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    plt.scatter(price_info['full_sq'].values,price_info['price_doc'].values,c='red')
    plt.xlabel('full_sq',fontsize=12)
    plt.ylabel('price',fontsize=12)
    plt.title('full_sq vs price')
    plt.savefig('full_sq_vs_price.png')
    plt.show()

#look at the relationship between the num_room and the price
def numRoom_Price(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    plt.scatter(price_info['price_doc'].values,price_info['num_room'].values,c='red')
    plt.title('num_room vs price')
    plt.xlabel('price',fontsize=12)
    plt.ylabel('num_room',fontsize=12)
    plt.savefig('num_room_vs_price.png')
    plt.show()

#look at the mean prices changes with the time
def price_timestamp(price_info):
    temp_data=price_info.copy()
    temp_data['yearmonth']=temp_data['timestamp'].apply(lambda x:int(x[:4]+x[5:7]))
    grouped=temp_data.groupby('yearmonth')['price_doc'].aggregate(np.mean).reset_index()
    print(grouped)
    plt.figure(figsize=(8,8))
    sns.set(style='darkgrid',color_codes=True)
    sns.pointplot('yearmonth','price_doc',data=grouped,color='r',alpha=0.8)
    plt.xlabel('Year-Month',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.title('Mean Price vs Time')
    plt.xticks(rotation='vertical')
    plt.savefig('price_timestamp.png')
    plt.show()
price_timestamp(price_info)
#look at the relationship between the price and the state
def state_price(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    price_info.loc[price_info.state == 33, 'state'] = np.nan
    sns.violinplot(x='state',y='price_doc',data=price_info)
    plt.savefig('state_price.png')
    plt.show()

#look at the relationship between the price and the build material
def material_price(price_info):
    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    sns.violinplot(x='material',y='price_doc',data=price_info)
    plt.savefig('material_price.png')
    plt.show()


#look at the price below 6000000
def price_below_6e(price_info):
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    plt.scatter(x=np.arange(price_info.shape[0]),y=price_info['price_doc'])
    plt.xlabel('index',fontsize=12)
    plt.ylabel('price',fontsize=12)
    plt.ylim(ymin=0,ymax=6e6)
    plt.title('Selling Prices below 6e6')
    plt.savefig('Price Distribution below 6e6')
    plt.show()

#plot the prices distributions of Investment and OwnerOccupier types
def price_invest_owner(price_info):
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    #train['year']=train['timestamp'].dt.year
    #data_all=pd.concat([train,test],axis=0,ignore_index=True)
    #s=train.groupby(['product_type','build_year'])['price_doc'].mean().reset_index()
    #s.columns=['type','build_year','price']
    investment_price=price_info.ix[price_info.product_type=='Investment','price_doc']
    investment_year=price_info.ix[price_info.product_type=='Investment','build_year']
    owner_price=price_info.ix[price_info.product_type=='OwnerOccupier','price_doc']
    owner_year=price_info.ix[price_info.product_type=='OwnerOccupier','build_year']
    plt.scatter(np.arange(investment_price.shape[0]),investment_price,label='investment')
    plt.scatter(np.arange(owner_price.shape[0]),owner_price,label='Owner')
    plt.xlabel('Build Year',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.ylim(ymin=1000000,ymax=10000000)
    plt.title('Investment and OwnerOccupier prices distribution')
    plt.savefig('Investment and OwnerOccupier prices distribution.png')
    plt.legend(loc='best')
    plt.show()

#plot predicted prices from xgboost vs the true prices
def predict_from_xgb():
    predictions=pd.read_csv('train_prediction_xgb.csv')
    train=pd.read_csv('newTrain.csv')
    true=train['price_doc']
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    plt.scatter(np.arange(predictions.shape[0]),true,label='True Prices')
    plt.scatter(np.arange(predictions.shape[0]),predictions['price_doc'],label='Predicted Prices')
    plt.xlabel('Index',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.legend(loc='best')
    plt.title('Predictions from XGboost')
    plt.savefig('Predictions from XGboost.png')
    plt.show()
    #show the line plot
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    plt.plot(predictions.loc[:50,'price_doc'],label='predicted prices')
    plt.plot(train.loc[:50,'price_doc'],label='true prices')
    plt.legend(loc='best')
    plt.show()

#plot predicted investment prices from xgboost vs the true investment prices
def investmentCompare():
    predictions = pd.read_csv('train_prediction_xgb.csv')
    train=pd.read_csv('newTrain.csv')
    true=train['price_doc']
    investment_index=train[train.ix[:,'product_type_Investment']==1].index
    investment_price=train.loc[investment_index,'price_doc']
    predict_investment_price=predictions.loc[investment_index,'price_doc']
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    plt.scatter(np.arange(predict_investment_price.shape[0]),investment_price,label='True Prices')
    plt.scatter(np.arange(predict_investment_price.shape[0]),predict_investment_price,label='Predicted Prices')
    plt.xlabel('Index',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.legend(loc='best')
    plt.title('Investment Prices')
    plt.show()

#plot predicted investment prices from xgboost vs the true owner-occupier prices
def ownerOccupierCompare():
    predictions = pd.read_csv('train_prediction_xgb.csv')
    train=pd.read_csv('newTrain.csv')
    true=train['price_doc']
    owner_index=train[train.ix[:,'product_type_OwnerOccupier']==1].index
    owner_price=train.loc[owner_index,'price_doc']
    predict_owner_price=predictions.loc[owner_index,'price_doc']
    plt.figure(figsize=(8,6))
    sns.set(style='whitegrid')
    plt.scatter(np.arange(predict_owner_price.shape[0]),owner_price,label='True Prices')
    plt.scatter(np.arange(predict_owner_price.shape[0]),predict_owner_price,label='Predicted Prices')
    plt.xlabel('Index',fontsize=12)
    plt.ylabel('Price',fontsize=12)
    plt.legend(loc='best')
    plt.title('Owner Prices')
    plt.show()














