import pandas as pd
import numpy as np

data=pd.read_csv('resale-flat-prices.csv',header=None)
data.columns=['Resale_month','Town','Flat_type','Block_number','Street_name','Storey_range','Floor_area','Flat_model','Lease_date',
              'Resale_price']
#print(data.head(10))
records=data.loc[(data['Flat_type']=='3 ROOM')&(data['Flat_model']=='TERRACE'),:]
results=pd.read_csv('q1.csv',header=None)
records.to_csv('records.csv',index=False)
print(records.equals(results))

