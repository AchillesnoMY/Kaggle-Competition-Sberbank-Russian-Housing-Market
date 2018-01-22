import pandas as pd
import numpy as np

#compare the predictions from lightgbm and xgboost
pred_lgb=pd.read_csv('output_lgb.csv')
pred_xgb=pd.read_csv('output_xgb_newFeatures.csv')
pred_comb=pd.read_csv('output_combination.csv')
id=pred_lgb.id
pred_lgb=pred_lgb['price_doc']
pred_xgb=pred_xgb['price_doc']
pred_comb=pred_comb['price_doc']
#print(pred_lgb.mean())
#print(pred_xgb.mean())
#print(pred_comb.mean())
final_pred=pred_xgb*0.25+pred_lgb*0.25+pred_comb*0.5
final_output=pd.DataFrame({'id':id,'price_doc':final_pred})
final_output.to_csv('final_output.csv',index=False)

#pred_xgb*0.4+pred_lgb*0.6=0.31339,0.31085
#pred_xgb*0.35+pred_lgb*0.65=0.31330, 0,31067
#pred_xgb*0.3+pred_lgb*0.7= 0.31324, 0.31052
#pred_xgb*0.2+pred_lgb*0.8=0.31320, 0.31026
#pred_xgb*0.25+pred_lgb*0.25+pred_comb*0.5  0.31137 0.30882
#pred_xgb*0.2+pred_lgb*0.2+pred_comb*0.6 0.31141 0.30892
#xgboost with new features, lightlgb without new features pred_xgb*0.2+pred_lgb*0.8=0.31323 0.30977

