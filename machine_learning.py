#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import datetime
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

#%%
model = XGBRegressor(objective = "reg:squarederror")
# %%
df = pd.read_csv("../data/model_export/train.csv")
# %%
df_location = df[['index','tract','lat','lon','city','county_name','region']]

# %%
df_train = df.drop(['index','tract','lat','lon','city','county_name','region'], axis =1)
# %%

Y = df_train['n_lsRestaurants']
X = df_train.drop(['n_lsRestaurants'], axis=1)

#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)
# %%
model_pred = model.fit(X_train, Y_train)
predictions = model.predict(X_test)
# %%
result = r2_score(Y_test, predictions)
result
# %%
predictions = pd.DataFrame(predictions)
# %%
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)
# %%
with open('model_pkl' , 'rb') as f:
    lr = pickle.load(f)

#%%
df_pred = model.predict(X)
df_pred = pd.DataFrame(df_pred)
#df_pred.head()
df['prediction'] = df_pred.reset_index()[0]
df.head()
# %%
df['predicted_difference'] = df['prediction'] - df['n_lsRestaurants']
# %%
# Show which columns have place for growth
df['growth_area'] = df['predicted_difference'].apply(lambda x: '1' if x > 0 else 0 )
df.head(10)
# %%
