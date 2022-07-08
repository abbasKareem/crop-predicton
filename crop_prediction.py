import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pickle

FILE_PATH = './main.xlsx'
df = pd.read_excel(FILE_PATH)


df_wheat = df.drop(['year', 'barley', 'maize', 'sorghmn'], axis=1)
df_barley = df.drop(['year', 'wheat', 'maize', 'sorghmn'], axis=1)
df_maize = df.drop(['year', 'wheat', 'barley', 'sorghmn'], axis=1)
df_sorghmn = df.drop(['year', 'wheat', 'maize', 'barley'], axis=1)

df_test = pd.read_excel(FILE_PATH, 'Sheet2')
df_test.drop(['السنة'], axis=1,  inplace=True)
df_test.columns = ['max_temp', 'min_temp',
                   'precipitation', 'humidity', 'wind_speed']

df_wheat_X = df_wheat.drop(['wheat'], axis=True)
df_wheat_y = df_wheat['wheat']

df_barley_X = df_barley.drop(['barley'], axis=True)
df_barley_y = df_barley['barley']

df_maize_X = df_maize.drop(['maize'], axis=True)
df_maize_y = df_maize['maize']

df_sorghmn_X = df_sorghmn.drop(['sorghmn'], axis=True)
df_sorghmn_y = df_sorghmn['sorghmn']

decision_tree_wheat_model = DecisionTreeRegressor()
decision_tree_wheat_model.fit(df_wheat_X.values, df_wheat_y)

decision_tree_barley_model = DecisionTreeRegressor()
decision_tree_barley_model.fit(df_barley_X.values, df_barley_y)

decision_tree_maize_model = DecisionTreeRegressor()
decision_tree_maize_model.fit(df_maize_X.values, df_maize_y)

decision_tree_sorghmn_model = DecisionTreeRegressor()
decision_tree_sorghmn_model.fit(df_sorghmn_X.values, df_sorghmn_y)


pickle.dump(decision_tree_wheat_model, open('./models/wheat_model.sav', 'wb'))

pickle.dump(decision_tree_barley_model, open(
    './models/barley_model.sav', 'wb'))

pickle.dump(decision_tree_maize_model, open('./models/maize_model.sav', 'wb'))

pickle.dump(decision_tree_sorghmn_model, open(
    './models/sorghmn_model.sav', 'wb'))
