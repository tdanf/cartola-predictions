# -*- coding: utf-8 -*-
"""
Criado em 2018-09-15

@author: Thiago Fleck
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

RODADA_ATUAL = 27

# Importing the dataset
df = pd.read_csv('../data/dados_2018.csv')

df = df[df['atletas.rodada_id'] >= 6]

df = df.drop(['A', 'CA', 'CV', 'DD', 'DP', 'FC', 'FD', 'FF', 'FS', 'FT', 'G', 'GC', \
              'GS', 'I', 'PE', 'PP', 'RB', 'SG'], axis=1)






###
### Modelo MLR with one-hot encode for position and opponet team
###
array_pos = df[['atletas.posicao_id']]
array_pos = pd.get_dummies(array_pos, drop_first=True)
array_adv_team = df[['adv_team_name']]
array_adv_team = pd.get_dummies(array_adv_team, drop_first=True)
df1 = df[['atletas.pontos_num','home_team','atletas.pontos_num_sum_last5','atletas.rodada_id']]
df1 = pd.concat([df1, array_pos, array_adv_team], axis=1)
df_atual = df1[df1['atletas.rodada_id'] == RODADA_ATUAL]
df_model = df1[df1['atletas.rodada_id'] != RODADA_ATUAL]
X = df_model.iloc[:,1:-1].values
y = df_model.iloc[:, 0].values
X_atual = df_atual.iloc[:,1:-1].values
# Splitting the df into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Regressão linear
linreg = LinearRegression()
linreg.fit(X_train, y_train)
# Predicting the Test set results
y_pred = linreg.predict(X_test)
y_pred_atual = linreg.predict(X_atual)
# coeficientes
linreg.coef_
linreg.intercept_
mean_squared_error(y_test, y_pred)
print(r2_score(y_test, y_pred))
df_atual = df[df['atletas.rodada_id'] == RODADA_ATUAL]
df_atual = df_atual[['atletas.apelido', 'atletas.clube.id.full.name', 'atletas.nome', 'atletas.posicao_id', \
                             'atletas.rodada_id', 'atletas.preco_num','atletas.pontos_num_sum_last5', 'atletas.media_num' ]]
df_atual['pred_score'] = y_pred_atual
df_atual.to_csv('../prediction/predict-MLR-1.csv', encoding='utf-8')
# cross-validation score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = linreg, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
accuracies.std()









