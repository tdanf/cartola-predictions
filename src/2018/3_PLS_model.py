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

CURRENT_ROUND = 26

# Importing the dataset
df = pd.read_csv('data/dados_2018.csv')

df = df[df['atletas.rodada_id'] >= 6]

df = df.drop(['A', 'CA', 'CV', 'DD', 'DP', 'FC', 'FD', 'FF', 'FS', 'FT', 'G', 'GC', \
              'GS', 'I', 'PE', 'PP', 'RB', 'SG'], axis=1)

###
### PLS model
###

df1 = df[['atletas.pontos_num',
 'atletas.posicao_id',
 'atletas.rodada_id',
 'home_team',
 'off_own_team',
 'def_own_team',
 'off_adv_team',
 'def_adv_team',
 'score_index',
 'take_goal_index',
 'CA_sum_last5',
 'CV_sum_last5',
 'DD_sum_last5',
 'DP_sum_last5',
 'FC_sum_last5',
 'GC_sum_last5',
 'GS_sum_last5',
 'RB_sum_last5',
 'SG_sum_last5',
 'A_sum_last5',
 'FD_sum_last5',
 'FF_sum_last5',
 'FS_sum_last5',
 'FT_sum_last5',
 'G_sum_last5',
 'I_sum_last5',
 'PE_sum_last5',
 'PP_sum_last5',
 'atletas.pontos_num_sum_last5',
 'atletas.media_num_sum_last5',
 'CA_last1',
 'CV_last1',
 'DD_last1',
 'DP_last1',
 'FC_last1',
 'GC_last1',
 'GS_last1',
 'RB_last1',
 'SG_last1',
 'A_last1',
 'FD_last1',
 'FF_last1',
 'FS_last1',
 'FT_last1',
 'G_last1',
 'I_last1',
 'PE_last1',
 'PP_last1',
 'atletas.pontos_num_last1',
 'atletas.media_num_last1']]

df_atual = df1[df1['atletas.rodada_id'] == CURRENT_ROUND]
df_model = df1[df1['atletas.rodada_id'] != CURRENT_ROUND]

X = df_model.iloc[:,3:-1].values
y = df_model.iloc[:, 0].values
X_atual = df_atual.iloc[:,3:-1].values

# Splitting the df into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_atual = sc_X.transform(X_atual)

sc_y = StandardScaler()
y = y.reshape(-1,1)
sc_y.fit(y)
y_train = y_train.reshape(-1,1)
y_train = sc_y.transform(y_train)
y_test = y_test.reshape(-1,1)
y_test = sc_y.transform(y_test)

# Regressão linear
from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=3, scale=False)
pls2.fit(X_train, y_train)

# Predicting the Test set results
y_pred = pls2.predict(X_test)
y_pred_atual = pls2.predict(X_atual)

# coeficientes
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

df_atual = df[df['atletas.rodada_id'] == CURRENT_ROUND]
df_atual = df_atual[['atletas.apelido', 'atletas.clube.id.full.name', 'atletas.nome', 'atletas.posicao_id', \
                             'atletas.rodada_id', 'atletas.preco_num','atletas.pontos_num_sum_last5', 'atletas.media_num' ]]
df_atual['pred_score'] = y_pred_atual
df_atual.to_csv('predictions/predict-PLS.csv', encoding='utf-8')
# cross-validation score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = pls2, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()









