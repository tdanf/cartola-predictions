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
from sklearn.model_selection import cross_val_score
    
CURRENT_ROUND = 27

# Importing the dataset
df = pd.read_csv('data/dados_2018.csv')

df = df[df['atletas.rodada_id'] >= 6]

all_features = ['atletas.pontos_num', 
 'atletas.posicao_id',
 'home_team', 'id_adv_team', 'off_own_team', 'def_own_team',
 'off_adv_team', 'def_adv_team', 'score_index', 'take_goal_index', 'CA_sum_last5', 'CV_sum_last5', 'DD_sum_last5',
 'DP_sum_last5', 'FC_sum_last5', 'GC_sum_last5', 'GS_sum_last5', 'RB_sum_last5', 'SG_sum_last5', 'A_sum_last5',
 'FD_sum_last5', 'FF_sum_last5', 'FS_sum_last5', 'FT_sum_last5', 'G_sum_last5', 'I_sum_last5', 'PE_sum_last5', 
 'PP_sum_last5', 'atletas.pontos_num_sum_last5', 'atletas.media_num_sum_last5', 'CA_last1', 'CV_last1', 'DD_last1',
 'DP_last1', 'FC_last1', 'GC_last1', 'GS_last1', 'RB_last1', 'SG_last1', 'A_last1', 'FD_last1', 'FF_last1', 'FS_last1',
 'FT_last1', 'G_last1', 'I_last1', 'PE_last1', 'PP_last1', 'atletas.pontos_num_last1', 'atletas.media_num_last1']

df = df.drop(['A', 'CA', 'CV', 'DD', 'DP', 'FC', 'FD', 'FF', 'FS', 'FT', 'G', 'GC', \
              'GS', 'I', 'PE', 'PP', 'RB', 'SG'], axis=1)

positions = ['tec', 'gol', 'zag', 'lat', 'mei', 'ata']
features_tec = ['atletas.pontos_num',
            'home_team','atletas.media_num_last1', 'off_own_team', 'def_own_team', 
            'atletas.rodada_id']
features_gol = ['atletas.pontos_num',
            'home_team','atletas.pontos_num_sum_last5', 'score_index', 'take_goal_index',
            'atletas.rodada_id']
features_zag = ['atletas.pontos_num',
            'home_team','atletas.pontos_num_sum_last5', 'score_index', 'take_goal_index', 'FS_sum_last5', 'SG_sum_last5', 
            'atletas.rodada_id']
features_lat = ['atletas.pontos_num',
            'home_team','atletas.pontos_num_sum_last5', 'score_index', 'take_goal_index', 'RB_sum_last5', 
            'atletas.rodada_id']
features_mei = ['atletas.pontos_num',
            'home_team','atletas.pontos_num_sum_last5', 'FS_sum_last5', 'RB_sum_last5', 'off_own_team', 'PE_sum_last5',
            'FF_sum_last5',
            'atletas.rodada_id']
features_ata = ['atletas.pontos_num',
            'home_team','atletas.pontos_num_sum_last5', 'FS_last1', 'FC_last1', 'FD_last1', 'FF_sum_last5','off_own_team',
            'atletas.rodada_id']

features = {
        'tec':features_tec,
        'gol':features_gol,
        'zag':features_zag,
        'lat':features_lat,
        'mei':features_mei,
        'ata':features_ata
        }

df_final = pd.DataFrame()

for pos in positions:
    print(pos)
    df1 = df[df['atletas.posicao_id'] == pos]
    df1 = df1[features[pos]]
    df_atual = df1[df1['atletas.rodada_id'] == CURRENT_ROUND]
    df_model = df1[df1['atletas.rodada_id'] != CURRENT_ROUND]
    X = df_model.iloc[:,1:-1].values
    y = df_model.iloc[:, 0].values
    X_atual = df_atual.iloc[:,1:-1].values
    # Splitting the df into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    # Regressão linear
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = linreg.predict(X_test)
    y_pred_atual = linreg.predict(X_atual)
    # coeficientes
    #linreg.coef_
    #linreg.intercept_
    #mean_squared_error(y_test, y_pred)
    print(r2_score(y_test, y_pred))
    df_atual = df[df['atletas.posicao_id'] == pos]
    df_atual = df_atual[df_atual['atletas.rodada_id'] == CURRENT_ROUND]
    df_atual = df_atual[['atletas.apelido', 'atletas.clube.id.full.name', 'atletas.nome', 'atletas.posicao_id', \
                                 'atletas.rodada_id', 'atletas.preco_num','atletas.pontos_num_sum_last5', 'atletas.media_num' ]]
    df_atual['pred_score'] = y_pred_atual
    df_final = df_final.append(df_atual)
    
    # cross-validation score
    accuracies = cross_val_score(estimator = linreg, X = X_train, y = y_train, cv = 5)
    print(accuracies.mean())
    #accuracies.std()

df_final.to_csv('predictions/predict-MLR-2.csv', encoding='utf-8')







