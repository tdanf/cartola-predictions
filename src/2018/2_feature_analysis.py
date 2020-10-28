# -*- coding: utf-8 -*-
"""
Criado em 2018-09-15

@author: Thiago Fleck
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_ROUND = 26

# Importing the dataset
df = pd.read_csv('../data/dados_2018.csv')

df = df[df['atletas.rodada_id'] != CURRENT_ROUND]
df = df[df['atletas.rodada_id'] >= 20]


#list(df) # to get columns names
features = ['atletas.pontos_num', 
 'atletas.posicao_id',
 'home_team', 'id_adv_team', 'off_own_team', 'def_own_team',
 'off_adv_team', 'def_adv_team', 'score_index', 'take_goal_index', 'CA_sum_last5', 'CV_sum_last5', 'DD_sum_last5',
 'DP_sum_last5', 'FC_sum_last5', 'GC_sum_last5', 'GS_sum_last5', 'RB_sum_last5', 'SG_sum_last5', 'A_sum_last5',
 'FD_sum_last5', 'FF_sum_last5', 'FS_sum_last5', 'FT_sum_last5', 'G_sum_last5', 'I_sum_last5', 'PE_sum_last5', 
 'PP_sum_last5', 'atletas.pontos_num_sum_last5', 'atletas.media_num_sum_last5', 'CA_last1', 'CV_last1', 'DD_last1',
 'DP_last1', 'FC_last1', 'GC_last1', 'GS_last1', 'RB_last1', 'SG_last1', 'A_last1', 'FD_last1', 'FF_last1', 'FS_last1',
 'FT_last1', 'G_last1', 'I_last1', 'PE_last1', 'PP_last1', 'atletas.pontos_num_last1', 'atletas.media_num_last1']

df_select = df[features]

df_corr = df_select.corr()

sns.pairplot(df, x_vars=['off_own_team','def_own_team','home_team',\
            'RB_sum_last5','SG_sum_last5','FS_sum_last5','atletas.pontos_num_sum_last5'], \
             y_vars='atletas.pontos_num', size=4, aspect=0.8)

#sns.relplot(x="atletas.pontos_num_sum_last5", y="atletas.pontos_num", hue="atletas.posicao_id", data=df_select);
sns.regplot(x="atletas.pontos_num_sum_last5", y="atletas.pontos_num", data=df_select);
sns.lmplot(x="atletas.pontos_num_sum_last5", y="atletas.pontos_num", hue="atletas.posicao_id", data=df_select);


sns.regplot(x="home_team", y="atletas.pontos_num", data=df_select);


df_select_tec = df_select[df_select['atletas.posicao_id'] == 'tec']
df_corr_tec = df_select_tec.corr()

df_select_gol = df_select[df_select['atletas.posicao_id'] == 'gol']
df_corr_gol = df_select_gol.corr()

df_select_zag = df_select[df_select['atletas.posicao_id'] == 'zag']
df_corr_zag = df_select_zag.corr()

df_select_lat = df_select[df_select['atletas.posicao_id'] == 'lat']
df_corr_lat = df_select_lat.corr()

df_select_mei = df_select[df_select['atletas.posicao_id'] == 'mei']
df_corr_mei = df_select_mei.corr()

df_select_ata = df_select[df_select['atletas.posicao_id'] == 'ata']
df_corr_ata = df_select_ata.corr()


features = [ 'atletas.pontos_num', 'atletas.apelido', 'atletas.atleta_id', 'atletas.clube.id.full.name', 'atletas.nome',
 'atletas.posicao_id',
 'home_team', 'id_adv_team', 'off_own_team', 'def_own_team',
 'off_adv_team', 'def_adv_team', 'score_index', 'take_goal_index', 'CA_sum_last5', 'CV_sum_last5', 'DD_sum_last5',
 'DP_sum_last5', 'FC_sum_last5', 'GC_sum_last5', 'GS_sum_last5', 'RB_sum_last5', 'SG_sum_last5', 'A_sum_last5',
 'FD_sum_last5', 'FF_sum_last5', 'FS_sum_last5', 'FT_sum_last5', 'G_sum_last5', 'I_sum_last5', 'PE_sum_last5', 
 'PP_sum_last5', 'atletas.pontos_num_sum_last5', 'atletas.media_num_sum_last5', 'CA_last1', 'CV_last1', 'DD_last1',
 'DP_last1', 'FC_last1', 'GC_last1', 'GS_last1', 'RB_last1', 'SG_last1', 'A_last1', 'FD_last1', 'FF_last1', 'FS_last1',
 'FT_last1', 'G_last1', 'I_last1', 'PE_last1', 'PP_last1', 'atletas.pontos_num_last1', 'atletas.media_num_last1']