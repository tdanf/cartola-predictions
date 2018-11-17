# -*- coding: utf-8 -*-
"""
Created on 2018-09-13

@author: Thiago Fleck
"""

import numpy as np
import pandas as pd

CURRENT_ROUND = 35

# Load data from all 2018 rounds
# Data from https://github.com/henriquepgomide/caRtola
rounds = []
rounds.append(pd.read_csv('../data/rodada-1.csv'))
rounds.append(pd.read_csv('../data/rodada-2.csv'))
rounds.append(pd.read_csv('../data/rodada-3.csv'))
rounds.append(pd.read_csv('../data/rodada-4.csv'))
rounds.append(pd.read_csv('../data/rodada-5.csv'))
rounds.append(pd.read_csv('../data/rodada-6.csv'))
rounds.append(pd.read_csv('../data/rodada-7.csv'))
rounds.append(pd.read_csv('../data/rodada-8.csv'))
rounds.append(pd.read_csv('../data/rodada-9.csv'))
rounds.append(pd.read_csv('../data/rodada-10.csv'))
rounds.append(pd.read_csv('../data/rodada-11.csv'))
rounds.append(pd.read_csv('../data/rodada-12.csv'))
rounds.append(pd.read_csv('../data/rodada-13.csv'))
rounds.append(pd.read_csv('../data/rodada-14.csv'))
rounds.append(pd.read_csv('../data/rodada-15.csv'))
rounds.append(pd.read_csv('../data/rodada-16.csv'))
rounds.append(pd.read_csv('../data/rodada-17.csv'))
rounds.append(pd.read_csv('../data/rodada-18.csv'))
rounds.append(pd.read_csv('../data/rodada-19.csv'))
rounds.append(pd.read_csv('../data/rodada-20.csv'))
rounds.append(pd.read_csv('../data/rodada-21.csv'))
rounds.append(pd.read_csv('../data/rodada-22.csv'))
rounds.append(pd.read_csv('../data/rodada-23.csv'))
rounds.append(pd.read_csv('../data/rodada-24.csv'))
rounds.append(pd.read_csv('../data/rodada-25.csv'))
rounds.append(pd.read_csv('../data/rodada-26.csv'))
rounds.append(pd.read_csv('../data/rodada-27.csv'))
rounds.append(pd.read_csv('../data/rodada-28.csv'))
rounds.append(pd.read_csv('../data/rodada-29.csv'))
rounds.append(pd.read_csv('../data/rodada-30.csv'))
rounds.append(pd.read_csv('../data/rodada-31.csv'))
rounds.append(pd.read_csv('../data/rodada-32.csv'))
rounds.append(pd.read_csv('../data/rodada-33.csv'))
rounds.append(pd.read_csv('../data/rodada-34.csv'))

df = pd.concat(rounds)
#df.shape[0]

cols_scouts_def = ['CA','CV','DD','DP','FC','GC','GS','RB','SG'] # alphabetical order
cols_scouts_atk = ['A','FD','FF','FS','FT','G','I','PE','PP'] # alphabetical order
cols_scouts = cols_scouts_def + cols_scouts_atk

# remove players who didn't play
df = df[(df['atletas.pontos_num'] !=0 ) & (df['atletas.variacao_num'] != 0)]
#df.shape[0]

# import list with all played matches and the matches of the next round
# data from https://github.com/henriquepgomide/caRtola
match_index = pd.read_csv('../data/2018_partidas.csv')

# creates a list with all players and asign the current round to then
players_list = df.drop_duplicates(subset = 'atletas.atleta_id')
players_list['atletas.rodada_id'] = CURRENT_ROUND
df = df.append(players_list)

# Incluir demais informações no DataFrame df necessárias para fazer o modelo
#
#   Será incluída coluna extras:
#       - idenficador de time
#       - informando se partida foi em casa ou fora de casa
#       - 5 colunas com resultados das partidas passadas
#       - índice de força do ataque e defesa dos times retira do site FiveThirtyEight (https://projects.fivethirtyeight.com/soccer-predictions/brasileirao/)
#       

# the two csv files above were created to make it easy to identify a club
#    in the 'rodada-xx.csv' e '2018_partidas.csv' files
club_id_partida = pd.read_csv('../data/clube_partida.csv', index_col = 0).to_dict()['id']
club_id_rodada = pd.read_csv('../data/clube_rodada.csv', index_col = 0).to_dict()['id']

club_id_rodada_2 = pd.read_csv('../data/clube_rodada.csv')

# Importa .csv com a força ofensiva e defensiva de cada time
# retirado de https://projects.fivethirtyeight.com/soccer-predictions/brasileirao/
forca_clubes = pd.read_csv('../data/forca-clubes-fte.csv')

# match place matrix
# 1 = home
# 0 = away 
# lines represent the rounds and columns represent the clubs.
home_or_away = pd.DataFrame(data = np.zeros((39, 20)))


# Inclui colunas extras no DataFrame de partidas (match_index)
# Novas columas são:
#   - id do time de casa
#   - id do time de fora

home_team_id_list = []
away_team_id_list = []

for index, row in match_index.iterrows():
    #print(row['home_team'], row['away_team'])
    home_team_id = club_id_partida[row['home_team']]
    away_team_id = club_id_partida[row['away_team']]
    home_team_id_list.append(home_team_id)
    away_team_id_list.append(away_team_id)
    home_or_away[home_team_id][row['round']] = 1
    home_or_away[away_team_id][row['round']] = 0
    
match_index['home_team_id'] = home_team_id_list
match_index['away_team_id'] = away_team_id_list

# new features for the DataFrame df
home_team_col = [] # Coluna para informação de jogo em casa ou fora.
id_team_col = [] # Coluna com o id do Time
id_adv_team_col = []
adv_team_name_col = []
off_own_team = [] # Coluna com força de ataque do time do jogador em questão
def_own_team = [] # Coluna com força de defesa do time do jogador em questão
off_adv_team = [] # Coluna com força de ataque do adversário do jogador em questão
def_adv_team = [] # Coluna com força de defesa do adversário do jogador em questão

for index, row in df.iterrows(): # itera jogador por jogador
    j = club_id_rodada[row['atletas.clube.id.full.name']] # pega nome do time no df e encontra o id dele
    i = row['atletas.rodada_id'] # numero da rodada

    # busca qual o id do time adversário
    try: # caso o jogador jogou em casa
        adv_id = match_index[ (match_index['round'] == i) & \
                                 (match_index['home_team_id'] == j)]['away_team_id'].iloc[0]
    except IndexError: # caso o jogador jogou fora
        try:
            adv_id = match_index[ (match_index['round'] == i) & \
                                 (match_index['away_team_id'] == j)]['home_team_id'].iloc[0]
        except IndexError: # caso não tenha sido encontrado o time adversário
            adv_id = j

    off_own_team.append(forca_clubes['Off'][j])
    def_own_team.append(forca_clubes['Def'][j])
    off_adv_team.append(forca_clubes['Off'][adv_id])
    def_adv_team.append(forca_clubes['Def'][adv_id])
    
    home_team_col.append(home_or_away[j][i])
    id_team_col.append(j)
    id_adv_team_col.append(adv_id)
    
    adv_team_name = club_id_rodada_2[club_id_rodada_2['id'] == adv_id]['clube'].iloc[0]
    adv_team_name_col.append(adv_team_name)

df['home_team'] = home_team_col
df['id_team'] = id_team_col
df['id_adv_team'] = id_adv_team_col
df['adv_team_name'] = adv_team_name_col
df['off_own_team'] = off_own_team
df['def_own_team'] = def_own_team
df['off_adv_team'] = off_adv_team
df['def_adv_team'] = def_adv_team

df['score_index'] = df['off_own_team'] * df['def_adv_team']
df['take_goal_index'] = df['off_adv_team'] * df['def_own_team']


# Transform cummulative feature into current round only feature
def fix_cummulative_feat (df, round_):
    suffixes = ('_curr', '_prev')
    cols_current = [col + suffixes[0] for col in cols_scouts]
    cols_prev = [col + suffixes[1] for col in cols_scouts]
    
    df_round = df[df['atletas.rodada_id'] == round_]
    if round_ == 1: 
        df_round.fillna(value=0, inplace=True)
        return df_round
    
    df_round_prev = df[df['atletas.rodada_id'] < round_].groupby('atletas.atleta_id', as_index=False)[cols_scouts].max()
    df_players = df_round.merge(df_round_prev, how='left', on=['atletas.atleta_id'], suffixes=suffixes)
    
    # if is the first round of a player, the scouts of previous rounds will be NaNs. Thus, set them to zero
    df_players.fillna(value=0, inplace=True)
    
    # compute the scouts 
    df_players[cols_current] = df_players[cols_current].values - df_players[cols_prev].values
    
    # update the columns
    df_players.drop(labels=cols_prev, axis=1, inplace=True)
    df_players = df_players.rename(columns=dict(zip(cols_current, cols_scouts)))
    df_players.SG = df_players.SG.clip_lower(0)
    
    return df_players

df1 = pd.DataFrame()

for round_ in range(1, CURRENT_ROUND):
    df_round = fix_cummulative_feat(df, round_)
    print("Dimensões da rodada #{0}: {1}".format(round_, df_round.shape))
    df1 = df1.append(df_round, ignore_index=True)

# get info for the last round from df
df1 = df1.append(df[df['atletas.rodada_id'] == CURRENT_ROUND], ignore_index=True)

# new features representing previous matchs points
points_n1 = [] # pontuação na rodada n-1
points_n2 = [] # pontuação na rodada n-2
points_n3 = [] # pontuação na rodada n-3
points_n4 = [] # pontuação na rodada n-4
points_n5 = [] # pontuação na rodada n-5
       
def previous_played_rounds(df, rodada, atleta):
    """Função usada para buscar no DataFrame df pontuação de rounds passadas
    
    Args:
        rodada: número da rodada onde será buscada a pontuação passada
        atleta: id od jogador a ser buscado
    """
    #if rodada <= 1: return [0,0,0,0,0]
    list_ = pd.DataFrame()
    
    for i in range(rodada-1, 0, -1): # loop reverso de (rodada-1) até 1
        row = df[ (df['atletas.rodada_id'] == (i)) & (df['atletas.atleta_id'] == atleta)]
        #print(row)
        if row.empty: # if player didnt play in round i, pass
            pass
#            print('pass')
        else:
            list_ = list_.append(row)
#            print('append')
        
        if len(list_) >= 5: 
            return list_
    
    # if there is 4 or less played round, append 0 to make len(list) = 5
    #for i in range(5-len(list_)): list_.append(0)
    
    return list_

#asd = previous_played_rounds(df1, 5, 88323)

df_last5 = pd.DataFrame()
df_last1 = pd.DataFrame()
cols_to_sum = cols_scouts + ['atletas.pontos_num', 'atletas.media_num']

# iteração para cada jogador em df
for index, row in df1.iterrows():
    rodada = row['atletas.rodada_id'] # rodada do jogador atual
    atleta = row['atletas.atleta_id'] # id do jogador atual
    
    # taking the sum of the last 5 scouts
    last_5_rounds = previous_played_rounds(df1, rodada, atleta)
    if last_5_rounds.empty:
        new_row = pd.DataFrame(columns=cols_to_sum)
        new_row.loc[0] = [0] * len(cols_to_sum)
        new_row5 = new_row.add_suffix('_sum_last5')
        df_last5 = df_last5.append(new_row5, ignore_index=True)
        
        new_row1 = new_row.add_suffix('_last1')
        df_last1 = df_last1.append(new_row1, ignore_index=True)

    else:
        last_5_rounds = last_5_rounds[cols_to_sum]
        new_row = pd.DataFrame(last_5_rounds.sum()).transpose()
        new_row5 = new_row.add_suffix('_sum_last5')
        df_last5 = df_last5.append(new_row5, ignore_index=True)
        
        new_row = last_5_rounds.iloc[0]
        new_row1 = new_row.add_suffix('_last1')
        df_last1 = df_last1.append(new_row1, ignore_index=True)
        #print(new_row)
    
#    i1, i2, i3, i4, i5 = previous_rounds_points(df1, rodada, atleta)
#    
#    points_n1.append(i1)
#    points_n2.append(i2)
#    points_n3.append(i3)
#    points_n4.append(i4)
#    points_n5.append(i5)

#df1['points_n1'] = points_n1
#df1['points_n2'] = points_n2
#df1['points_n3'] = points_n3
#df1['points_n4'] = points_n4
#df1['points_n5'] = points_n5

#df1['mean_5'] = (df1['points_n1'] + df1['points_n2'] + df1['points_n3'] + df1['points_n4'] + df1['points_n5']) / 5

df3 = pd.concat([df1, df_last5, df_last1], axis=1)

df3 = df3.drop(['Unnamed: 0', 'atletas.clube_id', 'atletas.foto','atletas.slug'], axis=1)


# salva em csv para ser usado no próximo script ou no R
df3.to_csv('../data/dados_2018.csv', encoding='utf-8', index=False)

#df = pd.read_csv('dados_2018.csv')



