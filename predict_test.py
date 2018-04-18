# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:02:34 2018

@author: Anumula_Anudeep
"""

import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import os
os.chdir(r"D:\EAISMSD\IPLdata")

test_score_df = pd.read_csv("TestDeliveries.csv")
test_match_df = pd.read_csv("Testmatches.csv")
test_score_df.head()
test_score_df.columns
test_match_df.columns
test_score_df.player_dismissed.unique() # gives name of the player dismissed
# runs and wickets per over #
test_score_df = pd.merge(test_score_df, test_match_df[['match_id','season','result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='match_id')
test_score_df.player_dismissed.fillna(0, inplace=True)
test_score_df['player_dismissed'].ix[test_score_df['player_dismissed'] != 0] = 1
test_df = test_score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
test_df.columns = test_df.columns.get_level_values(0)

# innings score and wickets #
test_df['innings_wickets'] = test_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
test_df['innings_score'] = test_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
test_df.head()

# Get the target column #
temp_df = test_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
temp_df = temp_df.ix[temp_df['inning']==1,:]
temp_df['inning'] = 2
temp_df.columns = ['match_id', 'inning', 'score_target']
test_df = test_df.merge(temp_df, how='left', on = ['match_id', 'inning'])
test_df['score_target'].fillna(-1, inplace=True)

# get the remaining target #
def get_remaining_target(row):
    if row['score_target'] == -1.:
        return -1
    else:
        return row['score_target'] - row['innings_score']

test_df['remaining_target'] = test_df.apply(lambda row: get_remaining_target(row),axis=1)

# get the run rate #
test_df['run_rate'] = test_df['innings_score'] / test_df['over']

# get the remaining run rate #
def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])
    
test_df['required_run_rate'] = test_df.apply(lambda row: get_required_rr(row), axis=1)

def get_rr_diff(row):
    if row['inning'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']
    
test_df['runrate_diff'] = test_df.apply(lambda row: get_rr_diff(row), axis=1)
test_df['is_batting_team'] = (test_df['team1'] == test_df['batting_team']).astype('int')
test_df['target'] = 0

test_df.head()



x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']



# create the input and target variables #
X = np.array(test_df[x_cols[:]])
Y = np.array(test_df['target'])

print (CV_rforest.best_params_)

test_df['target'] = CV_rforest.predict(X)
re_col=['match_id', 'target']
test_df[re_col]

final = test_df.groupby(['match_id']).last()

final = final[re_col]

file_name ='out2.csv'
final.to_csv(file_name, sep=',', encoding='utf-8')
