# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:02:34 2018

@author: Anumula_Anudeep
"""

import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import os
os.chdir(r"D:\EAISMSD\IPLdata")

score_df = pd.read_csv("TrainDeliveries.csv")
match_df = pd.read_csv("Trainmatches.csv")
score_df.head()
score_df.columns
match_df.columns
score_df.player_dismissed.unique() # gives name of the player dismissed
# runs and wickets per over #
score_df = pd.merge(score_df, match_df[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')
score_df.player_dismissed.fillna(0, inplace=True)
score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1
train_df = score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
train_df.columns = train_df.columns.get_level_values(0)

# innings score and wickets #
train_df['innings_wickets'] = train_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
train_df['innings_score'] = train_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
train_df.head()

# Get the target column #
temp_df = train_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
temp_df = temp_df.ix[temp_df['inning']==1,:]
temp_df['inning'] = 2
temp_df.columns = ['match_id', 'inning', 'score_target']
train_df = train_df.merge(temp_df, how='left', on = ['match_id', 'inning'])
train_df['score_target'].fillna(-1, inplace=True)

# get the remaining target #
def get_remaining_target(row):
    if row['score_target'] == -1.:
        return -1
    else:
        return row['score_target'] - row['innings_score']

train_df['remaining_target'] = train_df.apply(lambda row: get_remaining_target(row),axis=1)

# get the run rate #
train_df['run_rate'] = train_df['innings_score'] / train_df['over']

# get the remaining run rate #
def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])
    
train_df['required_run_rate'] = train_df.apply(lambda row: get_required_rr(row), axis=1)

def get_rr_diff(row):
    if row['inning'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']
    
train_df['runrate_diff'] = train_df.apply(lambda row: get_rr_diff(row), axis=1)
train_df['is_batting_team'] = (train_df['team1'] == train_df['batting_team']).astype('int')
train_df['target'] = (train_df['team1'] == train_df['winner']).astype('int')



#No Missing values
train_df.isnull().sum() .sum() #0

re_col=['match_id', 'target']
train_df[re_col]
train_df.columns




x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']



# create the input and target variables #
X = np.array(train_df[x_cols[:]])
Y = np.array(train_df['target'])







from sklearn.model_selection import train_test_split
#doing a stratified sample on Y i.e exited
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0,  stratify=Y)


## mdeling

from sklearn.tree import DecisionTreeClassifier
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier

#Cross validation
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion' :['entropy','gini'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[5,8,12,15],
    'min_samples_leaf':[5,8,12,15]
    
}

#rus = make_pipeline(RandomUnderSampler(),DecisionTreeClassifier(n_jobs=-1,random_state=5151))

### grid search for cart model
cart = DecisionTreeClassifier(random_state=5151)
CV_cart = GridSearchCV(estimator=cart, param_grid=param_grid, cv= 5)
CV_cart.fit(X_train,Y_train)
print (CV_cart.best_params_)
CV_cart.score(X_test, Y_test)  # 0.6912809845727163

### grid search for Random forest model

rforest = RandomForestClassifier(n_jobs=-1,random_state=5151,warm_start=True)

CV_rforest = GridSearchCV(estimator=rforest, param_grid=param_grid, cv= 5)
CV_rforest.fit(X_train,Y_train)
print (CV_rforest.best_params_)
#best_params_
CV_rforest.score(X_test, Y_test) #0.7020280811232449 estimators-deafult-10


### grid search for Gradientboost model
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[5,8,12,15],
    'min_samples_leaf':[5,8,12,15]
    
}
gboost = GradientBoostingClassifier(warm_start=True,random_state=5151)
CV_gboost = GridSearchCV(estimator=gboost, param_grid=param_grid, cv= 5)
CV_gboost.fit(X_train,Y_train)
print (CV_gboost.best_params_)
CV_gboost.score(X_test, Y_test)  #


### grid search for KNN classifier model
from sklearn.neighbors import KNeighborsClassifier
param_grid = {
    'n_neighbors': [3,5,8,9,10,12]
    
}
KNN= KNeighborsClassifier( n_jobs=-1)
CV_KNN = GridSearchCV(estimator=KNN, param_grid=param_grid, cv= 5)
CV_KNN.fit(X_train,Y_train)
print (CV_KNN.best_params_)
CV_KNN.score(X_test, Y_test) #0.6661466458658346

### grid search for adaboost model
param_grid = {
    'learning_rate': [0.1,0.2,0.3,0.4,0.5]
   
}

adaboost = AdaBoostClassifier()
CV_adaoost = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv= 5)
CV_adaoost.fit(X_train,Y_train)
print (CV_adaoost.best_params_)
CV_adaoost.score(X_test, Y_test) #0.6805338880221875

### grid search for Bagging classifier model
param_grid = {
    'n_estimators': [10,20,30,40,50,60,70,80],
    'max_samples':[5,10,15,20]
    
}

Bagg = BaggingClassifier(warm_start=True,random_state=5151)
CV_Bagg = GridSearchCV(estimator=Bagg, param_grid=param_grid, cv= 5)
CV_Bagg.fit(X_train,Y_train)
print (CV_Bagg.best_params_)
CV_Bagg.score(X_test, Y_test) #0.6619864794591783


def roc_auc_plot(y_true, y_proba, y_pred,label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, f1_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f, prec=%.3f, F1=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1]), 
                       precision_score(y_true,y_pred), f1_score(y_true,y_pred) ))

f, ax = plt.subplots(figsize=(6,6))

roc_auc_plot(Y_test, CV_cart.predict_proba(X_test), CV_cart.predict(X_test), label='CART', l='-')
roc_auc_plot(Y_test, CV_rforest.predict_proba(X_test),CV_rforest.predict(X_test), label='RFOREST', l='--')
#roc_auc_plot(Y_test, CV_gboost.predict_proba(X_test),CV_gboost.predict(X_test), label='GBOOST', l='-.')
roc_auc_plot(Y_test, CV_adaoost.predict_proba(X_test),CV_adaoost.predict(X_test), label='ADABOOST', l=':')
roc_auc_plot(Y_test, CV_KNN.predict_proba(X_test),CV_KNN.predict(X_test), label='KNN', l='-')
roc_auc_plot(Y_test, CV_Bagg.predict_proba(X_test),CV_Bagg.predict(X_test), label='BAGG', l='--')


ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic[ROC] curves')


