#Basic libraries
import gc,os,csv,logging,pickle,warnings,random,time,sys,random
from datetime import datetime
from numba import jit
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Matrix and array 
import numpy as np
import pandas as pd
import dask.dataframe as dd

#Modeling
import lightgbm as lgb
import xgboost as xgb
#from sklearn.linear_model import LinearRegression #,BayesianRidge
import catboost as cat
from catboost import Pool

# Model evaluation
#from sklearn.preprocessing import StandardScaler#, QuantileTransformer #, MinMaxScaler, KBinsDiscretizer
#from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score  #mean_absolute_error#, #mean_squared_error, make_scorer,
from sklearn.model_selection import StratifiedKFold #KFold,GroupKFold #,GroupShuffleSplit# RepeatedKFold # train_test_split, cross_val_score
#from imblearn.under_sampling import RandomUnderSampler


# parameters for models

param_lgb = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          #'max_depth': 13,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 66,
          "metric": 'xentropy',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }


param_cat = {
    'random_seed': 6666,
    'subsample': 0.7,
    'use_best_model': True,
    'loss_function': 'Logloss', #'CrossEntropy'
    'learning_rate': 0.01,
    'max_depth': 6
}

param_xgb = {
    'objective' : 'binary:logistic'
    , 'eta' : 0.05
    ,'gamma': 0.0
    , 'max_depth': 10
    , 'min_child_weight' : 1
    , 'subsample' : 0.8
    , 'colsample_bytree' : 1
    , 'lambda' : 0.8
    #, 'alpha' :
    , 'verbosity' : 0
    , 'nthread' : 4
    , 'booster' : 'gbtree'
    #, 'num_boost_round' : 100000
    , 'eval_metric':'auc'
}


param = param_lgb

def CrossVal(mod, param, num_round, isSample, SCALE, DISP,ERSR, target, train, train_y, test, folds, features, isRS=False, bag_num=5, sc=False):
    sts_time = time.time()
    seedlist = [6, 66, 666, 6666, 66666, 666666]
    print(train[features].shape)
    print(test[features].shape)
    if isSample:
        print(str(SCALE))
        train_1 = train[train[target]==1]
        train_0 = train[train[target]==0].sample(n=SCALE*len(train_1),replace=True,random_state=seedlist[c])
        train = pd.concat([train_1, train_0], axis=0)
        train_y_1 = train_y[train_y[target]==1]
        train_y_0 = train_y[train_y[target]==0].sample(n=SCALE*len(train_y_1),replace=True,random_state=seedlist[c])
        train_y = pd.concat([train_y_1, train_y_0], axis=0)
        #sampler = RandomUnderSampler(random_state=666)
        #train, train_y = sampler.fit_resample(train, train_y)
        print(train[train[target]==1].shape)
        print(train[train[target]==0].shape)    
    else:
        pass
    
    oof = np.zeros(len(train))
    #train_pred = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()
    #split_groups = train['_weekday']
    #for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train_y)):
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train_y, groups=split_groups)):
        strLog = f'Fold {fold_ + 1} started at {time.ctime()}'
        print(strLog)
                
        X_tr, X_val = train.iloc[trn_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train_y[target].iloc[trn_idx], train_y[target].iloc[val_idx]
        
        if sc == True:
            sc = StandardScaler()
            X_tr, X_val = sc.fit_transform(X_tr), sc.fit_transform(X_val)
        else:
            pass
        
        
        if mod == 'lgb':
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            if isRS:
                for i in range(bag_num):
                    
                    param.update({"bagging_seed": seedlist[i]})
                    model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=DISP,
                              early_stopping_rounds = ERSR)
                    oof[val_idx] +=  model.predict(X_val, num_iteration=model.best_iteration) /bag_num
                    #predictions
                    predictions += model.predict(test[features]) / (folds.n_splits*bag_num)
                
            else:
                model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=DISP,
                              early_stopping_rounds = ERSR)
                oof[val_idx] =  model.predict(X_val, num_iteration=model.best_iteration)
                #predictions
                predictions += model.predict(test[features]) / folds.n_splits
                       
    
            #feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = features
            fold_importance_df["importance"] = model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        elif mod == 'lnr':
            model = LinearRegression(normalize = True, n_jobs=-1).fit(X_tr, y_tr)
            oof[val_idx] =  model.predict(X_val) 
            #predictions
            predictions += model.predict(test[features]) / folds.n_splits
        
        elif mod == 'cat':
            trn_data = Pool(X_tr, y_tr)
            eval_set = (X_val, y_val)
            model = cat.train(pool=trn_data, params=param, verbose = DISP,  num_boost_round=num_round, eval_set=eval_set,
                              early_stopping_rounds=ERSR)
            oof[val_idx] = model.predict(X_val, prediction_type= 'Probability')[:,1]
            #predictions
        
            predictions += model.predict(test[features]) / folds.n_splits
            
        elif mod == 'xgb':
            trn_data = xgb.DMatrix(X_tr, label=y_tr)
            val_data = xgb.DMatrix(X_val, label=y_val)
            evalist = [(trn_data, 'train'), (val_data, 'test')]
            model = xgb.train(param, trn_data, num_round, evalist, verbose_eval=DISP, early_stopping_rounds = ERSR)
            oof[val_idx] = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_ntree_limit)
            
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = model.get_score().keys()
            fold_importance_df["importance"] = model.get_score().values
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            predictions += model.predict(xgb.DMatrix(test[features])) / folds.n_splits
        else:
            pass

        
    
        ed_time = time.time()
        calc_time = ed_time - sts_time
        print('Calc Time : %.2f [sec]' % calc_time)
        gc.collect()

    end_time = time.time()
    calc_time = end_time - sts_time
    print('Calc Time : %.2f [sec]' % calc_time)
    
    score = roc_auc_score(train_y[target], oof)
    print("CV OOF score: {:<8.5f}".format(score))
    #print("CV AVG score: {:<8.5f}".format(metric(train_t_y['target'], train_t_pred)))
    
    return [oof, predictions, feature_importance_df, score, model]

def save_pred(oof, predictions, sub_version, ver, PRDPATH):
    dt = datetime.now().strftime("%Y%m%d_%H:%M")
    path_oof = f'{PRDPATH}/'+sub_version+'_'+dt+'oof_'+ver+'.pickle'
    with open(path_oof, mode='wb') as f:
        pickle.dump(oof,f)
    path_pred = f'{PRDPATH}/'+sub_version+'_'+dt+'predictions_'+ver+'.pickle'
    with open(path_pred, mode='wb') as f:
        pickle.dump(predictions,f)
        
'''
def time_folds(train, test, key, keylist,i, folds):
    fold = i
    trn_idx = train[train[key].isin(keylist[i:i+6-folds])].index
    val_idx = train[(train[key] == keylist[i+(7-folds)])].index
    #tes_idx = test[test[key] == keylist[i+7]].index
    return [fold, trn_idx, val_idx ]
'''

def time_folds(train, test, key, keylist,i, folds):
    fold = i
    trn_idx = train[train[key].isin(keylist[i:i+6-folds])].index
    val_idx = train[(train[key] == keylist[i+(7-folds)])].index
    #tes_idx = test[test[key] == keylist[i+7]].index
    return [fold, trn_idx, val_idx ]

def TimeVal(mod, param, num_round,  DISP,ERSR, target, train, train_y, test, folds, features, sc=False):
    sts_time = time.time()
    
    print(train[features].shape)
    print(test[features].shape)
    
    oof = np.zeros(len(train))
    #train_pred = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    yearmonth = ['2017/12', '2018/1', '2018/2', '2018/3', '2018/4', '2018/5','2018/6']
    keylist = yearmonth
    key = '_year_month'   
    for i in range(folds):
        fold_, trn_idx, val_idx = time_folds(train, test, key, yearmonth,i,folds)
        strLog = f'Fold {fold_ + 1} started at {time.ctime()}'
        print(strLog)
                
        X_tr, X_val = train.iloc[trn_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train_y[target].iloc[trn_idx], train_y[target].iloc[val_idx]
        
        if sc == True:
            sc = StandardScaler()
            X_tr, X_val = sc.fit_transform(X_tr), sc.fit_transform(X_val)
        else:
            pass
        
        
        if mod == 'lgb':
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=DISP,
                              early_stopping_rounds = ERSR)
            oof[val_idx] =  model.predict(X_val, num_iteration=model.best_iteration) 
    
            #feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = features
            fold_importance_df["importance"] = model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        elif mod == 'lnr':
            model = LinearRegression(normalize = True, n_jobs=-1).fit(X_tr, y_tr)
            oof[val_idx] =  model.predict(X_val) 
        else:
            pass    

        #predictions
        
        predictions += model.predict(test[features]) / folds
    
        ed_time = time.time()
        calc_time = ed_time - sts_time
        print('Calc Time : %.2f [sec]' % calc_time)
        gc.collect()

    end_time = time.time()
    calc_time = end_time - sts_time
    print('Calc Time : %.2f [sec]' % calc_time)

    idx =  train[train[key].isin(keylist[7-folds:])].index
    score = roc_auc_score(train_y.loc[idx, target], oof[idx])
    print("CV OOF score: {:<8.5f}".format(score))

    
    return [oof, predictions, feature_importance_df, score, model]