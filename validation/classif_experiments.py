import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def pred_perso(y_pred_probas,treshold=0.5):
    tmp_df = pd.DataFrame({'y_pred_probas': y_pred_probas})
    #tmp_df = df_y_preds_probas.copy()
    tmp_df['res_final'] = tmp_df['y_pred_probas'].apply(lambda x: 1 if x >= treshold else 0)
    return tmp_df['res_final'].tolist()

def subsample_to_ratio_indices(X,y,ratio,seed_sub,output_dir_subsampling,name_subsampling_file,
                              has_previous_under_sampling=False,previous_under_sampling=None):
    Path(output_dir_subsampling).mkdir(parents=True, exist_ok=True) ## build the directory if it does not exists
    if has_previous_under_sampling:
        
        X_under,y_under = X[previous_under_sampling,:], y[previous_under_sampling]
        X_positifs = X_under[np.array(y_under, dtype=bool)]
        X_negatifs = X_under[np.array(1-y_under, dtype=bool)]
    else :
        X_positifs = X[np.array(y, dtype=bool)]
        X_negatifs = X[np.array(1-y, dtype=bool)]
              
    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int( (ratio*len(X_negatifs))/(1-ratio) ) ## compute number of sample to keeep
    ##int() inr order to have upper integer part
    df_X = pd.DataFrame(data = X)
    if has_previous_under_sampling :
        indices_positifs_kept = np.random.choice(df_X.loc[previous_under_sampling].loc[np.array(y_under, dtype=bool)].index.values,
                               size=n_undersampling_sub,replace=False)
        indices_negatifs_kept = df_X.loc[previous_under_sampling].loc[np.array(1-y_under, dtype=bool),:].index.values
        indices_kept = np.hstack((indices_positifs_kept,indices_negatifs_kept))
    else:
        indices_positifs_kept = np.random.choice(df_X.loc[np.array(y, dtype=bool)].index.values,
                               size=n_undersampling_sub,replace=False)
        indices_negatifs_kept = df_X.loc[np.array(1-y, dtype=bool)].index.values
        indices_kept = np.hstack((indices_positifs_kept,indices_negatifs_kept))
    
    np.save(os.path.join(output_dir_subsampling,name_subsampling_file+".npy"), indices_kept)
    return indices_kept

def read_subsampling_indices(X,y,dir_subsampling,name_subsampling_file,get_indexes=False):
    indexes_subsampling = np.load(os.path.join(dir_subsampling,name_subsampling_file+".npy"))
    if get_indexes:
        return indexes_subsampling, X[indexes_subsampling,:], y[indexes_subsampling]
    else:
        return X[indexes_subsampling,:], y[indexes_subsampling]

####### run_eval ##########
def subsample_to_ratio(X,y,ratio,seed_sub):
    X_positifs = X[np.array(y, dtype=bool)]
    X_negatifs = X[np.array(1-y, dtype=bool)]
    
    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int( (ratio*len(X_negatifs))/(1-ratio) ) ## compute the number of sample to keep
    ##int() in order to have upper integer part
    idx = np.random.randint(len(X_positifs), size=n_undersampling_sub)
            
    X_positifs_selected = X_positifs[idx,:]
    y_positifs_selected = y[np.array(y, dtype=bool)][idx]
    
    X_res = np.concatenate((X_negatifs,X_positifs_selected),axis=0)
    y_res = np.concatenate((y[np.array(1-y, dtype=bool)],y_positifs_selected),axis=0)
    X_res, y_res = shuffle(X_res,y_res)
    return X_res,y_res

def run_eval(output_dir, name_file, X, y, list_oversampling_and_params,splitter,
             subsample=False,subsubsample=False,subsubsubsample=False,seed_sub=11,seed_subsub=9,
             seed_subsubsub=5,to_standard_scale=True):
    """
    Main function of the procol.
    output_dir is the path where the output files will be stored.
    list_oversampling_and_params is a list composed of tuple like (name, function, function_params, classifier).
    to_standard_scale is a boolean.
    """
    
    
    Path(output_dir).mkdir(parents=True, exist_ok=True) ## build the directory if it does not exists
    ################## INITIALISATION #################
    n_strategy = len(list_oversampling_and_params)
    list_names_oversamplings =['y_true'] + [config[0] for config in list_oversampling_and_params]
    list_names_oversamplings.append('fold')

    list_all_preds = [[] for i in range(n_strategy+2)]
    list_tree_depth = []
    list_tree_depth_name = []
    
    X_copy, y_copy = X.copy(), y.copy()
    ############## Check if undersampling is necessary ##################
    X_positifs = X_copy[np.array(y_copy, dtype=bool)]
    X_negatifs = X_copy[np.array(1-y_copy, dtype=bool)]
    if subsample:
        X_copy,y_copy = subsample_to_ratio(X_copy,y_copy,ratio=0.2,seed_sub=seed_sub)
        if subsubsample :
            X_copy,y_copy = subsample_to_ratio(X_copy,y_copy,ratio=0.1,seed_sub=seed_subsub)
            if subsubsubsample :
                X_copy,y_copy = subsample_to_ratio(X_copy,y_copy,ratio=0.01,seed_sub=seed_subsubsub)
    np.random.seed(seed=None)
    
    folds = list(splitter.split(X_copy,y_copy))
    ##############################################
    ######## Start protocol by strategy    #######
    ##############################################
    for i,(oversampling_name, oversampling_func, oversampling_params, model) in enumerate(list_oversampling_and_params):
        
        for fold, (train, test) in enumerate(folds):
            ################## Folds data are prepared #############
            X_train,y_train = X_copy[train],y_copy[train]
            X_test = X_copy[test]
            if to_standard_scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                
                
            X_res, y_res = oversampling_func.fit_resample(X=X_train, y=y_train, **oversampling_params)
            ######### Run of the given fold ###############
            X_res, y_res = shuffle(X_res,y_res) # to put in oversampling_func
            model.fit(X_res,y_res)
            forest = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'get_depth')
            if forest:
                curent_tree_depth = [estimator.get_depth() for estimator in model.estimators_]
                list_tree_depth.append(curent_tree_depth)
                list_tree_depth_name.append(oversampling_name)
                
            if to_standard_scale:
                X_test = scaler.transform(X_test)
            y_pred_probas = model.predict_proba(X_test)[:,1]
            
            ######## Results are saved ###################
            list_all_preds[i+1].extend(y_pred_probas)
            if i == 0:
                list_all_preds[-1].extend(np.full((len(test),),fold))# save information of the ciurrent testing fold
                list_all_preds[0].extend(y_copy[test])#save the information of the target value

    pd.DataFrame(np.array(list_tree_depth).T,columns=list_tree_depth_name
            ).to_csv(os.path.join(output_dir,"depth"+name_file[:-4]+".csv"))
    runs_path_file_strats = os.path.join(output_dir,"preds_"+ name_file)
    np.save(runs_path_file_strats,np.array(list_all_preds).T)
    np.save(os.path.join(output_dir,"name_strats"+name_file),list_names_oversamplings)


def compute_metrics(output_dir,name_file,list_metric):
    n_metric = len(list_metric)
    list_names_metrics = []
    for m in range(n_metric):
        list_names_metrics.append(list_metric[m][1])
    list_names_oversamplings = np.load(os.path.join(output_dir,"name_strats"+name_file))
    array_all_preds_strats_final = np.load(os.path.join(output_dir,"preds_"+ name_file))
    df_all = pd.DataFrame(array_all_preds_strats_final,columns=list_names_oversamplings)
    
    name_col_strategies = df_all.columns.tolist()[1:-1] # We remove  'y_true' and 'fold'
        
    array_resultats_metrics = np.zeros((n_metric,len(name_col_strategies)))
    array_resultats_metrics_std = np.zeros((n_metric,len(name_col_strategies)))
    for k in range(n_metric):
        for col_number,col_name in enumerate(name_col_strategies):
            ### Mean of the metrics on the 5 test folds:
            list_value=[]
            for j in range(5):
                df = df_all[df_all['fold']==j]
                y_true = df['y_true'].tolist()
                pred_probas_all = df[col_name].tolist()
                y_pred = pred_perso(y_pred_probas=pred_probas_all,treshold=0.5)

                if list_metric[k][2] == 'pred':
                    value_metric =list_metric[k][0](y_true=y_true,y_pred=y_pred)
                else:
                    value_metric =list_metric[k][0](y_true=y_true,y_score=pred_probas_all)
                list_value.append(value_metric)
            array_resultats_metrics[k,col_number] = np.mean(list_value)
            array_resultats_metrics_std[k,col_number] = np.std(list_value)
            
    df_mean_metric = pd.DataFrame(array_resultats_metrics,columns=name_col_strategies,index=list_names_metrics)
    df_std_metric = pd.DataFrame(array_resultats_metrics_std,columns=name_col_strategies,index=list_names_metrics)
    return df_mean_metric, df_std_metric
    

def compute_metrics_several_protocols(output_dir,init_name_file,list_metric,bool_roc_auc_only=True,n_iter=100):

    list_res = []

    
######### CASE  ROC AUC only is computed ######
    if bool_roc_auc_only==True:
        for i in range(n_iter):
            name_file = init_name_file + str(i) +'.npy'
            df_metrics_mean, df_metrics_std = compute_metrics(
                output_dir=output_dir,
                name_file=name_file,
                list_metric=[(roc_auc_score,'roc_auc','proba')] )
            list_res.append(df_metrics_mean.to_numpy())

            
        name_cols = df_metrics_mean.columns
        array_res = np.array(list_res)
        df_final_mean = pd.DataFrame(np.mean(array_res,axis=0).reshape((1,-1)), columns=name_cols)
        df_final_std = pd.DataFrame(np.std(array_res,axis=0).reshape((1,-1)), columns=name_cols)
        df_final_mean.index =['ROC AUC']
        df_final_std.index =['ROC AUC']
        
######## CASE all the metrics are computed #######
    else :
        for i in range(n_iter): 
            name_file = init_name_file + str(i) +'.npy'
            df_metrics_mean, df_metrics_std = compute_metrics(
                output_dir=output_dir,
                name_file=name_file,
                list_metric=list_metric )

            list_res.append(df_metrics_mean.to_numpy())

        name_cols = df_metrics_mean.columns
        array_res = np.array(list_res)
        df_final_mean = pd.DataFrame(np.mean(array_res,axis=0), columns=name_cols)
        df_final_std = pd.DataFrame(np.std(array_res,axis=0), columns=name_cols)
        list_metric_func,list_metric_name,list_metric_type = zip(*list_metric)
        df_final_mean.index =list_metric_name
        df_final_std.index =list_metric_name
        
    return df_final_mean,df_final_std

    

class MyTimeSeriesSplit(TimeSeriesSplit):
    """
    MyTimeSeriesSplit
    The starting split can be chosen with this child class from TimeSeriesSplit.
    """
    
    def __init__(self,n_splits=10, starting_split=5,max_train_size=None, test_size=None, gap=0):
        """
        """
        super().__init__(n_splits=n_splits,max_train_size=max_train_size, test_size=test_size, gap=gap)
        self.starting_split = starting_split
        
    def split(self,X, y=None, groups=None):
        """
        """
        folds = list(super().split(X))
        folds_from_starting_split = folds[self.starting_split:]
        return folds_from_starting_split

class MyTimeSeriesSplit_groupout(TimeSeriesSplit):
    """
    MyTimeSeriesSplit with group out on col_name_id. 
    All the samples with ID that have been seen during the training phase, are removed of the test set.
    """
    
    def __init__(self,meta_df,col_name_id,n_splits=10, starting_split=5,max_train_size=None, test_size=None, gap=0):
        """
        col_name_id : name of the column containing the ID
        """
        super().__init__(n_splits=n_splits,max_train_size=max_train_size, test_size=test_size, gap=gap)
        self.starting_split = starting_split
        self.meta_df = meta_df
        self.col_name_id = col_name_id
        
    def split(self,X, y=None, groups=None):
        """
        """
        folds = list(super().split(X))
        folds_from_starting_split = folds[self.starting_split:]
        final_folds_from_starting_split =[]
        for fold, (train_index, test_index) in enumerate(folds_from_starting_split):
            
            # Split:
            meta_df_train, meta_df_test = self.meta_df.iloc[train_index], self.meta_df.iloc[test_index]
            # Samples with ID that have been seen during training are removed from the test set:
            id_in_train = meta_df_train[self.col_name_id].unique().tolist()
            test_indices_to_keep = meta_df_test.index[~meta_df_test[self.col_name_id].isin(id_in_train)].tolist()
            tmp = (train_index,test_indices_to_keep)
            final_folds_from_starting_split.append(tmp)
            
        return final_folds_from_starting_split
    


def plot_roc_curves(output_dir,name_file):

    list_names_oversamplings = np.load(os.path.join(output_dir,"name_strats"+name_file))
    array_all_preds_strats_final = np.load(os.path.join(output_dir,"preds_"+ name_file))
    df_all = pd.DataFrame(array_all_preds_strats_final,columns=list_names_oversamplings)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for col in df_all.drop(['y_true','fold'],axis=1).columns:
        fpr, tpr, _ = roc_curve(df_all[['y_true']].values, df_all[[col]].values)
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=roc_auc).plot(label=col,ax=ax)
        
    ax.set_title("Receiver Operating Characteristic (ROC) curves")
    ax.grid(linestyle="--")
    plt.legend()
    plt.show()