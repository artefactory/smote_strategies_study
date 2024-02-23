import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def pred_perso(y_pred_probas,treshold=0.5):
    tmp_df = pd.DataFrame({'y_pred_probas': y_pred_probas})
    #tmp_df = df_y_preds_probas.copy()
    tmp_df['res_final'] = tmp_df['y_pred_probas'].apply(lambda x: 1 if x >= treshold else 0)
    return tmp_df['res_final'].tolist()



####### run_eval ##########


def run_eval(output_dir, name_file, X, y, list_oversampling_and_params, splitter):
    ################## INITIALISATION #################
    n_strategy = len(list_oversampling_and_params)
    list_names_oversamplings =['y_true'] + [config[0] for config in list_oversampling_and_params]
    list_names_oversamplings.append('fold')

    #folds = splitter.split(X,y) # do it once for all models ## NOT POSSIBLE
    list_all_preds = [[] for i in range(n_strategy+2)]
    list_tree_depth = []
    
    X_copy, y_copy = X.copy(), y.copy()
    folds = list(splitter.split(X,y))
    for i,(oversampling_name, oversampling_func, oversampling_params, model) in enumerate(list_oversampling_and_params):
        forest = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'get_depth')
        
        for fold, (train, test) in enumerate(folds):
            ################## On prépare les données du Fold #############    
            X_res, y_res = oversampling_func.fit_resample(X=X_copy[train], y=y_copy[train], **oversampling_params)
            ######### On lance le modèle du Fold en question ###############

            X_res, y_res = shuffle(X_res,y_res) # to put in oversampling_func
            model.fit(X_res,y_res)
            if forest:
                curent_tree_depth = [estimator.get_depth() for estimator in model.estimators_]
                list_tree_depth.append(curent_tree_depth)
            y_pred_probas = model.predict_proba(X_copy[test])[:,1]
            
            ######## On stocke les résultats ###################
            list_all_preds[i+1].extend(y_pred_probas)
        #print(len(list_all_preds[i+1])) 
            if i == 0:
                list_all_preds[-1].extend(np.full((len(test),),fold))# enregistre info fold
                list_all_preds[0].extend(y_copy[test])# enregistre la cible
          
           
       
    #return list_all_preds
    #print('shape array : ',np.array(list_all_preds).shape)
    #print('shape array T : ',np.array(list_all_preds).T.shape)
    runs_path_file_strats = os.path.join(output_dir,"preds_"+ name_file)
    np.save(runs_path_file_strats,np.array(list_all_preds).T)
    np.save(os.path.join(output_dir,"name_strats"+name_file),list_names_oversamplings)
    np.save(os.path.join(output_dir,"depth"+name_file),np.array(list_tree_depth))

    
def compute_metrics(output_dir,name_file,list_metric):
    n_metric = len(list_metric)
    list_names_metrics = []
    for m in range(n_metric):
        list_names_metrics.append(list_metric[m][1])
    list_names_oversamplings = np.load(os.path.join(output_dir,"name_strats"+name_file))
    array_all_preds_strats_final = np.load(os.path.join(output_dir,"preds_"+ name_file))
    df_all = pd.DataFrame(array_all_preds_strats_final,columns=list_names_oversamplings)
    
    name_col_strategies = df_all.columns.tolist()[1:-1] # On enlève les colonnes  'y_true' et 'fold'
        
    array_resultats_metrics = np.zeros((n_metric,len(name_col_strategies)))
    array_resultats_metrics_std = np.zeros((n_metric,len(name_col_strategies)))
    for k in range(n_metric):
        for col_number,col_name in enumerate(name_col_strategies):
            ### moyenne des metrics sur les 5-fold de test:
            list_value=[]
            for j in range(5):
                df = df_all[df_all['fold']==j]
                y_true = df['y_true'].tolist()
                pred_probas_all = df[col_name].tolist()#.to_numpy().ravel()#tolist()
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


from sklearn.metrics import roc_auc_score
def compute_metrics_several_protocols(output_dir,init_name_file,list_metric,bool_roc_auc_only=True,n_iter=100):

    list_res = []

    
######### CAS ROC AUC
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
        
######## CAS toutes les métriques #######
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


def plot_roc_curves(output_dir,name_file):

    list_names_oversamplings = np.load(os.path.join(output_dir,"name_strats"+name_file))
    array_all_preds_strats_final = np.load(os.path.join(output_dir,"preds_"+ name_file))
    df_all = pd.DataFrame(array_all_preds_strats_final,columns=list_names_oversamplings)
    
    #plt.figure(figsize=(10,6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for col in df_all.drop(['y_true','fold'],axis=1).columns:
        fpr, tpr, _ = roc_curve(df_all[['y_true']].values, df_all[[col]].values)
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=roc_auc).plot(label=col,ax=ax)
        
    ax.set_title("Receiver Operating Characteristic (ROC) curves")
    ax.grid(linestyle="--")
    plt.legend()
    plt.show()
    
    
def max_depth_function(output_dir,name_file):
    list_names_oversamplings = np.load(os.path.join(output_dir,"name_strats"+name_file))[1:-1] # On enlève les colonnes  'y_true' et 'fold'
    array_tree_depth = np.load(os.path.join(output_dir,
                                        "depth"+name_file))
    array_tree_depth_mean=array_tree_depth.mean(axis=1)
    list_res_par_strat = []
    int_pas = len(list_names_oversamplings)

    for fold in range(5):
        list_res_par_strat.append(array_tree_depth_mean[0+int_pas*fold:int_pas+int_pas*fold])

    df_max_depth_by_fold = pd.DataFrame(np.array(list_res_par_strat),columns=list_names_oversamplings)
    return df_max_depth_by_fold



################ run eval subsample #################

def subsample_to_10(X,y,seed_sub):
    X_positifs = X[np.array(y, dtype=bool)]
    X_negatifs = X[np.array(1-y, dtype=bool)]
    
    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int( (0.1*len(X_negatifs))/(1-0.1) ) ## on calcule le nb d'éléments à conserver
    ##int pour avoir partie entière suppérieur
    idx = np.random.randint(len(X_positifs), size=n_undersampling_sub)
            
    X_positifs_selected = X_positifs[idx,:]
    y_positifs_selected = y[np.array(y, dtype=bool)][idx]
    
    X_res = np.concatenate((X_negatifs,X_positifs_selected),axis=0)
    y_res = np.concatenate((y[np.array(1-y, dtype=bool)],y_positifs_selected),axis=0)
    X_res, y_res = shuffle(X_res,y_res)
    return X_res,y_res

def subsample_to_1(X,y,seed_sub):
    X_positifs = X[np.array(y, dtype=bool)]
    X_negatifs = X[np.array(1-y, dtype=bool)]
    
    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int( (0.01*len(X_negatifs))/(1-0.01) ) ## on calcule le nb d'éléments à conserver
    ##int pour avoir partie entière suppérieur
    idx = np.random.randint(len(X_positifs), size=n_undersampling_sub)
            
    X_positifs_selected = X_positifs[idx,:]
    y_positifs_selected = y[np.array(y, dtype=bool)][idx]
    
    X_res = np.concatenate((X_negatifs,X_positifs_selected),axis=0)
    y_res = np.concatenate((y[np.array(1-y, dtype=bool)],y_positifs_selected),axis=0)
    X_res, y_res = shuffle(X_res,y_res)
    return X_res,y_res

def run_eval_subsample(output_dir, name_file, X, y, list_oversampling_and_params,splitter,
                       subsample,subsubsample,seed_sub,seed_subsub):
    ################## INITIALISATION #################
    n_strategy = len(list_oversampling_and_params)
    list_names_oversamplings =['y_true'] + [config[0] for config in list_oversampling_and_params]
    list_names_oversamplings.append('fold')

    #folds = splitter.split(X,y) # do it once for all models ## NOT POSSIBLE
    list_all_preds = [[] for i in range(n_strategy+2)]
    list_tree_depth = []
    
    X_copy, y_copy = X.copy(), y.copy()
    ############## On regarde s'il faut undersampler ##################
    X_positifs = X_copy[np.array(y_copy, dtype=bool)]
    X_negatifs = X_copy[np.array(1-y_copy, dtype=bool)]

    if subsample:
        X_copy,y_copy = subsample_to_10(X_copy,y_copy,seed_sub=seed_sub)
        if subsubsample :
            X_copy,y_copy = subsample_to_1(X_copy,y_copy,seed_sub=seed_subsub)
    np.random.seed(seed=None)
    folds = list(splitter.split(X_copy,y_copy))
    ##############################################
    ######## Début protocole par stratgie #######
    ##############################################
    for i,(oversampling_name, oversampling_func, oversampling_params, model) in enumerate(list_oversampling_and_params):
        forest = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'get_depth')
        
        for fold, (train, test) in enumerate(folds):
            ################## On prépare les données du Fold #############
            X_res, y_res = oversampling_func.fit_resample(X=X_copy[train], y=y_copy[train], **oversampling_params)
            ######### On lance le modèle du Fold en question ###############
            X_res, y_res = shuffle(X_res,y_res) # to put in oversampling_func
            model.fit(X_res,y_res)
            if forest:
                curent_tree_depth = [estimator.get_depth() for estimator in model.estimators_]
                list_tree_depth.append(curent_tree_depth)
            y_pred_probas = model.predict_proba(X_copy[test])[:,1]
            #print('fold %i' % fold, roc_auc_score(y_copy[test], y_pred_probas))
            
            ######## On stocke les résultats ###################
            list_all_preds[i+1].extend(y_pred_probas)
            if i == 0:
                list_all_preds[-1].extend(np.full((len(test),),fold))# enregistre info fold
                list_all_preds[0].extend(y_copy[test])# enregistre la cible

    
    runs_path_file_strats = os.path.join(output_dir,"preds_"+ name_file)
    np.save(runs_path_file_strats,np.array(list_all_preds).T)
    np.save(os.path.join(output_dir,"name_strats"+name_file),list_names_oversamplings)
    np.save(os.path.join(output_dir,"depth"+name_file),np.array(list_tree_depth))