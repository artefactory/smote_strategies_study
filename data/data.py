import os

import pandas as pd
import numpy as np


def load_abalone_data():
    """
    Load Abalone diabates data set from data\dexternals folder
    The name of the file shoulde be : abalone.data
    """
    df_abalone =  pd.read_csv(os.path.join(os.getcwd(), 'externals', 'abalone.data'),names=['Sex', 'Length', 'Diameter','Height', 'Whole_weight','Shucked_weight','Viscera_weight','Shell_weight', 'Rings'])
    ### the abalone with a number of Rings equall to 18 are the samples of the minority class
    dict_mapping = {15: 0, 7: 0, 9: 0, 10: 0, 8: 0, 20: 0, 16: 0, 19: 0, 14: 0, 11: 0, 12: 0,18: 1, 13: 0, 5: 0, 4: 0, 6: 0, 21: 0, 17: 0, 22: 0, 1: 0, 3: 0, 26: 0, 23: 0,29: 0, 2: 0, 27: 0, 25: 0, 24: 0} 
    df_abalone.replace({'Rings':dict_mapping},inplace=True)

    X_abalone = df_abalone.drop(['Rings','Sex'],axis=1).to_numpy()
    y_abalone = df_abalone[['Rings']].values.ravel()
    return X_abalone, y_abalone

def load_pima_data():
    """
    Load PIMA diabates data set from data\dexternals folder
    The name of the file shoulde be : diabetes.csv
    """
    df_diabete = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'diabetes.csv'))
    X_pima = df_diabete.drop(['Outcome'],axis=1).to_numpy()
    y_pima = df_diabete[['Outcome']].values.ravel()
    return X_pima, y_pima


def load_phoneme_data():
    """
    Load Phoneme diabates data set from data\dexternals folder
    The name of the file shoulde be : phoneme.csv
    """
    df_phoneme = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'phoneme.csv'))
    X_phoneme = df_phoneme.drop([' Class'],axis=1).to_numpy()
    y_phoneme = df_phoneme[[' Class']].values.ravel()
    return X_phoneme, y_phoneme

def load_df_phoneme_positifs():
    """
    Load Phoneme diabates data set from data\dexternals folder and then keep only keep the minority (positive) samples
    The name of the file shoulde be : phoneme.csv
    """
    df_phoneme = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'phoneme.csv'))
    df_phoneme_positifs = df_phoneme[df_phoneme[' Class']==1].copy().reset_index(drop=True)
    df_phoneme_positifs.drop([' Class'],axis=1,inplace=True)
    return df_phoneme_positifs

####### GA4 ##########

def preprocess_ga4_7features(df):
    
    #Preprocessing
    df["browser"] = df["browser"].apply(lambda x: "Other" if x=='<Other>' else x )
    df["operating_system"] = df["operating_system"].apply(lambda x: "Other" if x=='<Other>' else x )

    #Converts dates into datetimes:
    df['start_feature_date'] = pd.to_datetime(df['start_feature_date'])
    df['end_feature_date'] = pd.to_datetime(df['end_feature_date'])
    
    #Sort samples in chrologic time. 
    df = df.sort_values(by=['start_feature_date'], ascending=True)
    df.reset_index(drop=True,inplace=True)
    
    #build the meta dataframe
    meta_df = df[['user_pseudo_id']].copy()
    meta_df['fold']=0 # futre colonne fold
    meta_df['keep_in_test']=False
    
    X = df[['total_events','total_page_view','total_add_to_cart','number_of_sessions','engaged_sessions',
        'bounce_rate','total_purchase']].copy()
    y= df[['target_label']].copy() 
    
    return X, y,meta_df

def load_ga4_data():
    """
    Load GA4 diabates data set from data\dexternals folder.
    The name of the file shoulde be : data_ga4.csv. 
    """
    df = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'data_ga4.csv'))
    X, y,meta_df = preprocess_ga4_7features(df=df)
    return X.to_numpy(),y.to_numpy().ravel(),meta_df

def load_df_positifs_ga4():
    df_GA4,meta_df_GA4 = load_ga4_data()
    df_positifs_GA4 = df_GA4[df_GA4['target_label']==1].copy().reset_index(drop=True)
    df_positifs_GA4.drop(['target_label'],axis=1,inplace=True)
    df_positifs_GA4
    return df_positifs_GA4






