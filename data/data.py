import os

import pandas as pd
import numpy as np



def load_pima_data():
    df_diabete = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'diabetes.csv'))
    X_pima = df_diabete.drop(['Outcome'],axis=1).to_numpy()
    y_pima = df_diabete[['Outcome']].values.ravel()
    return X_pima, y_pima


def load_phoneme_data():
    df_phoneme = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'phoneme.csv'))
    X_phoneme = df_phoneme.drop([' Class'],axis=1).to_numpy()
    y_phoneme = df_phoneme[[' Class']].values.ravel()
    return X_phoneme, y_phoneme

def load_df_phoneme_positifs():
    df_phoneme = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'phoneme.csv'))
    df_phoneme_positifs = df_phoneme[df_phoneme[' Class']==1].copy().reset_index(drop=True)
    df_phoneme_positifs.drop([' Class'],axis=1,inplace=True)
    return df_phoneme_positifs


######################
####### GA4 ##########
######################


### INPUT: données de bigquery
### OUTPUT: X,y et meta_df initialisé
def preprocess_ga4_7features(df):
    
    ## On enlève les chevrons des noms des modalités car pose pb avec XGB:
    df["browser"] = df["browser"].apply(lambda x: "Other" if x=='<Other>' else x )
    df["operating_system"] = df["operating_system"].apply(lambda x: "Other" if x=='<Other>' else x )

    #On convertit les dates en datetimes:
    df['start_feature_date'] = pd.to_datetime(df['start_feature_date'])
    df['end_feature_date'] = pd.to_datetime(df['end_feature_date'])
    
    #On range les observations par ordre chronologique. 
    #Cela est nécesaire afin de découper notre Cross validation chronologique.
    df = df.sort_values(by=['start_feature_date'], ascending=True)
    df.reset_index(drop=True,inplace=True)
    
    #### On s'occupe de meta_df #####
    meta_df = df[['user_pseudo_id']].copy()
    meta_df['fold']=0 # On crée la futur colonne des fold
    meta_df['keep_in_test']=False
    
    # On enregistre X et y sur des copies. On ne "modifie" plus df à partir d'ici
    X = df[['total_events','total_page_view','total_add_to_cart','number_of_sessions','engaged_sessions',
        'bounce_rate','total_purchase']].copy()
    y= df[['target_label']].copy() 
    
    return X, y,meta_df

def load_ga4_continuous():
    df = pd.read_csv(os.path.join(os.getcwd(), 'externals', 'data_ga4.csv'))
    X, y,meta_df = preprocess_ga4_7features(df=df)
    return X.to_numpy(),y.to_numpy().ravel(),meta_df

def load_df_positifs_ga4():
    df_GA4,meta_df_GA4 = load_ga4_continuous()
    df_positifs_GA4 = df_GA4[df_GA4['target_label']==1].copy().reset_index(drop=True)
    df_positifs_GA4.drop(['target_label'],axis=1,inplace=True)
    df_positifs_GA4
    return df_positifs_GA4






