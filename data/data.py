import os

import pandas as pd
from scipy.io.arff import loadarff 

DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data", "externals"
)


def load_abalone_data():
    """
    Loads Abalone data set from data\dexternals folder.
    The name of the file shoulde be : abalone.data
    """
    filename = "abalone.data"
    try:
        df_abalone = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            names=[
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole_weight",
                "Shucked_weight",
                "Viscera_weight",
                "Shell_weight",
                "Rings",
            ],
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Abalone dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )
    ### the abalone with a number of Rings equall to 18 are the samples of the minority class
    dict_mapping = {18: 1}
    for key in [
        15,
        7,
        9,
        10,
        8,
        20,
        16,
        19,
        14,
        11,
        12,
        13,
        5,
        4,
        6,
        21,
        17,
        22,
        1,
        3,
        26,
        23,
        29,
        2,
        27,
        25,
        24,
    ]:
        dict_mapping[key] = 0

    df_abalone = df_abalone.replace({"Rings": dict_mapping})

    X_abalone = df_abalone.drop(["Rings", "Sex"], axis=1).to_numpy()
    y_abalone = df_abalone[["Rings"]].values.ravel()
    return X_abalone, y_abalone


def load_pima_data():
    """
    Load PIMA diabates data set from data\dexternals folder
    The name of the file shoulde be : diabetes.csv
    """
    df_diabete = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "diabetes.csv",
        )
    )
    X_pima = df_diabete.drop(["Outcome"], axis=1).to_numpy()
    y_pima = df_diabete[["Outcome"]].to_numpy().ravel()  # be consistent with X
    return X_pima, y_pima


def load_phoneme_data():
    """
    Load Phoneme diabates data set from data\dexternals folder
    The name of the file shoulde be : phoneme.csv
    """
    df_phoneme = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "phoneme.csv",
        ),
        names=["Aa", " Ao", " Dcl", " Iy", " Sh", " Class"],
    )
    X_phoneme = df_phoneme.drop([" Class"], axis=1).to_numpy()
    y_phoneme = df_phoneme[[" Class"]].to_numpy().ravel()
    return X_phoneme, y_phoneme


def load_df_phoneme_positifs():
    """
    Load Phoneme data set from data\dexternals folder and then keep only keep the minority (positive) samples
    The name of the file shoulde be : phoneme.csv
    """
    df_phoneme = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "phoneme.csv",
        ),
        names=["Aa", " Ao", " Dcl", " Iy", " Sh", " Class"],
    )
    df_phoneme_positifs = (
        df_phoneme[df_phoneme[" Class"] == 1].copy().reset_index(drop=True)
    )
    df_phoneme_positifs.drop(
        [" Class"], axis=1, inplace=True
    )  # personnally I prefer not to use inplace
    return df_phoneme_positifs


def load_yeast_data():
    """
    Load Yeast data set from data\dexternals folder
    The name of the file shoulde be : yeast.data
    """
    df_yeast = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "yeast.data",
        ),
        sep=r"\s+",
        names=['Sequence_Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','localization_site'],
    )
    df_yeast.replace(
        {
            "localization_site": {
                "MIT": 0,
                "NUC": 0,
                "CYT": 0,
                "ME1": 0,
                "EXC": 0,
                "ME2": 0,
                "VAC": 0,
                "POX": 0,
                "ERL": 0,
                "ME3": 1,
            }
        },
        inplace=True,
    )

    X_yeast = df_yeast.drop(["Sequence_Name", "localization_site"], axis=1).to_numpy()
    y_yeast = df_yeast[["localization_site"]].to_numpy().ravel()
    return X_yeast, y_yeast


def load_haberman_data():
    """
    Load Haberman data set from data\dexternals folder
    The name of the file shoulde be : haberman.data
    """
    df_haberman = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "haberman.data",
        ),
        sep=",",
        header=None,
    )
    df_haberman.columns = ["Age", "year_Op", "npand", "Class"]
    df_haberman.replace({"Class": {1: 0, 2: 1}}, inplace=True)
    X_haberman = df_haberman.drop(["Class"], axis=1).to_numpy()
    y_haberman = df_haberman[["Class"]].to_numpy().ravel()
    return X_haberman, y_haberman


def load_magictel_data():
    """
    Load haberman data set from data\dexternals folder
    The name of the file shoulde be : magictel.arff
    """
    raw_magic = loadarff(
                         os.path.join(
                             os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                             "data",
                             "externals",
                             "magictelescope.arff",
                             )
        )
    df_magic = pd.DataFrame(raw_magic[0])
    df_magic.replace({'class':{b'h':0,b'g':1}},inplace=True)
    X_magic = df_magic.drop(['class'],axis=1).to_numpy()   
    y_magic = df_magic[['class']].to_numpy().ravel()

    return X_magic, y_magic


def load_california_data():
    """
    Load California data set from data\dexternals folder
    The name of the file shoulde be : california.arff
    """
    raw_cal_housing = loadarff(
                         os.path.join(
                             os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                             "data",
                             "externals",
                             "california.arff",
                             )
        )
    df_cal_housing = pd.DataFrame(raw_cal_housing[0])
    df_cal_housing.replace({'price':{b'True':1,b'False':0}},inplace=True)
    X_cal_housing = df_cal_housing.drop(['price'],axis=1).to_numpy()
    y_cal_housing = df_cal_housing[['price']].to_numpy().ravel()

    return X_cal_housing, y_cal_housing


def load_house_data():
    """
    Load House_16h data set from data\dexternals folder
    The name of the file shoulde be : house_16H.arff
    """
    raw_house_16H = loadarff(
                         os.path.join(
                             os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                             "data",
                             "externals",
                             "house_16H.arff",
                             )
        )
    df_house_16H = pd.DataFrame(raw_house_16H[0])
    df_house_16H.replace({'binaryClass':{b'P':0,b'N':1}},inplace=True) 
    X_house_16H = df_house_16H.drop(['binaryClass'],axis=1).to_numpy()
    y_house_16H = df_house_16H[['binaryClass']].to_numpy().ravel()

    return X_house_16H, y_house_16H

def load_credit_data():
    """
    Load Creditcard data set from data\dexternals folder
    The name of the file shoulde be : creditcard.csv
    """
    df_credit = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "creditcard.csv",
        )
    )
    meta_df_credit = df_credit[['Time']]
    X_credit = df_credit.drop(['Time','Class'],axis=1).to_numpy()
    y_credit = df_credit[['Class']].to_numpy().ravel()

    return X_credit, y_credit, meta_df_credit


from ucimlrepo import fetch_ucirepo 
def load_wine_data():

    """
    Load wine data set from ucimlrepo
    You should have installl ucimlrepo
    """
    # fetch dataset 
    wine_quality = fetch_ucirepo(id=186) 

    # data (as pandas dataframes) 
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    df_wine = pd.concat([X,y],axis=1)

    dict_mapping = {5:0, 6:0, 8:1}
    df_wine = df_wine[df_wine['quality'].isin([5,6,8])].copy()
    df_wine.replace({'quality':dict_mapping},inplace=True)
    X_wine = df_wine.drop(['quality'],axis=1).to_numpy()
    y_wine = df_wine[['quality']].to_numpy().ravel()
    return X_wine,y_wine

