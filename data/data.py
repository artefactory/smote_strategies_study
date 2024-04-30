import os

import pandas as pd

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
        18,
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


####### GA4 ##########


def preprocess_ga4_7features(df):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Preprocessing
    df["browser"] = df["browser"].apply(lambda x: "Other" if x == "<Other>" else x)
    df["operating_system"] = df["operating_system"].apply(
        lambda x: "Other" if x == "<Other>" else x
    )

    # Converts dates into datetimes:
    df["start_feature_date"] = pd.to_datetime(df["start_feature_date"])
    df["end_feature_date"] = pd.to_datetime(df["end_feature_date"])

    # Sort samples in chrologic time.
    df = df.sort_values(by=["start_feature_date"], ascending=True)
    df.reset_index(drop=True, inplace=True)

    # build the meta dataframe
    meta_df = df[["user_pseudo_id"]].copy()
    meta_df["fold"] = 0  # futre colonne fold
    meta_df["keep_in_test"] = False

    X = df[
        [
            "total_events",
            "total_page_view",
            "total_add_to_cart",
            "number_of_sessions",
            "engaged_sessions",
            "bounce_rate",
            "total_purchase",
        ]
    ].copy()
    y = df[["target_label"]].copy()

    return X, y, meta_df


def load_ga4_data():
    """
    Load preprocessed GA4 diabates data set from data\dexternals folder.
    The name of the file shoulde be : data_ga4.csv.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "data_ga4.csv",
        )
    )
    X, y, meta_df = preprocess_ga4_7features(df=df)
    return X.to_numpy(), y.to_numpy().ravel(), meta_df


def load_df_positifs_ga4():
    """small docstring here."""
    df_GA4, meta_df_GA4 = load_ga4_data()
    df_positifs_GA4 = df_GA4[df_GA4["target_label"] == 1].copy().reset_index(drop=True)
    df_positifs_GA4.drop(["target_label"], axis=1, inplace=True)
    df_positifs_GA4
    return df_positifs_GA4
