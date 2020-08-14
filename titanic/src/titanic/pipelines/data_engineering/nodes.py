import pandas as pd


def preprocess_train(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for companies.
        Args:
            data: Source data.
        Returns:
            Preprocessed data.
    """

    data["Name"] = data["Name"].apply(lambda x: x.lower())
    return data

def preprocess_test(data: pd.DataFrame) -> pd.DataFrame:
    data["Name"] = data["Name"].str.upper()
    return data

def final_train(data: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    return data

def final_test(data: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    return data
