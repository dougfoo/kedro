import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(0)   # mostly Cabin NaN kinda meangingless
    data['Embarked'] = pd.Categorical(data['Embarked'], categories=['0','C','Q','S'])

    data = pd.get_dummies(data, columns=['Sex','Embarked'])
    data['Name'] = data['Name'].apply(lambda x: len(x))
    data = data.drop(columns=['Ticket','Cabin'])   # ticket seems meaningless, not sure about Cabin too..

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_scaled

def final_train(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed

def final_test(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed
