import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# onehot and clear NaNs
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
#    data = data.fillna(0)   # mostly Cabin NaN kinda meangingless
    print("nulls test:" + str(data.isnull().any()))

    data['Embarked'].fillna(data['Embarked'].mode().item(), inplace = True)
    data['Age'].fillna(data['Age'].mean().item() , inplace = True)
    data['Fare'].fillna(data['Fare'].mode().item() , inplace = True)

    print("should have no nulls:" + str(data.isnull().any()))

    data['Embarked'] = pd.Categorical(data['Embarked'], categories=['C','Q','S'])
    data = pd.get_dummies(data, columns=['Sex','Embarked'])
    data['Name'] = data['Name'].apply(lambda x: len(x))

    data = data.drop(columns=['Ticket'])   # ticket seems meaningless, not sure about Cabin too..

    data['Cabin'].fillna(0, inplace = True)
    def getCabin(value):
        val_dict = {
            'A' : 6,
            'B' : 5,
            'C' : 4,
            'D' : 3,
            'E' : 2,
            'F' : 1,
            'T' : 1   ## Taking T same as F, taking it to be an error     
        }
        return val_dict.get(str(value)[0], 0)

    data['Cabin'] = data["Cabin"].apply(getCabin)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_scaled

# non-onehot version
def preprocess2(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(0)   # mostly Cabin NaN kinda meangingless
    # data['Embarked'] = pd.Categorical(data['Embarked'], categories=['C','Q','S'])
    # data['Sex'] = pd.Categorical(data['Sex'], categories=['male','female'])
    data['Embarked'] = data['Embarked'].astype(str)
    le_embarked = preprocessing.LabelEncoder().fit(['C','Q','S'])
    le_sex = preprocessing.LabelEncoder().fit(['male','female'])
    data['Embarked'] = le_embarked.transform(data['Embarked'])
    data['Sex'] = le_sex.transform(data['Sex'])

    # data = pd.get_dummies(data, columns=['Sex','Embarked'])
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
