import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# onehot and clear NaNs
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data['Embarked'].fillna(data['Embarked'].mode().item(), inplace = True)
    data['Age'].fillna(data['Age'].mean().item() , inplace = True)
    data['Fare'].fillna(data['Fare'].mode().item() , inplace = True)

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

    data = data.drop(columns=['Cabin','Sex_female'])   # ticket seems meaningless, not sure about Cabin too..

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data

def final_train(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed

def final_test(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed
