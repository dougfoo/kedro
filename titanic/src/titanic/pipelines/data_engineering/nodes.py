import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import preprocessing

# onehot and clear NaNs
def preprocess(data: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
    data['Embarked'].fillna(data['Embarked'].mode().item(), inplace = True)
    data['Age'].fillna(data[['Pclass','Sex','Age']].groupby(['Sex','Pclass']).transform(np.mean).iloc[:,0], inplace=True)
    data['Fare'].fillna(data[['Pclass','Sex','Fare']].groupby(['Sex','Pclass']).transform(np.mean).iloc[:,0], inplace=True)

    data['Embarked'] = pd.Categorical(data['Embarked'], categories=['C','Q','S'])
    data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    data['Title'] = pd.Categorical(data['Title'], categories=['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
       'Jonkheer', 'Dona'])

    lbl= LabelEncoder()
    lbl.fit(list(data['Title'].values)) 
    data['Title'] = lbl.transform(list(data['Title'].values))

    # data = pd.get_dummies(data, columns=['Title'])

    data = pd.get_dummies(data, columns=['Sex','Embarked'])
    data['Name'] = data['Name'].apply(lambda x: len(x))

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
    data['FamilySize'] = data["SibSp"] + data["Parch"] + 1
    data['BinnedAge'] = pd.cut(data['Age'], bins=5, labels=[1,2,3,4,5])
    data['QBinnedAge'] = pd.qcut(data['Age'], q=5, labels=[1,2,3,4,5])
    data['FarePerPerson']= data['Fare'] / data['FamilySize']

    data = data.drop(columns=['Ticket','Sex_female','Fare','Age','QBinnedAge','SibSp','Parch','Cabin','Name'])
#    data = data.drop(columns=['Ticket'])

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return (data, scaler)

def final_train(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed

def final_test(original: pd.DataFrame, processed: pd.DataFrame, refdata: pd.DataFrame) -> pd.DataFrame:    
    processed['PassengerId'] = original['PassengerId']   # reset passengerId
    return processed
