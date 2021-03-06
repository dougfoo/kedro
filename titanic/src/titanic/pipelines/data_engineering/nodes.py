import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import preprocessing

# onehot and clear NaNs
def preprocess(data: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):    
    data['CabinNull'] = data['Cabin'].isnull()
    data['AgeNull' ] = data['Age'].isnull()
    # data['FareNull'] = data['Fare'].isnull()
    # data['EmbarkedNull'] = data['Embarked'].isnull()

    data['Age'].fillna(data[['Pclass','Sex','Age']].groupby(['Sex','Pclass']).transform(np.mean).iloc[:,0], inplace=True)
    data['Fare'].fillna(data[['Pclass','Sex','Fare']].groupby(['Sex','Pclass']).transform(np.mean).iloc[:,0], inplace=True)
#    data['Embarked'].fillna(data['Embarked'].mode().item(), inplace = True)
    data['FamilySize'] = data["SibSp"] + data["Parch"] + 1
    data['BinnedFamilySize'] = pd.cut(data['FamilySize'], bins= [0, 3, 5, 8, 16], labels=[1,2,3,4])
    data['BinnedAge'] = pd.cut(data['Age'], bins= [0, 8, 16, 32, 64, 128], labels=[1,2,3,4,5])
    data['QBinnedAge'] = pd.qcut(data['Age'], q=5, labels=[1,2,3,4,5])
    portmap = data[['Pclass','Sex','Embarked','QBinnedAge','Fare']].groupby(['Sex','Pclass','QBinnedAge']).apply(
            lambda x: x.sort_values('Fare').head(1)['Embarked'].values[0])
    portmap = portmap.rename('NewEmb').reset_index()
    data2 = pd.merge(how='inner',left=data, right=portmap)
    print(data2[data2.Embarked.isnull()])
    # alot of work for filling 2 null values...
    data3 = data2.copy()
    data3['Embarked'] = data3['Embarked'].fillna(data3['NewEmb'])
    data = data3.drop(columns=['NewEmb'])

    data['Embarked'] = pd.Categorical(data['Embarked'], categories=['C','Q','S'])
    data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    data['Title'] = pd.Categorical(data['Title'], categories=['Mr','Col','Capt', 'Master','Don','Rev','Sir','Dr','Major', 
        'Mme','Jonkheer','Dona','Ms','Lady', 'Mrs', 'Miss', 'Mlle', 'the Countess'])

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
    data['FarePerPerson']= data['Fare'] / data['FamilySize']

    data = data.drop(columns=['Ticket','Sex_female','Fare','Age','QBinnedAge','SibSp','Parch','Name','FamilySize'])
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
