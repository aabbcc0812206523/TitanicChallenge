import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# chargement des donnÃ©es
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# pre-processing
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
y = train['Survived']
X = train[['Age', 'Sex']]

# sÃ©paration en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# entrainement d'une rÃ©gression linÃ©aire
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# predictions
test['Survived'] = model.predict(test[['Age', 'Sex']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)
