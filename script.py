import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# chargement des données
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Ici on liste les noms qui peuvent représenter une certaine classe sociale
rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'] 
simple_titles = ["Mlle","Ms", "Mr", "Mme", "Miss", "Miss", "Mrs", "Master"]

# pre-processing
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)

# On remplace récupère uniquement le titre de la personne
train['Name'].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
train['Name'].replace(rare_titles, 1, inplace=True)
train['Name'].replace(simple_titles, 0, inplace=True)


test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)

test['Name'].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
test['Name'].replace(rare_titles, 1, inplace=True)
test['Name'].replace(simple_titles, 0, inplace=True)


y = train['Survived']
X = train[['Age', 'Sex', 'Name']]

# séparation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# entrainement d'une régression linéaire
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# predictions
test['Survived'] = model.predict(test[['Age', 'Sex', 'Name']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)
