import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# chargement des données
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Ici on liste les noms qui peuvent représenter une certaine classe sociale
rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'] 
simple_titles = ["Mlle","Ms", "Mr", "Mme", "Miss", "Miss", "Mrs", "Master"]

# pre-processing
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
# On récupère le prix du ticket en insérant la médiane du prix pour les champs vide
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
# On construit la variable FamilySize en prenant les parents + enfants + frere/soeur + la personne elle meme
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

# On remplace récupère uniquement le titre de la personne
train['Name'].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
# On remplace tous les titres rare par une meme variable et on fait les dummies
train['Name'].replace(rare_titles, "Rare title", inplace=True)
train['Name'].replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)
train['Name'] = pd.get_dummies(train['Name'])


#################### TEST
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1

test['Name'].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
test['Name'].replace(rare_titles, "Rare title", inplace=True)
test['Name'].replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)
test['Name'] = pd.get_dummies(test['Name'])


y = train['Survived']
X = train[['Age', 'Sex', 'Name', 'Fare', 'FamilySize']]

# séparation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


models = [
    RandomForestRegressor(n_estimators=20),
    RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
    LogisticRegression(),
    RandomForestClassifier(n_estimators=20),
    RandomForestClassifier(max_depth=2, random_state=0, n_estimators=100),
    RandomForestClassifier(max_depth=10, n_estimators=20),
    RandomForestClassifier(max_depth=20, n_estimators=50)

]

best_score = 0
best_model = LogisticRegression()

# On test chaque model pour prendre le meilleur
for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model
    print(score)
    
 
print("best score : ", best_score)
# predictions
test['Survived'] = best_model.predict(test[['Age', 'Sex', 'Name', 'Fare', 'FamilySize']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)
