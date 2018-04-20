import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# chargement des donnees
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)

# pre-processing

# On remplace les ages null par des ages correspondant
# par 24 pour la 3 eme classe car les plus jeunes ont plus de chance de prendre la 3 eme classe
# par 37 pour la premiere classe car les plus agées ont plus de chance de prendre la premiere classe
# et par 34 pour la 2 eme classe
def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(age_approx, axis=1)

# On supprime les donnees null restante comme il y en a que 2 on n'en perd pas trop
train.dropna(inplace=True)

# pre-processing
train['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Embarked']=pd.get_dummies(test['Embarked'],drop_first=True)
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
y = train['Survived']
X = train[['Age', 'Sex', 'Pclass']]

# separation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# entrainement d'une regression lineaire
model = DecisionTreeClassifier(max_depth=3, random_state=100)
model = model.fit(X, y)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = model.fit(X, y)

# predictions
test['Survived'] = model.predict(test[['Age', 'Sex', 'Pclass']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)

print(model.score(X_test, y_test))