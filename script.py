import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Différents modèles
def LogisticRegressionModel(X_train, X_test, y_train, y_test):
  model = LogisticRegression()
  model.fit(X_train, y_train)

  print('LogisticRegression -- Score test: {}'.format(model.score(X_test, y_test)))

def RandomForestClassifierModel(X_train, X_test, y_train, y_test):
  model = RandomForestClassifier()
  model.fit(X_train, y_train)

  print('RandomForestClassifier -- Score test: {}'.format(model.score(X_test, y_test)))

def SVCModel(X_train, X_test, y_train, y_test):
  model = SVC(C=1000, gamma=0.001, kernel='rbf')
  model.fit(X_train, y_train)

  print('SVC -- Score test: {}'.format(model.score(X_test, y_test)))

def predict(X_train, y_train, test):
  # model = SVC()
  model = LogisticRegression()
  model.fit(X_train, y_train)

  print('LogisticRegression -- Score test: {}'.format(model.score(X_test, y_test)))

  test['Survived'] = model.predict(test.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'Sex']))
  test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)

# Helper functions
def filter_family_size(x):
  if x == 1:
    return 'Solo'
  elif x < 4:
    return 'Small'
  else:
    return 'Big'

def get_title(x):
  y = x[x.find(',')+1:].replace('.', '').replace(',', '').strip().split(' ')
  if y[0] == 'the':    # Search for the countess
    title = y[1]
  else:
    title = y[0]
  return title

def filter_title(title, sex):
  if title in ['Countess', 'Dona', 'Lady', 'Jonkheer', 'Mme', 'Mlle', 'Ms', 'Capt', 'Col', 'Don', 'Sir', 'Major', 'Rev', 'Dr']:
    if sex:
      return 'Rare_male'
    else:
      return 'Rare_female'
  else:
    return title


# chargement des données
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Family Size
# Pour connaître la taille de la famille, on additionne les colonnes Parch et SibSp, et nous ajoutons 1
# Nous classons les familles en 3 catégories
for df in [train, test]:
  size = df['Parch'] + df['SibSp'] + 1
  df['FamilySize'] = size.apply(filter_family_size)

# pre-processing
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

# Nous remplaçons les valeurs nulles par S, qui est le plus fréquent.
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')

# Puis nous transformons les dummies
for df in [train, test]:
  df['Sex'] = df['Sex'].apply(lambda x : 1 if x == 'male' else 0)

train = pd.get_dummies(train, columns=['Pclass'])
test = pd.get_dummies(test, columns=['Pclass'])

train = pd.get_dummies(train, columns=['FamilySize'])
test = pd.get_dummies(test, columns=['FamilySize'])

train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

for df in [train, test]:
    df['NameLength'] = df['Name'].apply(lambda x : len(x))
    df['Title'] = df['Name'].apply(get_title)

for df in [train, test]:
  df['Title'] = df.apply(lambda x: filter_title(x['Title'], x['Sex']), axis=1)

train = pd.get_dummies(train, columns=['Title'])
test = pd.get_dummies(test, columns=['Title'])

y = train['Survived']
X = train.drop(columns=['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'Sex'])

# séparation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Test des différents modèles
LogisticRegressionModel(X_train, X_test, y_train, y_test)
RandomForestClassifierModel(X_train, X_test, y_train, y_test)
SVCModel(X_train, X_test, y_train, y_test)

# predict(X_train, y_train, test)
