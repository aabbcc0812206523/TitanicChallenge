import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Différents modèles
def LogisticRegressionModel(X_train, X_test, y_train, y_test):
  model = LogisticRegression()
  model.fit(X_train, y_train)

  print('LogisticRegression -- Score test: {}'.format(model.score(X_test, y_test)))

def RandomForestClassifierModel(X_train, X_test, y_train, y_test):
  model = RandomForestClassifier(n_estimators=100)
  model.fit(X_train, y_train)

  print('RandomForestClassifier -- Score test: {}'.format(model.score(X_test, y_test)))

def SVCModel(X_train, X_test, y_train, y_test):
  model = SVC(C=1000, gamma=0.001, kernel='rbf')
  model.fit(X_train, y_train)

  print('SVC -- Score test: {}'.format(model.score(X_test, y_test)))

def ExtraTreesClassifierModel(X_train, X_test, y_train, y_test):
  model = ExtraTreesClassifier(n_estimators=100)
  model.fit(X_train, y_train)

  print('ExtraTreesClassifier -- Score test: {}'.format(model.score(X_test, y_test)))

def GradientBoostingClassifierModel(X_train, X_test, y_train, y_test):
  model = GradientBoostingClassifier(n_estimators=100)
  model.fit(X_train, y_train)

  print('GradientBoostingClassifier -- Score test: {}'.format(model.score(X_test, y_test)))

# On utilise le meilleur modèle (ici la LogisticRegression) pour générer le fichier de validation Kaggle
def predict(X_train, y_train, test):
  model = LogisticRegression()
  model.fit(X_train, _train)

  print('LogisticRegression -- Score test: {}'.format(model.score(X_test, y_test)))

  test['Survived'] = model.predict(test.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'Sex', 'Ticket', 'TicketLetter']))
  test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)

# On filtre la taille de la famille en fonction du nombre de personnes dans la famille
def filter_family_size(x):
  if x == 1:
    return 'Solo'
  elif x < 4:
    return 'Small'
  else:
    return 'Big'

# Récupère le titre de la valeur d'entrée (ici le nom) et récupère le titre
def get_title(x):
  y = x[x.find(',')+1:].replace('.', '').replace(',', '').strip().split(' ')
  if y[0] == 'the':    # Search for the countess
    title = y[1]
  else:
    title = y[0]
  return title

# Filtre les titres pour avoir un peu moins de valeurs
def filter_title(title, sex):
  if title in ['Countess', 'Dona', 'Lady', 'Jonkheer', 'Mme', 'Mlle', 'Ms', 'Capt', 'Col', 'Don', 'Sir', 'Major', 'Rev', 'Dr']:
    if sex:
      return 'Rare_male'
    else:
      return 'Rare_female'
  else:
    return title

# Filtre les billets en plusieurs catégories
def filter_ticket(x):
  if x in ['9', '8', '5', 'L', '6', 'F', '7', '4', 'W', 'A']:
    return 'Rare'
  elif x in ['C', 'P', 'S', '1']:
    return 'Frequent'
  elif x == '2':
    return 'Common'
  elif x == '3':
    return 'Commonest'


# chargement des données
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Family Size
# Pour connaître la taille de la famille, on additionne les colonnes Parch et SibSp, et nous ajoutons 1
# Nous classons les familles en 3 catégories
for df in [train, test]:
  size = df['Parch'] + df['SibSp'] + 1
  df['FamilySize'] = size.apply(filter_family_size)

# Nous transformons les valeurs en string en valeurs numériques, 1 ou 0
for df in [train, test]:
  df['Sex'] = df['Sex'].apply(lambda x : 1 if x == 'male' else 0)

# On remplit l'age manquant par la moyenne des ages
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

# Nous remplaçons les valeurs nulles par S, qui est le plus fréquent.
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')

# Nous transformons les colonnes en dummies
train = pd.get_dummies(train, columns=['Pclass'])
test = pd.get_dummies(test, columns=['Pclass'])

train = pd.get_dummies(train, columns=['FamilySize'])
test = pd.get_dummies(test, columns=['FamilySize'])

train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

# Ici nous générons la colonne title en fonction du name
for df in [train, test]:
    df['NameLength'] = df['Name'].apply(lambda x : len(x))
    df['Title'] = df['Name'].apply(get_title)

# Ici nous filtrons la colonne title pour avoir un peu moins de valeurs
# Nous regroupons tous les titres "rares" sous un même groupe
for df in [train, test]:
  df['Title'] = df.apply(lambda x: filter_title(x['Title'], x['Sex']), axis=1)

train = pd.get_dummies(train, columns=['Title'])
test = pd.get_dummies(test, columns=['Title'])

# Nous récupérons les lettres des billets
for df in [train, test]:
    df['TicketLetter'] = df['Ticket'].apply(lambda x : str(x)[0])

# Puis nous regroupons les billets en catégories
for df in [train, test]:
    df['TicketCategory'] = df['TicketLetter'].apply(filter_ticket)

train = pd.get_dummies(train, columns=['TicketCategory'])
test = pd.get_dummies(test, columns=['TicketCategory'])

# Nous supprimons les données non utiles
formatted_X = train.drop(columns=['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'Sex', 'Ticket', 'TicketLetter'])
formatted_test = test.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'Sex', 'Ticket', 'TicketLetter'])

y = train['Survived']
X = formatted_X

# séparation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Test des différents modèles et affichage du score
LogisticRegressionModel(X_train, X_test, y_train, y_test)
RandomForestClassifierModel(X_train, X_test, y_train, y_test)
SVCModel(X_train, X_test, y_train, y_test)
ExtraTreesClassifierModel(X_train, X_test, y_train, y_test)
GradientBoostingClassifierModel(X_train, X_test, y_train, y_test)

# Génération du fichier de prédict
predict(X_train, y_train, formatted_test)
