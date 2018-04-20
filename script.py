import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import make_scorer, accuracy_score, f1_score

# chargement des données
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# pre-processing
#print(train.head)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
train['Pclass'] = pd.get_dummies(train['Pclass'], drop_first=True)

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
test['Pclass'] = pd.get_dummies(test['Pclass'], drop_first=True)


y = train['Survived']
X = train[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch']]

# séparation en datasets de train et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# entrainement d'une régression linéaire
#model = LogisticRegression()
#model.fit(X_train, y_train)
#model.score(X_test, y_test)

#print(model.score(X_test, y_test))

#NETTOYAGE DES DONNEES
#Pour chaque colonne d'entier ou de flottant
for column in train.columns:
    if train[column].dtype == "float64" or train[column].dtype == "int64" :
        #on calcul la moyenne sur la colonne
        moyenne = round(train[column].mean(), 1)
        #et on remplace les valeurs vides par cette moyenne
        train[column].fillna(moyenne, inplace=True)
#même chose pour le dataframe de test
for column in test.columns:
    if test[column].dtype == "float64" or test[column].dtype == "int64" :
        moyenne = round(test[column].mean(), 1)
        test[column].fillna(moyenne, inplace=True)

#Modèle RandomForest
#Après plusieurs tests, ces paramètres ont eu les meilleurs résultats
rf_params = {
    'n_estimators' :  [500],
    'max_features' : ['log2'],
    'criterion' : ['entropy'],
    'min_samples_split' :  [2],
    'min_samples_leaf' : [1],
    'random_state' : [4321],
    'max_depth': [5]
}

random_forest = RandomForestClassifier()
acc_scorer = make_scorer(accuracy_score)
rf_models = GridSearchCV(random_forest, rf_params, scoring=acc_scorer, n_jobs=-1)
rf_models = rf_models.fit(X_train, y_train)

rf_best = rf_models.best_estimator_
rf_best = rf_best.fit(X_train, y_train)

rf_model = {
    'Name' : 'Random forest', 
    'CVScore' : rf_models.best_score_, 
    'CVStd' : rf_models.cv_results_['std_test_score'][rf_models.best_index_],
    'Result_train' : rf_best.predict(X_train),
    'Result_test' : rf_best.predict(X_test),
    'Model' : rf_best
}
best_idx = rf_models.best_index_
print('Best model - avg:', rf_model['CVScore'], '+/-', rf_model['CVStd'])

#permet d'afficher pour chaque paramètre, la valeur donnant le meilleur résultat
#print(rf_models.best_estimator_)


# predictions
test['Survived'] = rf_best.predict(test[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)

#Modèle stacking
'''from sklearn.svm import SVC
from sklearn import preprocessing

train_test = pd.concat([X_train, X_test], ignore_index=True)
train_test_normalized = preprocessing.scale(train_test)
X_train_normalized = train_test_normalized[:len(X_train), :]
X_test_normalized = train_test_normalized[len(X_train):len(X_train) + len(X_test), :]
x_validation_normalized = train_test_normalized[len(X_train) + len(X_test):, :]


svm_params = {
    'C' : [0.3],
    'kernel' : ['rbf'],
    'tol' : [1e-3],
    'degree' : [2],
    'random_state' : [4321]
}
acc_scorer = make_scorer(accuracy_score)
svc = SVC()
svc_classifiers = GridSearchCV(svc, svm_params, scoring=acc_scorer, n_jobs=-1)
svc_classifiers = svc_classifiers.fit(X_train_normalized, y_train)

svc_best = svc_classifiers.best_estimator_
svc_best = svc_best.fit(X_train_normalized, y_train)

svc_model = {
    'Name' : 'SVC', 
    'CVScore' : svc_classifiers.best_score_, 
    'CVStd' : svc_classifiers.cv_results_['std_test_score'][svc_classifiers.best_index_],
    'Result_train' : svc_best.predict(X_train_normalized),
    'Result_test' : svc_best.predict(X_test_normalized),
    'Model' : svc_best
}
best_idx = svc_classifiers.best_index_
print('Best model - avg:', 
      svc_model['CVScore'], 
      '+/-', 
      svc_model['CVStd'])
print()
print(svc_classifiers.best_estimator_)'''


