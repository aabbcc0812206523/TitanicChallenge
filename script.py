import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
#moyenne sur la colonne age
for column in train.columns:
    if train[column].dtype == "float64" or train[column].dtype == "int64" :
        moyenne = round(train[column].mean(), 1)
        train[column].fillna(moyenne, inplace=True)

#train['Age'] = train['Age'].astype(int)
#print(train)

#print(model.score(X_train, y_train))


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
print('Best model - avg:', 
      rf_model['CVScore'],
      '+/-', 
      rf_model['CVStd'])
print(train['Pclass'][0])
print(rf_models.best_estimator_)


# predictions
test['Survived'] = rf_best.predict(test[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch']])
test[['PassengerId', 'Survived']].to_csv('predictions/predictions.csv', index=False)


