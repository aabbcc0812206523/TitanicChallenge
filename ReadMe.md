# Titanic Challenge
20 Avril 2018

### Principe
Il s'agit d'un challenge de Machine Learning (classification binaire) dans lequel le but est de prédire si un passager du Titanic a survécu ou non au naufrage à partir de caractéristiques comme son âge ou son sexe.  
La métrique permettant de mesurer la qualité du modèle est la précision (= pourcentage de bonnes prédictions).  

Le challenge, les données ainsi qu'un descriptif détaillé sont disponibles sur la plateforme Kaggle:  
https://www.kaggle.com/c/titanic
	
### Durée
3h (de 16h à 19h)

### Objectif
Vérifier votre capacité à:
- mettre en place un algorithme de Machine Learning le plus performant possible
- faire face à des problématiques data concrètes (nettoyage et traitement de la donnée, gestion des valeurs manquantes, feature engineering...etc)

### Déroulé
Un script de départ vous est remis en début de séance.
- il met en place un algorithme simple (régression linéaire)
- seules 2 features sont exploitées (âge et sexe)  

Charge à vous d'ajouter des features et d'utiliser des modèles plus complexes afin d'obtenir un meilleur score.
Vous pouvez pour cela utiliser toutes les ressources à votre disposition (internet, forum Kaggle...etc).  
Chaque groupe clone le repo et apporte au script les modifications qu'il souhaite.  
	
### Attendus
- le code (commenté)
- les prédictions qui ont été envoyées à la plateforme (dans le dossier predictions/)
- un compte rendu détaillant ce qui a été fait et citant les ressources utilisées (à la place du ReadMe.md)

Dans une nouvelle branche du repo actuel (1 branche par groupe)

### Critères de notation
- qualité de la démarche (cohérence des choix qui ont été faits)
- qualité du compte-rendu
- capacité à aller chercher des informations (sur le net, sur le forum de Kaggle...etc)
- performances au challenge


## Compte rendu

### Analyse des données
Compréhension des données en utilisant le [Data Dictionary](https://www.kaggle.com/c/titanic/data). On a essayé de définir les features qui pouvaient sembler utile, puis on a sorti les valeurs inutiles. Soit parce qu'il y avait trop de données manquantes, soit parce que les données ne semblaient pas avoir un grand intérêt dans la prédiction.

### Agrégation des données
Nous avons commencé par traiter les données faciles à traiter, c'est à dire les features "Pclass" et "Embarked" qui ont juste besoin d'être transformé en dummies, pour le "Embarked" on a juste du remplacer les valeurs nulles par la valeur la plus fréquente qui est "S".

Concernant les features "Parch" et "SibSp", il nous a semblé pertinent de les utiliser pour générer une nouvelle feature, la taille de la famille. Nous nous sommes inspirés du code rédigé sur ce [Kaggle](https://www.kaggle.com/battuzz94/a-guide-to-titanic-challenge?scriptVersionId=1059478)

Les noms des passagers peuvent nous être utiles : Ils contiennent des titres tels que Mr, ou Countess, qui peuvent être utiles. En effet, nous savons que les personnes les plus aisées sont celles qui ont le plus survécu au naufrage. Nous avons extrait ces titres des noms, avant de les transformer en dummies.

Nous avons essayé d'autres features, mais qui n'ont pas eu réellement d'impact sur le score, comme la catégorisation des billets.

### Choix du modèle
Pour le choix du modèle utilisé, nous avons crée une fonction pour chaque modèle différent en passant les paramètres qu'on souhaite "fit". Dans chaque modèle on affiche le score afin de voir quel modèle est les plus efficace. Dans un premier temps nous avons testé les modèles LogisticRegression, RandomForestClassifier et SVC après avoir analysé les modèles les plus performants sur plusieurs posts Kaggle [RandomForestClassifier](https://www.kaggle.com/zhenqiliu/titanic-survival-python-solution) et [SVC](https://www.kaggle.com/battuzz94/a-guide-to-titanic-challenge?scriptVersionId=1059478)

Nous avons remarqué que sans ajouter de features (donc seulement Sex et Age), le modèle Regression Logistique restait le plus efficace. C'est en ajoutant des features qu'on a commencé à voir des améliorations avec le modèle SVC (0,81 vs 0,79).

### Optimisation des paramètres
Nous avons essayé plusieurs techniques d'optimisation des hyper-paramètres : GridSearch et RandomSearch, mais les résultats ne variant pas, voir baissant par rapport à une utilisation de la fonction par défaut, nous sommes revenus à ceux-ci.

### Pistes d'amélioration
Pour améliorer le score de notre prédiction, nous aurions pu essayer plus de modèles, ou les aggréger pour tirer le meilleur de chacuns d'entre-eux. La sélection des hyper-paramètres à l'aide de GridSearch ou RandomSearch est également une voie à explorer, mais qui nécessite du temps et beaucoup de puissance de calcul pour avoir des améliorations significatives de la prédiction.
