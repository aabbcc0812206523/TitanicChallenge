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


### Compte rendu Exercice
On a d'abord essayé de rajouter de la donnée avec les données "Pclass" et "Embarked"
Vu que les gens ayant acheté des tickets + cher étaient donc mieux logé sur le bateau

On a ensuite nettoyé les données :
- On a donc enlevé les champs "PassengerId", "Name", "Ticket" et "Cabin" que nous jugions pas pertinent

On a nettoyé les données nulles du dataset.
Pour les ages on les a complété par des âges fixé arbitrairement en fonction du Pclass.
Plus les personnes sont mieux logés plus ils sont probablement âgé.

On a aussi rajouté les données d'embarquement à notre prediction.

Puis on a fait plusieurs test avec différents model.
On a remplacé le model lineaire par un arbre de decision avec un max_depth de 3.
Ce qui a augmenté notre score.

