# Titanic Challenge
20 Avril 2018
Matthieu Brillaxis - Guilhem Canivet

### Compte rendu Exercice
On a d'abord essayé de rajouter de la donnée avec les données "Pclass" et "Embarked" dans nos test.
On a ensuite remplis les données null ou vide, on a remplacé les ages manquant en fonction des Pclass (Plus les classes étaient cher plus on mettait des âges élevés), et on a supprimé les Embarked manquant car il y en avait que 2.
On a créer une nouvelle donnée, la taille de la famille "FamilySize", en additionant "Parch", "Sib sp" et en ajoutant 1.


On a ensuite nettoyé les données :
- On a donc enlevé les champs "PassengerId", "Name", "Ticket" et "Cabin" que nous jugions pas pertinent pour nos tests.

On a testé deux models :
- Regression logistique (0.79 en local et 0.77 sur Kaggle)
- Arbre de décision (0.81 en local et 0.75 sur Kaggle)

On a remarqué qu'on avait un meilleur score sur nos machines avec l'arbre de décision , mais sur Kaggle on avait un meilleur score avec une regression logistique.
On a donc choisit d'utilisé le model de regression logistique.

On aurait pu rajouté d'autres données dans nos tests, ou on aurait pu tester d'autres modeles.

Nom du Kaggle utilisé : matthieubrillaxis

Site utilisé :
- http://www.data-mania.com/blog/logistic-regression-example-in-python/
- https://www.kaggle.com/battuzz94/a-guide-to-titanic-challenge
