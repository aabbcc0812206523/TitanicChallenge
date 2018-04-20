################
### Rapport ####
################

##SCORE KAGGLE: 0.77990 - Simon Milleto

# Utilisation de la variable Name
Le nom des passagers recèlent des informations importantes concernant leur classe sociale. Nous avons simplifié la vision globale des passagers en regrouppant les titres royaux dans une seul groupe ('Rare Title'), et en lassant le reste comme tel. Partant de là, nous avons fait des dummies. Il était en effet fort possible que les personnes issues des meilleurs millieux arrivaient a obtenir de meilleurs cannaux et a passer en premier, maximisant leurs chances de survie.

# Construction d'une nouvelle feature: Family Size.
La taille de la famille devait problablement jouer un rôle crucial dans la survie des passagers: les plus nombreux devaient avoir plus de mal a trouver des places dans les cannaux, et ne voulaient surement pas être séparés. Les moins nombreux / personnes seules, avaient plus de chance.

Pour construire cette variable, nous nous sommes basés sur deux features du dataSet: SibSp (nombre de frères / soeurs ) et Parch (parents / enfants). Nous avons fait l'addition : SibSp + Parch + 1, pour compter la personne elle même.

# Utilisation de la variable Fare (prix du ticket)
Le prix du ticket étant fortemment corrélé à la classe sociale (plus la classe est élevée, plus le ticket acheté est cher), il nous a semblé bon d'inclure cette variable dans notre modele.
Certaines données étant manquantes, nous avons pris la décision de remplir les trous avec la valeur médiane plutôt qu'enlever les lignes.

# Selection du meilleur modèle possible
Nous savions que les résultats étaient très différents selon les différents modèles utilisés (regression logistique, random forest classifier, random forest regressor..).
Nous avons donc créé un petit algorythme qui calculait le score de chacun de ces modèles et selectionnait le meilleur d'entre eux.
