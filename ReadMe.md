# TITANIC CHALLENGE

Profil sur Kaggle : rthoreau2

## Approche
Le but du Titanic Challenge est de prédire la survie ou non des différents passagers.
De ce fait nous avons déjà un indice sur la direction qu’il faut prendre, le fait qu’il y ait seulement deux issues possibles (0-1) nous dirige vers un modèle de classification.

## Etapes
Dans un premier temps, nous avons analysé les différentes données afin de prioriser certaines informations.
Les premières valeurs qui semblent être importantes sont l’âge et le sexe.
Or nous remarquons que la colonne « Age » comporte des valeurs manquantes, deux choix étaient donc possible : Supprimer la ligne (au prix d’une donnée), ou affecter une valeur en fonction des autres lignes.
Notre choix s’est porté sur l’affectation de valeur en utilisant la moyenne d’âge des personnes.

Par la suite nous avons souhaité ajouter des colonnes au dataset, les colonnes qui nous semblaient pertinentes étant Pclass (La classe où se situait la personne (1,2,3)), SibSp (frère et soeurs, et époux) et Parch (lien de parenté).

Après plusieurs modification de paramètres, nous nous sommes aperçu que le modèle Linéaire n’est pas optimal pour ce cas précis ( Score local : 0.78, Kaggle : 0.76). Nous avons essayé le modèle RandomForest Classifier. Le classifier est plus performant car il permet de calculer une classification en sortie (et pas une quantité).

Nous avons utilisé la fonction GridSearchCV pour nous aider à définir les meilleurs paramètres.

Le résultat fut plus probant avec un score local de 0.82 et un score kaggle de 0.78.
Pour améliorer notre score, nous avons essayé d’utiliser les autres données mais sans meilleur résultat.

Après plusieurs recherches sur sklearn et n’ayant plus d’idée afin de peaufiner ce score avec ce modèle, nous avons décidé de tester un autre modèle : SVC (Support Vector Classification).

Malgré plusieurs tests, le score ne s’est pas amélioré, c’est pourquoi nous avons décidé de rester sur le RandomForestClassifier.
