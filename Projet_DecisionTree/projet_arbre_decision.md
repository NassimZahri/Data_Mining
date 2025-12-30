# Projet : Arbres de décision, extensions et applications

## Contexte

Les arbres de décision constituent une famille de modèles de classification supervisée largement utilisés en data mining et en apprentissage automatique. Ils permettent de représenter un modèle sous forme de règles compréhensibles, et servent de base à des méthodes plus avancées comme les forêts aléatoires ou le boosting.

L'objectif de ce projet est de passer des notions théoriques vues en cours (impureté, gain, conditions de test, types d'attributs, etc.) à une étude pratique et expérimentale, en implémentant un mini-classificateur par arbre de décision, puis en le comparant à ses extensions modernes sur des jeux de données réels.

## Objectifs pédagogiques

À l'issue du projet, l'étudiant(e) devra être capable de :

- Formaliser un problème de classification supervisée à partir d'un cas réel
- Implémenter les briques de base d'un arbre de décision (impuretés, choix de split, structure récursive)
- Analyser les effets de la profondeur, de la taille minimale de nœud, etc. sur le sur-apprentissage
- Utiliser et comparer des extensions : élagage, forêts aléatoires, éventuellement boosting
- Appliquer ces méthodes à un jeu de données réel (crédit, santé, churn, etc.) et interpréter les règles obtenues
- Présenter les résultats sous forme de rapport structuré et de notebooks reproductibles

## Organisation du projet

Le projet est structuré en quatre parties. Les parties 1 et 2 se concentrent sur la compréhension et l'implémentation, les parties 3 et 4 sur les extensions et les applications.

---

## Partie 1 – Révision des concepts et expériences numériques (théorie + Python)

### 1.1 Rappels théoriques (1-2 pages)

Rappeler brièvement les notions suivantes :

- Classification supervisée (définition, couple (x, y), notion de classe)
- Principe général d'un arbre de décision (nœuds internes, feuilles, parcours)
- Mesures d'impureté d'un nœud de décision : indice de Gini, entropie, erreur de classification, avec leurs formules mathématiques

### 1.2 Exemples numériques d'impureté

Proposer plusieurs exemples numériques d'impureté :

- Cas au moins équilibré (par ex. 10 positifs / 10 négatifs)
- Cas très pur (par ex. 18 positifs / 2 négatifs)
- Ajouter au moins trois cas supplémentaires (par ex. 9/1, 5/5, 1/9)
- Comparer Gini, entropie et erreur de classification pour chaque cas

### 1.3 Implémentation Python

Implémenter, dans un notebook Python, des fonctions :

- `gini(counts)`
- `entropy(counts)`
- `classification_error(counts)`
- Illustrer leur comportement sur vos exemples numériques à l'aide d'un petit tableau ou graphique

---

## Partie 2 – Implémentation d'un mini-arbre de décision « from scratch »

### 2.1 Choix du jeu de données

Choisir un petit jeu de données pédagogique, par exemple :

- Un dataset de crédit simplifié (quelques attributs : propriétaire, état matrimonial, revenu, défaut)
- Ou un autre dataset binaire de petite taille choisi par le groupe

### 2.2 Éléments à implémenter

Implémenter les éléments suivants dans un notebook :

- Une structure de nœud (class Python ou dictionnaire) contenant le type de nœud (interne/feuille), l'attribut de split, le seuil ou la partition, les sous-arbres enfants
- Une fonction de recherche du « meilleur split » sur un nœud donné :
  - Pour les attributs catégoriels : splits binaires ou multi-branches
  - Pour les attributs continus : test de seuils candidats (tri + balayage)
- Le calcul du gain d'impureté : `Gain = P - M`, où P est l'impureté du nœud parent et M l'impureté pondérée des enfants

### 2.3 Construction récursive de l'arbre

Construire l'arbre récursivement jusqu'à une condition d'arrêt :

- Nœud pur (toutes les instances de la même classe)
- Profondeur maximale atteinte (`max_depth`)
- Nombre minimal d'exemples dans un nœud (`min_samples_leaf`)

### 2.4 Fonction de prédiction

Implémenter une fonction de prédiction `predict_one(x)` qui classe un nouvel exemple en parcourant l'arbre.

### 2.5 Comparaison avec sklearn

Comparer les performances de votre mini-arbre avec celles de `sklearn.tree.DecisionTreeClassifier` sur le même jeu de données (taux de bonne classification, matrice de confusion).

---

## Partie 3 – Extensions : sur-apprentissage, élagage et forêts aléatoires

### 3.1 Sélection du jeu de données

Sélectionner au moins un jeu de données réel (par exemple, un dataset de breast cancer, wine, crédit, churn, etc.).

### 3.2 Division des données

Diviser le dataset en apprentissage et test (par exemple, 70% / 30%).

### 3.3 Entraînement avec paramètres variés

Entraîner plusieurs arbres de décision avec des paramètres différents :

- `max_depth` (par exemple 2, 3, 4, 5, None)
- Éventuellement `min_samples_leaf` (1, 5, 10)

### 3.4 Analyse des performances

Pour chaque configuration :

- Mesurer la performance sur l'ensemble d'apprentissage et sur l'ensemble de test (accuracy, F1-score, etc.)
- Tracer des courbes montrant l'évolution des performances en fonction de la profondeur
- Commenter les phénomènes de sous-apprentissage et sur-apprentissage

### 3.5 Forêts aléatoires

Comparer avec des forêts aléatoires :

- Entraîner un `RandomForestClassifier` avec plusieurs valeurs de `n_estimators` (par ex. 10, 50, 100)
- Comparer les performances et la stabilité par rapport à un arbre unique

### 3.6 Boosting (Optionnel)

Tester une méthode de boosting (par exemple, `AdaBoostClassifier`) et analyser si elle améliore les résultats.

---

## Partie 4 – Application métier et interprétation

### 4.1 Description du domaine d'application

Choisir un domaine d'application (crédit, santé, spam, churn, etc.) et décrire clairement :

- La problématique métier
- Les variables explicatives disponibles (features)
- La variable cible

### 4.2 Entraînement du modèle final

Entraîner un modèle final (arbre ou forêt aléatoire) adapté à ce domaine.

### 4.3 Extraction des règles de décision

Extraire et présenter quelques règles de décision représentatives sous forme lisible pour un non-expert.

### 4.4 Discussion

Discuter :

- L'interprétabilité du modèle (arbres vs forêts)
- Les limites observées (zones d'erreur, biais éventuels, données manquantes, etc.)

---

## Livrables

Chaque groupe doit fournir :

- Un rapport écrit (10-15 pages) au format PDF, structuré et rédigé en bon français
- Un ou plusieurs notebooks (Jupyter ou Colab) contenant le code, les figures et les commentaires

---

## Barème indicatif

- **20%** – Compréhension théorique et clarté des explications (Partie 1)
- **30%** – Implémentation du mini-arbre et qualité du code (Partie 2)
- **25%** – Étude expérimentale des extensions, courbes et analyse (Partie 3)
- **15%** – Application métier, interprétation et discussion (Partie 4)
- **10%** – Qualité globale du rapport et des notebooks (présentation, structure, lisibilité)

---

## Remarque importante

Toute forme de plagiat (copie de code ou de texte sans référence) sera pénalisée selon le règlement de l'établissement.