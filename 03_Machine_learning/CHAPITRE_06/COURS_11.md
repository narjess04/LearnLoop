Partie B · Machine Learning spécialisé · Cours 11/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[ranking]]

Cours 11 — Recommender Systems & Learning to Rank
LearnLoop · Partie B — Machine Learning spécialisé · Cours 11/12

Niveau : Intermédiaire → Avancé · Durée estimée : 3–4 h

Dans le cours précédent, nous avons étudié la Survival Analysis pour prédire le temps avant un événement.

Nous allons maintenant découvrir une autre grande famille du Machine Learning : les systèmes de recommandation.

Comment Netflix choisit-il les films à proposer ? Comment Spotify recommande-t-il des morceaux ? Comment une plateforme e-commerce classe-t-elle des milliers de produits pour chaque utilisateur ?

Ces problèmes combinent deux objectifs :

## 1. Recommender Systems : identifier les éléments susceptibles d'intéresser un utilisateur.


## 2. Learning to Rank : organiser ces éléments dans l'ordre le plus pertinent.

Ce cours est particulièrement important pour les applications modernes de ML, les moteurs de recherche et les plateformes personnalisées.


## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Comprendre l'architecture d'un système de recommandation.
- Différencier Content-Based et Collaborative Filtering.
- Implémenter un système User-Based et Item-Based.
- Comprendre la factorisation matricielle et les embeddings.
- Exploiter le feedback implicite.
- Résoudre partiellement le problème du Cold Start.
- Comprendre les approches Pointwise, Pairwise et Listwise.
- Calculer Precision@K, Recall@K, MAP et NDCG.
- Construire un mini-système de recommandation en Python.
- Comprendre les enjeux de production : diversité, exploration, biais et monitoring.
## Section A du cours — Fondements des systèmes de recommandation

## 2. Qu'est-ce qu'un Recommender System ?

Un système de recommandation est un modèle qui sélectionne des éléments pertinents pour un utilisateur à partir d'un ensemble de candidats.

Les éléments peuvent être :
- Des films ou séries.
- Des chansons.
- Des produits.
- Des articles.
- Des cours.
- Des offres d'emploi.
- Des contenus pédagogiques.
Dans LearnLoop, nous pourrions par exemple recommander le prochain cours selon les connaissances, les difficultés et les objectifs de chaque apprenant.
Architecture générale
Pipeline de recommandation

Les systèmes industriels utilisent souvent plusieurs étapes afin de limiter le coût de calcul.

## Étape 1 — Candidate Generation

Sélectionner rapidement quelques centaines de candidats parmi des milliers ou des millions d'éléments.

## Étape 2 — Ranking

Attribuer un score de pertinence à chaque candidat.

## Étape 3 — Re-ranking

Réorganiser les résultats pour respecter des contraintes supplémentaires : diversité, disponibilité, nouveautés ou règles métier.


## 3. Les principales approches

Approche	Principe	Exemple
Popularity-Based	Recommander les éléments populaires	Top 10 des cours
Content-Based	Utiliser les caractéristiques des éléments	Cours similaires
Collaborative Filtering	Exploiter les interactions des utilisateurs	Utilisateurs similaires
Matrix Factorization	Apprendre des représentations latentes	Embeddings utilisateurs et items
Hybrid Recommender	Combiner plusieurs approches	Contenu + interactions
Learning to Rank	Optimiser l'ordre des résultats	Classement personnalisé

Nous allons étudier chacune des approches principales.

## Section B du cours — Content-Based Filtering

## 4. Recommandation basée sur le contenu

Le Content-Based Filtering recommande des éléments similaires à ceux qu'un utilisateur a déjà appréciés.

Supposons qu'un utilisateur ait terminé plusieurs cours de Machine Learning.

Nous pouvons lui recommander des cours portant sur des sujets proches.
Exemple
Cours	Catégorie	Niveau
Régression logistique	ML supervisé	Débutant
Random Forest	ML supervisé	Intermédiaire
XGBoost	ML supervisé	Avancé
CNN	Deep Learning	Intermédiaire
Transformers	Deep Learning	Avancé

Nous pouvons représenter chaque cours à partir de ses caractéristiques.

Puis calculer la similarité entre les représentations.

## 5. Similarité cosinus

La similarité cosinus mesure l'angle entre deux vecteurs.
\[
\boxed{
sim(A,B)=
\frac{A\cdot B}
{\|A\|\|B\|}
}
\]

Pour deux vecteurs non nuls, sa valeur est comprise entre -1 et 1.

Dans les représentations TF-IDF classiques, dont les composantes sont non négatives, elle se situe généralement entre 0 et 1.
Intuition
Deux cours décrits par des mots similaires auront des vecteurs proches.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Recommander les cours similaires

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Avantages
- Fonctionne sans interactions entre utilisateurs.
- Permet de recommander de nouveaux éléments dès que leur contenu est disponible.
- Relativement interprétable.
- Adapté aux catalogues avec descriptions riches.
Limites
- Risque de recommander toujours des éléments très similaires.
- Difficulté à découvrir des intérêts nouveaux.
- Dépend de la qualité des caractéristiques.
- Personnalisation parfois limitée.

## Section C du cours — Collaborative Filtering

## 6. Le principe du filtrage collaboratif

Le Collaborative Filtering repose sur une idée simple :

Les utilisateurs ayant eu des comportements similaires peuvent apprécier des éléments similaires.

Nous n'avons pas nécessairement besoin de connaître le contenu détaillé des éléments.

Nous exploitons les interactions.
Exemple de matrice utilisateurs-items
Interactions fictives — notes de 1 à 5
Utilisateur	ML	DL	GenAI	Python
Alice	5	4	—	5
Bob	5	5	—	4
Chloé	1	2	5	—
David	—	4	5	2

Le symbole « — » représente une interaction inconnue, et non une note de zéro.
Alice et Bob ont des préférences proches.

Si Bob apprécie un nouveau cours, celui-ci pourrait intéresser Alice.

Il existe deux variantes principales.


## 7. User-Based Collaborative Filtering

Nous recherchons les utilisateurs similaires à un utilisateur cible.

Puis nous utilisons leurs interactions pour générer des recommandations.
Étapes

## 1. Construire la matrice utilisateurs-items.


## 2. Calculer les similarités entre utilisateurs.


## 3. Identifier les voisins les plus proches.


## 4. Agréger leurs préférences.


## 5. Recommander les éléments non encore consommés.

Formule simplifiée
Pour l'utilisateur \(u\) et l'item \(i\) :
\[
\boxed{
\hat r_{u,i}
=
\frac{
\sum_{v\in N(u)}sim(u,v)r_{v,i}
}{
\sum_{v\in N(u)}|sim(u,v)|
}
}
\]

Cette formule suppose que le dénominateur est non nul et que les interactions des voisins sont disponibles.

Dans les systèmes de notation explicite, il est souvent préférable de centrer les notes autour de la moyenne de chaque utilisateur afin de tenir compte de leurs habitudes de notation.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Attention
Remplacer les valeurs manquantes par zéro peut biaiser la similarité lorsque les utilisateurs ont peu d'interactions.

Pour des systèmes plus avancés, nous pouvons utiliser :
- Des similarités calculées sur les éléments communs.
- Des méthodes de normalisation.
- Des modèles à facteurs latents.
- Des embeddings appris.


## 8. Item-Based Collaborative Filtering

Cette fois, nous recherchons les items similaires entre eux à partir des interactions des utilisateurs.

Par exemple :

Les utilisateurs qui suivent le cours Random Forest suivent également souvent XGBoost.

Nous pouvons donc recommander XGBoost à une personne qui a terminé Random Forest.
Différence
User-Based	Item-Based
Similarité entre utilisateurs	Similarité entre items
Dépend des profils utilisateurs	Dépend des relations entre items
Peut évoluer rapidement	Souvent plus stable
Utile pour petites communautés	Utile pour catalogues relativement stables

Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

La transposition .T permet de comparer les colonnes, donc les items.

## Section D du cours — Matrix Factorization

## 9. Pourquoi la factorisation matricielle ?

Les matrices utilisateurs-items sont généralement très creuses.

Un utilisateur n'interagit qu'avec une petite partie du catalogue.

Par exemple, une plateforme peut avoir :
- 100 000 utilisateurs.
- 20 000 cours.
- Quelques interactions par utilisateur.
La matrice complète contiendrait :
\[
100\,000\times20\,000=2\,000\,000\,000
\]

positions.

Mais la majorité de ces positions sont inconnues.

La factorisation matricielle cherche à représenter utilisateurs et items dans un espace latent de faible dimension.

## 10. Principe mathématique

Nous approximons la matrice des interactions :
\[
\boxed{R\approx UV^\top}
\]

où :
- \(R\in\mathbb R^{m\times n}\) : matrice utilisateurs-items.
- \(U\in\mathbb R^{m\times k}\) : embeddings utilisateurs.
- \(V\in\mathbb R^{n\times k}\) : embeddings items.
- \(k\) : nombre de facteurs latents.
La prédiction d'une note est :
\[
\boxed{\hat r_{u,i}=U_u^\top V_i}
\]

Visualisation

Que représentent les facteurs latents ?
Ils peuvent capturer des préférences non explicitement définies.

Par exemple :
- Intérêt pour les sujets techniques.
- Préférence pour les contenus courts.
- Niveau de difficulté préféré.
- Intérêt pour la théorie ou la pratique.
Ces dimensions ne sont pas nécessairement directement interprétables.

Elles sont apprises automatiquement à partir des interactions.


## 11. Fonction de perte de la factorisation

Pour les interactions observées \(\Omega\), une fonction de perte classique est :
\[
\boxed{
L=
\sum_{(u,i)\in\Omega}
(r_{u,i}-U_u^\top V_i)^2
+
\lambda
\left(
\|U\|_F^2+\|V\|_F^2
\right)
}
\]

Le premier terme mesure les erreurs de reconstruction.

Le second régularise les embeddings pour limiter le surapprentissage.
Hyperparamètres importants
- Dimension latente \(k\).
- Coefficient de régularisation \(\lambda\).
- Learning rate.
- Nombre d'itérations.
- Stratégie de sampling.


## 12. Implémentation avec SVD

Pour les notes explicites, nous pouvons utiliser une approche de factorisation matricielle.

La bibliothèque surprise propose notamment un algorithme appelé SVD, basé sur un modèle de facteurs latents.

~~~shell
python -m pip install scikit-surprise
~~~

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Ce très petit dataset illustre uniquement l'API. Il ne permet pas une évaluation statistique fiable.

Pour un système réel, il faut beaucoup plus d'interactions et une stratégie de validation adaptée.

## Section E du cours — Implicit Feedback

## 13. Feedback explicite vs implicite

Jusqu'ici, nous avons principalement utilisé des notes.

Mais de nombreuses plateformes ne disposent pas de notes explicites.

Elles enregistrent plutôt des comportements.
Feedback explicite
- Note de 1 à 5.
- Like ou dislike.
- Évaluation d'un cours.
Feedback implicite
- Clic.
- Temps de visionnage.
- Achat.
- Ajout aux favoris.
- Lecture d'un article.
- Complétion d'un cours.
Le feedback implicite est souvent plus abondant, mais plus ambigu.

Un utilisateur qui ne clique pas sur un cours ne le déteste pas nécessairement.

Peut-être ne l'a-t-il jamais vu.

C'est la distinction entre absence d'interaction et préférence négative.


## 14. Weighted Matrix Factorization pour feedback implicite

Une approche courante consiste à définir :
\[
p_{u,i}=
\begin{cases}
1,&r_{u,i}>0\\
0,&r_{u,i}=0
\end{cases}
\]

Puis une confiance :
\[
c_{u,i}=1+\alpha r_{u,i}
\]

La fonction de perte devient :
\[
\boxed{
L=
\sum_{u,i}
c_{u,i}
(p_{u,i}-U_u^\top V_i)^2
+
\lambda
\left(
\|U\|_F^2+\|V\|_F^2
\right)
}
\]

Cette formulation est associée à l'approche Implicit ALS.

Les interactions observées reçoivent davantage de poids, tandis que les interactions absentes ne sont pas simplement ignorées.
Implémentation Python

~~~shell
python -m pip install implicit scipy
~~~

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

## Section F du cours — Cold Start et systèmes hybrides

## 15. Le problème du Cold Start

Le Cold Start apparaît lorsqu'un système ne dispose pas de suffisamment d'informations pour personnaliser les recommandations.
Deux situations
User Cold Start : un nouvel utilisateur n'a encore aucune interaction.

Item Cold Start : un nouveau cours vient d'être ajouté au catalogue.
Solutions possibles
Problème	Solution
Nouvel utilisateur	Popularité, onboarding, préférences déclarées
Nouvel item	Content-Based Filtering
Peu d'interactions	Approche hybride
Catalogue évolutif	Embeddings de contenu
Nouveaux centres d'intérêt	Exploration contrôlée

Recommender hybride
Un système hybride combine plusieurs sources d'information.

Par exemple :
\[
\boxed{
Score(u,i)=
\alpha S_{\text{collab}}(u,i)
+
(1-\alpha)S_{\text{content}}(u,i)
}
\]

où :
\[
0\leq\alpha\leq1
\]

Les deux scores doivent être rendus comparables avant de les combiner.
Expérimentation interactive
Poids du Collaborative Filtering

## 50 %


Content-Based : 50 %

Collaborative : 50 %

1

Random Forest

0.800

2

XGBoost

0.800

3

Transformers

0.750

4

CNN

0.575

Scores fictifs normalisés pour illustrer la combinaison hybride. Modifie le poids pour voir évoluer le classement.

## Section G du cours — Learning to Rank

## 16. Qu'est-ce que le Learning to Rank ?

Jusqu'ici, nous avons cherché à identifier les items pertinents.

Mais un système de recommandation doit aussi les présenter dans le bon ordre.

Le Learning to Rank consiste à apprendre une fonction :
\[
\boxed{s=f(u,i,c)}
\]

où :
- \(u\) : utilisateur.
- \(i\) : item.
- \(c\) : contexte.
- \(s\) : score de classement.
Le modèle cherche à placer les éléments les plus pertinents en haut de la liste.
Trois approches principales
Pointwise

Score individuel

Prédire la pertinence de chaque item.

Pairwise

Comparaison par paires

Apprendre que A doit précéder B.

Listwise

Liste complète

Optimiser le classement global.


## 17. Pointwise Ranking

Dans l'approche Pointwise, chaque item est évalué indépendamment.

Le problème peut être formulé comme une régression :
\[
\hat r_{u,i}=f(X_{u,i})
\]

Ou comme une classification :
\[
P(\text{clic}=1\mid u,i)
\]

Exemple
Item	Score
Cours A	0,92
Cours B	0,75
Cours C	0,40

Nous trions les éléments par score décroissant.
Limite
Le modèle optimise une perte individuelle, pas nécessairement la qualité du classement final.


## 18. Pairwise Ranking

L'approche Pairwise apprend à comparer deux items.

Nous souhaitons :
\[
s(u,i)>s(u,j)
\]

lorsque l'utilisateur préfère \(i\) à \(j\).
Exemple
Si l'utilisateur préfère le cours XGBoost au cours Random Forest :
\[
s(u,\text{XGBoost})
>
s(u,\text{Random Forest})
\]

Pairwise Logistic Loss
Une fonction de perte possible est :
\[
\boxed{
L_{pair}=
\log\left(
1+\exp(-(s_i-s_j))
\right)
}
\]

Cette perte pénalise les situations où l'item préféré obtient un score inférieur.
Exemples d'algorithmes
- RankNet.
- Bayesian Personalized Ranking (BPR).
- Certaines variantes de LambdaRank.


## 19. Listwise Ranking

L'approche Listwise considère une liste entière d'items.

L'objectif est de produire un ordre global pertinent.

Elle est particulièrement adaptée aux métriques de classement comme NDCG.
Exemples
- ListNet.
- ListMLE.
- LambdaMART, qui utilise des gradients influencés par les variations de métriques de ranking.
LambdaMART est une méthode importante dans les moteurs de recherche et les systèmes de ranking tabulaire.

## Section H du cours — Métriques de recommandation

## 20. Pourquoi Accuracy et RMSE ne suffisent pas ?

Dans un système de recommandation, nous nous intéressons surtout aux premiers résultats.

Un utilisateur consulte rarement des centaines de recommandations.

Nous devons donc mesurer la pertinence du Top-K.

Les principales métriques sont :
- Precision@K.
- Recall@K.
- Average Precision et MAP.
- DCG et NDCG.
- Hit Rate@K.


## 21. Precision@K

Precision@K mesure la proportion d'items pertinents parmi les K premiers éléments recommandés.
\[
\boxed{
Precision@K=
\frac{\text{Items pertinents dans Top-K}}{K}
}
\]

Exemple
Nous recommandons cinq cours :
Position	Cours	Pertinent ?
1	XGBoost	Oui
2	CNN	Non
3	Random Forest	Oui
4	Transformers	Oui
5	SVM	Non

Nous avons trois cours pertinents parmi cinq.
\[
\boxed{Precision@5=\frac35=0.6}
\]


## 22. Recall@K

Recall@K mesure la proportion des items pertinents retrouvés dans le Top-K.
\[
\boxed{
Recall@K=
\frac{\text{Items pertinents dans Top-K}}
{\text{Nombre total d'items pertinents}}
}
\]

Si l'utilisateur possède six cours pertinents au total, et que trois figurent dans le Top-5 :
\[
\boxed{Recall@5=\frac36=0.5}
\]

Différence
Precision@K répond à :
Parmi les recommandations proposées, combien sont pertinentes ?

Recall@K répond à :
Parmi tous les éléments pertinents, combien avons-nous retrouvés ?


## 23. Average Precision et MAP

L'Average Precision récompense les classements qui placent les items pertinents tôt dans la liste.

Pour une liste binaire avec \(R\) items pertinents :
\[
\boxed{
AP=
\frac{1}{R}
\sum_{k=1}^{N}
Precision@k\cdot rel_k
}
\]

où :
\[
rel_k=
\begin{cases}
1,&\text{item pertinent}\\
0,&\text{sinon}
\end{cases}
\]

La Mean Average Precision est la moyenne des AP sur plusieurs utilisateurs ou requêtes.
\[
\boxed{
MAP=
\frac1U\sum_{u=1}^{U}AP_u
}
\]

Pour AP@K, il existe plusieurs conventions de normalisation. Il faut préciser celle utilisée pour comparer les résultats.


## 24. DCG et NDCG

Le Discounted Cumulative Gain (DCG) tient compte de la position des items.

Les éléments pertinents placés en haut de liste reçoivent davantage de poids.

Une formulation courante est :
\[
\boxed{
DCG@K=
\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
}
\]

où \(rel_i\) est le niveau de pertinence à la position \(i\).

Le NDCG normalise le DCG par le meilleur classement possible.
\[
\boxed{
NDCG@K=
\frac{DCG@K}{IDCG@K}
}
\]

Pour une liste avec un gain idéal positif :
\[
0\leq NDCG@K\leq1
\]

L’interaction au début de ce support permet d’explorer cet exemple.

Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

## Section I du cours — Mini-projet Learning to Rank

## 25. Construire un modèle de ranking avec LightGBM

Nous allons construire un petit exemple de Learning to Rank avec LightGBM.

Le dataset représente plusieurs utilisateurs et des cours candidats.

Chaque ligne correspond à un couple utilisateur-cours.

Les variables peuvent inclure :
- Similarité avec les cours déjà suivis.
- Popularité du cours.
- Niveau de difficulté.
- Taux de complétion historique.
- Score de pertinence.

## Étape 1 — Préparer les données


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 2 — Entraîner LambdaMART


~~~shell
python -m pip install lightgbm
~~~

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Pourquoi group est-il important ?
En Learning to Rank, les observations appartiennent à des groupes de classement.

Dans notre cas, chaque utilisateur constitue un groupe.

Le modèle apprend à comparer les items au sein de chaque groupe, et non à comparer arbitrairement des items de différents utilisateurs.

## Étape 3 — Prédire


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Important : ce dataset est volontairement minuscule et le code prédit sur les données d'entraînement uniquement pour illustrer l'API. Il ne permet pas de mesurer la généralisation. Un véritable benchmark doit séparer les utilisateurs ou les périodes selon le scénario d'utilisation, éviter toute fuite de données et évaluer les classements sur des groupes indépendants.


## 26. Évaluation offline vs online

Un système de recommandation peut obtenir un excellent NDCG offline sans améliorer l'expérience utilisateur en production.

Pourquoi ?

Parce que les interactions observées dépendent aussi des recommandations précédemment affichées.

Un item peu montré reçoit peu de clics, même s'il pourrait intéresser de nombreux utilisateurs.

C'est un exemple de Exposure Bias.
Offline Evaluation
Nous utilisons des interactions historiques.

Métriques courantes :
- Recall@K.
- Precision@K.
- NDCG@K.
- MAP@K.
- Coverage.
- Diversity.
Online Evaluation
Nous mesurons le comportement réel des utilisateurs.

Métriques possibles :
- Click-Through Rate.
- Taux de complétion.
- Taux de conversion.
- Engagement.
- Satisfaction.
- Rétention.
Les tests A/B sont souvent utilisés pour comparer deux stratégies de recommandation.

## Section J du cours — Production et bonnes pratiques

## 27. Diversité et exploration

Un moteur de recommandation ne doit pas nécessairement proposer uniquement les éléments les plus similaires.

Sinon, l'utilisateur risque de rester enfermé dans un ensemble très limité de contenus.

Nous pouvons intégrer :
Diversité
Proposer des éléments issus de différentes catégories.
Nouveauté
Mettre en avant des éléments que l'utilisateur ne connaît pas encore.
Sérendipité
Recommander des éléments inattendus mais pertinents.
Exploration
Tester occasionnellement des éléments moins connus afin d'apprendre davantage sur les préférences.
Exploitation
Privilégier les éléments dont la pertinence estimée est déjà élevée.

Le compromis exploration-exploitation est particulièrement important dans les systèmes adaptatifs.


## 28. Architecture de production


Monitoring recommandé
En production, nous devons surveiller :
- La qualité du ranking.
- La distribution des scores.
- Le taux de couverture du catalogue.
- La diversité des recommandations.
- La latence.
- Les utilisateurs sans historique.
- Les nouveaux items.
- Le drift des comportements.
- Les performances par segment.


## 29. Erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Considérer une absence de clic comme un dislike	Labels biaisés	Modéliser l'exposition
Évaluer uniquement avec RMSE	Ranking mal évalué	Utiliser NDCG, Recall@K
Ignorer le Cold Start	Mauvaises recommandations initiales	Ajouter du Content-Based
Mélanger passé et futur	Data leakage	Validation temporelle
Évaluer sur des items déjà vus	Résultats artificiels	Filtrer les interactions connues
Utiliser uniquement les items populaires	Faible diversité	Re-ranking
Ignorer les nouveaux items	Faible couverture	Exploration
Optimiser uniquement le CTR	Objectif métier incomplet	Mesurer satisfaction et rétention
Comparer des scores non normalisés	Combinaison hybride incohérente	Calibrer ou normaliser les scores
Ignorer la latence	Expérience dégradée	Pipeline multi-étapes

[[quiz]]


## 31. Synthèse du cours

Les concepts essentiels à retenir :
- Recommender Systems : sélectionner des éléments pertinents pour un utilisateur.
- Content-Based Filtering : recommandations basées sur les caractéristiques des items.
- Collaborative Filtering : recommandations basées sur les interactions collectives.
- User-Based / Item-Based : similarité entre utilisateurs ou entre items.
- Matrix Factorization : apprentissage de facteurs latents.
- Implicit Feedback : exploitation des clics, vues et autres comportements.
- Cold Start : difficulté à recommander sans historique.
- Hybrid Recommender : combinaison de plusieurs stratégies.
- Learning to Rank : optimisation de l'ordre des résultats.
- Pointwise, Pairwise, Listwise : trois grandes familles de ranking.
- Precision@K, Recall@K, MAP et NDCG : métriques adaptées aux recommandations.
- Production : équilibre entre pertinence, diversité, exploration et latence.
Prochain cours
Cours 12 — Data-Centric Machine Learning

Partie B · 05/05 — Dernier cours du module ML spécialisé

Nous allons étudier comment améliorer les performances d'un système ML en travaillant directement sur la qualité des données : Label Noise, audits de datasets, Active Learning, Weak Supervision, détection des erreurs d'annotation et stratégies de sélection des données.


## Notes sur l’implémentation et le protocole

Le lab annonce un gain exponentiel 2**rel-1. L’appel direct sklearn.ndcg_score aux pertinences brutes utilise un gain linéaire : pour reproduire le gain exponentiel, fournir 2**relevance-1. Les égalités de score sont départagées de façon stable dans l’exercice. Les requêtes de validation et de test sont distinctes ; les paramètres du ranker sont choisis sur validation. La factorisation NumPy ignore les notes inconnues, et l’ALS pondéré différencie préférence et confiance. Les intégrations Surprise et implicit des originaux restent des extensions locales optionnelles.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np

def ranking_metrics(relevance,scores,k=5,total_relevant=None):
    relevance=np.asarray(relevance,dtype=float);scores=np.asarray(scores,dtype=float)
    if relevance.ndim!=1 or not len(relevance) or scores.shape!=relevance.shape or not np.isfinite(relevance).all() or not np.isfinite(scores).all() or np.any(relevance<0) or not isinstance(k,int) or k<1:
        raise ValueError('Pertinences et scores finis alignés, pertinences >=0 et K entier positif requis.')
    order=np.argsort(-scores,kind='stable');limit=min(k,len(relevance));ranked=relevance[order[:limit]]
    binary=ranked>0;available=int(np.sum(relevance>0))
    total_relevant=available if total_relevant is None else total_relevant
    if total_relevant<available:raise ValueError('Total pertinent incompatible avec les candidats.')
    discounts=np.log2(np.arange(2,limit+2))
    dcg=np.sum(np.expm1(np.log(2)*ranked)/discounts)
    ideal=np.sort(relevance)[::-1][:limit];idcg=np.sum(np.expm1(np.log(2)*ideal)/discounts)
    precisions=np.cumsum(binary)/np.arange(1,limit+1)
    # AP@K normalisée par le total pertinent ; 0 quand aucune pertinence.
    return {'precision':float(binary.sum()/k),'recall':float(binary.sum()/total_relevant) if total_relevant else 0.,
            'AP':float(np.sum(precisions*binary)/total_relevant) if total_relevant else 0.,
            'NDCG':float(dcg/idcg) if idcg>0 else 0.}

def recommend_unseen(scores,seen,k=3):
    scores=np.asarray(scores,dtype=float);seen=np.asarray(seen,dtype=bool)
    if scores.ndim!=1 or scores.shape!=seen.shape or not np.isfinite(scores).all() or k<1:raise ValueError('Scores et masque alignés requis.')
    candidates=np.flatnonzero(~seen)
    return candidates[np.argsort(-scores[candidates],kind='stable')[:k]]

def explicit_factorization(ratings,factors=3,epochs=120,learning_rate=.015,regularization=.05):
    ratings=np.asarray(ratings,dtype=float)
    if ratings.ndim!=2 or not np.isfinite(ratings[~np.isnan(ratings)]).all():raise ValueError('Matrice de notes avec NaN pour inconnus requise.')
    rng=np.random.default_rng(42);U=rng.normal(0,.2,(ratings.shape[0],factors));V=rng.normal(0,.2,(ratings.shape[1],factors))
    observed=np.argwhere(~np.isnan(ratings))
    for epoch in range(epochs):
        for i,j in observed[rng.permutation(len(observed))]:
            old=U[i].copy();error=ratings[i,j]-old@V[j]
            U[i]+=learning_rate*(error*V[j]-regularization*old)
            V[j]+=learning_rate*(error*old-regularization*V[j])
    return U,V

def implicit_als(interactions,factors=3,alpha=10.,regularization=.1,iterations=15):
    interactions=np.asarray(interactions,dtype=float)
    if interactions.ndim!=2 or not np.isfinite(interactions).all() or np.any(interactions<0):raise ValueError('Comptages non négatifs finis requis.')
    rng=np.random.default_rng(42);U=rng.normal(0,.1,(interactions.shape[0],factors));V=rng.normal(0,.1,(interactions.shape[1],factors))
    eye=np.eye(factors);preference=(interactions>0).astype(float);confidence=1+alpha*interactions
    for iteration in range(iterations):
        for i in range(len(U)):
            c=confidence[i];U[i]=np.linalg.solve((V.T*c)@V+regularization*eye,V.T@(c*preference[i]))
        for j in range(len(V)):
            c=confidence[:,j];V[j]=np.linalg.solve((U.T*c)@U+regularization*eye,U.T@(c*preference[:,j]))
    return U,V

def run(plot=False):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from lightgbm import LGBMRanker
    courses=['Régression','Forêt','Boosting','CNN','Transformers']
    descriptions=['supervised classification linear model','supervised classification ensemble trees',
                  'supervised boosting decision trees','deep learning computer vision images','deep learning attention language models']
    text=TfidfVectorizer().fit_transform(descriptions);similarity=cosine_similarity(text)
    print('Contenu similaire à Forêt :',[courses[i] for i in recommend_unseen(similarity[1],np.arange(5)==1,2)])
    ratings=np.array([[5,4,np.nan,5],[5,5,np.nan,4],[1,2,5,np.nan],[np.nan,4,5,2.]])
    U,V=explicit_factorization(ratings);print('Notes prédites sur interactions inconnues :',(U@V.T)[np.isnan(ratings)])
    interactions=np.array([[3,1,0,0],[2,2,0,1],[0,0,4,3],[0,1,3,2.]])
    A,B=implicit_als(interactions);print('Items non consommés, utilisateur 0 :',recommend_unseen((A@B.T)[0],interactions[0]>0,3))
    rng=np.random.default_rng(42);rows=[]
    for query in range(90):
        for item in range(8):
            similarity,popularity,match=rng.random(3)
            relevance=int(np.clip(np.floor(2*similarity+match+.3*popularity+rng.normal(0,.2)),0,3))
            rows.append({'query':query,'similarity':similarity,'popularity':popularity,'match':match,'relevance':relevance})
    data=pd.DataFrame(rows).sort_values('query',kind='stable');features=['similarity','popularity','match']
    train=data[data['query']<54];val=data[(data['query']>=54)&(data['query']<72)];test=data[data['query']>=72]
    def fit(frame,trees):
        model=LGBMRanker(objective='lambdarank',n_estimators=trees,num_leaves=7,min_child_samples=5,random_state=42,verbosity=-1,n_jobs=1)
        model.fit(frame[features],frame['relevance'],group=frame.groupby('query',sort=False).size().to_numpy())
        return model
    def score(model,frame):
        scored=frame.copy();scored['score']=model.predict(frame[features])
        return np.mean([ranking_metrics(g['relevance'],g['score'],5)['NDCG'] for _,g in scored.groupby('query')])
    candidates=[(trees,score(fit(train,trees),val)) for trees in [30,80]]
    best=max(candidates,key=lambda pair:pair[1])[0];print('NDCG@5 exponentiel sur validation :',candidates)
    final=fit(pd.concat([train,val]),best);test_score=score(final,test)
    print('Test avec requêtes distinctes : NDCG@5',test_score)
    print('Convention : gain 2**rel-1 ; sklearn.ndcg_score utilise un gain linéaire si rel est fourni directement.')
    return {'test_NDCG':test_score,'trees':best}

if __name__=='__main__':run()
~~~

## Challenge Python

Compléter ranking_metrics et recommend_unseen. Utiliser le gain exponentiel 2**rel-1 pour NDCG, une normalisation AP@K par le total pertinent et exclure les items déjà consommés. Préciser la gestion des ex æquo et des requêtes sans pertinence.

[[assessment]]

[[interview]]
