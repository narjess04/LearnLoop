Partie B · Machine Learning spécialisé · Cours 12/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[datacentric]]

Cours 12 — Data-Centric Machine Learning
LearnLoop · Partie B — Machine Learning spécialisé · Cours 12/12

Niveau : Intermédiaire → Avancé · Durée estimée : 3–4 h

Dans les cours précédents, nous avons appris à entraîner, combiner, évaluer et optimiser des modèles de Machine Learning.

Mais une question fondamentale reste à explorer :

Et si, pour améliorer un modèle, la meilleure solution n'était pas de changer d'algorithme, mais d'améliorer ses données ?

C'est l'idée centrale du Data-Centric Machine Learning.

Un modèle sophistiqué entraîné sur des données incorrectes, incohérentes ou mal annotées peut obtenir de moins bons résultats qu'un modèle simple entraîné sur des données fiables.

Dans ce dernier cours du module ML spécialisé, nous allons découvrir comment auditer, nettoyer, annoter et sélectionner les données de manière méthodique.


## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Distinguer les approches Model-Centric et Data-Centric.
- Identifier les principales erreurs d'annotation.
- Comprendre le Label Noise et ses conséquences.
- Détecter des labels potentiellement incorrects.
- Réaliser un audit de qualité des données ML.
- Comprendre l'Active Learning et ses stratégies.
- Utiliser le Weak Supervision pour produire des labels à moindre coût.
- Comprendre les risques liés aux données synthétiques.
- Construire une boucle d'amélioration continue des données.
- Évaluer les gains réels d'une démarche Data-Centric.
## Section A du cours — Comprendre le Data-Centric ML

## 2. Model-Centric vs Data-Centric

Traditionnellement, une grande partie du travail en Machine Learning consiste à améliorer les algorithmes.

Nous essayons par exemple :
- Logistic Regression.
- Random Forest.
- XGBoost.
- Réseaux de neurones.
- Optimisation des hyperparamètres.
- Ensembles de modèles.
Cette approche est appelée Model-Centric.

Le Data-Centric ML adopte une perspective complémentaire : améliorer systématiquement les données utilisées pour entraîner et évaluer les modèles.
Model-Centric	Data-Centric
Modifier les algorithmes	Améliorer les données
Optimiser les hyperparamètres	Corriger les annotations
Changer l'architecture	Identifier les exemples ambigus
Ajouter des couches	Améliorer la couverture des données
Optimiser la fonction de perte	Améliorer la cohérence des labels
Comparer les modèles	Comparer les versions des datasets

Ces deux approches ne sont pas opposées. Un système performant combine généralement les deux.
Exemple
Imaginons un classifieur de documents.

Le modèle confond régulièrement les catégories « Facture » et « Bon de commande ».

Deux solutions sont possibles.

Approche Model-Centric : essayer un Transformer plus puissant.

Approche Data-Centric : vérifier si les exemples de factures et de bons de commande ont été correctement annotés, si les classes sont clairement définies et si les documents difficiles sont suffisamment représentés.

Dans certains cas, corriger les données apporte davantage qu'un changement de modèle.


## 3. Le cycle de vie Data-Centric

Boucle d'amélioration continue des données
Une démarche Data-Centric est itérative : chaque amélioration doit être mesurée sur des données d'évaluation indépendantes.
L'objectif n'est pas de nettoyer une seule fois le dataset, mais de mettre en place un processus reproductible.

## Section B du cours — Label Noise

## 4. Qu'est-ce que le Label Noise ?

Le Label Noise désigne les erreurs ou incohérences dans les étiquettes utilisées pour entraîner un modèle supervisé.

Exemple :
Document	Label enregistré	Label correct
Facture client	Facture	Facture
Bon de commande	Facture	Bon de commande
Contrat	Contrat	Contrat
Facture fournisseur	Bon de commande	Facture

Le modèle apprend à partir des labels enregistrés, même lorsqu'ils sont incorrects.

Il peut donc apprendre des relations trompeuses.
Notation
Nous distinguons :
\[
Y^*=\text{label réel}
\]

\[
\tilde Y=\text{label observé}
\]

Le Label Noise apparaît lorsque :
\[
\boxed{P(\tilde Y\neq Y^*)>0}
\]

Types de Label Noise
Random Label Noise : les erreurs apparaissent de manière aléatoire.

Class-Conditional Noise : la probabilité d'erreur dépend de la classe.

Instance-Dependent Noise : la probabilité d'erreur dépend des caractéristiques de l'observation.

Le troisième cas est particulièrement difficile.

Par exemple, les documents très flous peuvent être plus fréquemment mal annotés que les documents lisibles.


## 5. Matrice de transition du bruit

Dans une classification multiclasse, nous pouvons représenter les erreurs d'annotation par une matrice.
\[
\boxed{
T_{ij}=P(\tilde Y=j\mid Y^*=i)
}
\]

Chaque ligne représente une classe réelle et chaque colonne un label observé.
Exemple
Matrice de transition fictive
Réel / Observé	Facture	Commande	Contrat
Facture	90 %	8 %	2 %
Commande	12 %	85 %	3 %
Contrat	2 %	3 %	95 %

Exemple illustratif. Chaque ligne totalise 100 %.
Dans cet exemple, 12 % des vrais bons de commande sont annotés comme des factures.

Le modèle peut donc avoir du mal à distinguer ces deux catégories.


## 6. Pourquoi le Label Noise est-il dangereux ?

Les conséquences dépendent du type de bruit, du modèle et de la fonction de perte.

Il peut notamment :
- Réduire les performances de généralisation.
- Perturber la calibration des probabilités.
- Encourager le surapprentissage des annotations incorrectes.
- Détériorer les performances des classes minoritaires.
- Rendre les évaluations trompeuses si le test contient lui aussi des erreurs.
L’interaction au début de ce support permet d’explorer cet exemple.

Il faut distinguer la proportion de labels erronés et la perte de performance du modèle : leur relation n'est pas nécessairement linéaire.

## Section C du cours — Détection des erreurs d'annotation

## 7. Comment identifier les labels suspects ?

Une approche naïve consiste à chercher les observations pour lesquelles le modèle prédit une classe différente du label enregistré.

Mais cela ne suffit pas.

Le modèle peut lui-même se tromper.

Nous devons donc considérer les désaccords comme des indices à examiner, et non comme des preuves d'erreur.
Exemple
Un document est annoté « Facture ».

Le modèle prédit :
\[
P(\text{Facture})=0.05
\]

\[
P(\text{Commande})=0.92
\]

\[
P(\text{Contrat})=0.03
\]

Cette observation mérite probablement une vérification.

Mais le label ne doit pas être modifié automatiquement.


## 8. Prédictions Out-of-Fold

Pour identifier des labels suspects, il est préférable d'utiliser des prédictions produites par des modèles qui n'ont pas été entraînés sur les observations évaluées.

C'est le principe des prédictions Out-of-Fold (OOF).
Processus

## 1. Diviser le dataset en plusieurs folds.


## 2. Entraîner un modèle sur les autres folds.


## 3. Prédire les probabilités du fold exclu.


## 4. Répéter pour tous les folds.


## 5. Analyser les désaccords entre labels et prédictions.

Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Ce code utilise un dataset synthétique dans lequel nous connaissons les erreurs injectées.

Dans un véritable projet, les labels corrects ne sont généralement pas disponibles.


## 9. Confident Learning

Le Confident Learning est une famille de méthodes visant notamment à identifier les erreurs potentielles d'annotation à partir des probabilités prédites et des labels observés.

La bibliothèque Python cleanlab propose des outils pour ce travail.
Installation

~~~shell
python -m pip install cleanlab
~~~

Exemple

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Limites
Les performances de détection dépendent de la qualité des probabilités et des hypothèses de la méthode.

Il faut donc :
- Vérifier les observations suspectes.
- Prioriser les cas les plus importants.
- Conserver un historique des corrections.
- Éviter de modifier automatiquement tous les labels signalés.
Principe fondamental : un modèle peut aider à détecter les erreurs d'annotation, mais il ne constitue pas une vérité terrain.

## Section D du cours — Data Quality Audits

## 10. Qu'est-ce qu'un audit de données ML ?

Un audit de données vise à identifier les problèmes susceptibles d'affecter l'apprentissage ou l'évaluation.

Il ne se limite pas aux valeurs manquantes.

Nous devons également examiner :
- Les types et formats.
- Les doublons.
- Les incohérences.
- Les distributions.
- Les déséquilibres.
- Les annotations.
- La représentativité.
- La fuite de données.
- Les différences entre train et test.
- Les sous-populations mal couvertes.

## 11. Les dimensions de qualité

Dimension	Question principale	Exemple
Completeness	Les informations nécessaires sont-elles présentes ?	Valeurs manquantes
Validity	Les valeurs respectent-elles les règles ?	Âge négatif
Consistency	Les informations sont-elles cohérentes ?	Labels contradictoires
Uniqueness	Existe-t-il des doublons ?	Même document plusieurs fois
Accuracy	Les valeurs sont-elles correctes ?	Mauvaise annotation
Representativeness	Les données couvrent-elles la population cible ?	Segment sous-représenté
Timeliness	Les données sont-elles suffisamment récentes ?	Ancien comportement utilisateur
Leakage	Certaines variables révèlent-elles la cible ?	Information postérieure à l'événement

Audit Python simple

~~~python
from ch06_cours12_lab import audit_dataset, label_quality, select_for_annotation, weak_majority, lf_promotion, lf_trusted_sender

def audit_dataset(df,target_col):
    if target_col not in df:raise ValueError('Colonne cible absente.')
    return {'n_rows':len(df),'n_columns':df.shape[1],'duplicates':int(df.duplicated().sum()),
            'missing_rate':(100*df.isna().mean()).round(2).to_dict(),
            'target_distribution':df[target_col].value_counts(normalize=True).to_dict()}
~~~

Ce rapport constitue une première étape.

Un audit complet doit intégrer des règles métier et des contrôles spécifiques au problème.


## 12. Détecter les doublons

Les doublons peuvent provoquer des problèmes importants.

Par exemple, si un même document apparaît dans le train et le test, les performances peuvent être artificiellement élevées.

Il existe plusieurs types de doublons :
- Doublons exacts.
- Doublons avec modifications mineures.
- Documents quasi identiques.
- Images très similaires.
- Textes paraphrasés.
Les doublons exacts sont faciles à détecter avec Pandas.

Les quasi-doublons peuvent nécessiter :
- Similarité cosinus.
- MinHash.
- Embeddings.
- Perceptual Hashing pour les images.
Exemple

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Il faut particulièrement vérifier les doublons entre les ensembles d'entraînement et de test.

## Section E du cours — Active Learning

## 13. Qu'est-ce que l'Active Learning ?

L'Active Learning consiste à sélectionner intelligemment les observations à annoter.

Au lieu d'annoter aléatoirement des milliers d'exemples, nous demandons au modèle d'identifier les observations potentiellement les plus utiles.
Exemple
Une entreprise dispose de 100 000 documents non annotés.

L'annotation manuelle coûte du temps et de l'argent.

Plutôt que d'annoter tous les documents, nous pouvons sélectionner les plus informatifs.

Le cycle devient :


## 14. Stratégies d'Active Learning


## 14.1 Uncertainty Sampling

Nous sélectionnons les observations pour lesquelles le modèle est le moins certain.

En classification binaire, les probabilités proches de 0,5 sont souvent les plus incertaines.
\[
\boxed{
U(x)=1-\max_k P(Y=k\mid x)
}
\]

Plus \(U(x)\) est élevé, plus l'incertitude est importante.
Exemple
Observation	P(classe 1)	Incertitude
A	0,99	Faible
B	0,51	Élevée
C	0,72	Moyenne
D	0,50	Maximale

Nous sélectionnerions prioritairement B et D.
Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## 15. Entropy Sampling

Une autre stratégie consiste à utiliser l'entropie des probabilités.
\[
\boxed{
H(Y\mid x)=
-\sum_{k=1}^{K}
p_k(x)\log p_k(x)
}
\]

L'entropie est élevée lorsque la distribution des probabilités est relativement uniforme.
Exemple
Pour trois classes :
\[
P_1=[0.98,0.01,0.01]
\]

\[
P_2=[0.34,0.33,0.33]
\]

La deuxième distribution possède une entropie plus élevée.

Le modèle est moins certain de sa prédiction.
Implémentation

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## 16. Margin Sampling

Le Margin Sampling utilise la différence entre les deux probabilités les plus élevées.
\[
\boxed{
Margin(x)=p_{(1)}(x)-p_{(2)}(x)
}
\]

où \(p_{(1)}\) et \(p_{(2)}\) sont les deux probabilités les plus élevées.

Une faible marge signifie que le modèle hésite entre deux classes.
Exemple
\[
P_A=[0.90,0.08,0.02]
\]

Marge :
\[
0.90-0.08=0.82
\]

\[
P_B=[0.48,0.47,0.05]
\]

Marge :
\[
0.48-0.47=0.01
\]

L'observation B est plus incertaine.


## 17. Limites de l'Uncertainty Sampling

L'incertitude seule ne suffit pas toujours.

Le modèle peut être incertain sur :
- Des observations aberrantes.
- Des données de mauvaise qualité.
- Des exemples impossibles à annoter.
- Des observations très similaires entre elles.
Si nous sélectionnons uniquement les exemples les plus incertains, nous risquons de constituer un lot peu diversifié.

Des stratégies plus avancées combinent :
\[
\boxed{
Score(x)=
\alpha\cdot Uncertainty(x)
+
\beta\cdot Diversity(x)
}
\]

avec des scores normalisés.

L'objectif est de sélectionner des observations à la fois informatives et représentatives.

## Section F du cours — Weak Supervision

## 18. Qu'est-ce que le Weak Supervision ?

Le Weak Supervision consiste à créer des labels approximatifs à partir de règles, de sources externes ou de modèles existants.

L'objectif est de réduire la dépendance à une annotation manuelle exhaustive.
Exemple
Nous souhaitons classifier des emails en deux catégories :
- Spam.
- Non-spam.
Nous pouvons créer des règles :
- Si le texte contient certains termes promotionnels, proposer le label spam.
- Si l'expéditeur appartient à une liste de contacts connus, proposer non-spam.
- Si un modèle préentraîné détecte une forte probabilité de spam, proposer spam.
Ces règles sont appelées Labeling Functions.


## 19. Labeling Functions

Une fonction d'annotation retourne généralement :
- Un label.
- Ou une abstention.
Exemple Python

~~~python
from ch06_cours12_lab import audit_dataset, label_quality, select_for_annotation, weak_majority, lf_promotion, lf_trusted_sender

def lf_promotion(text):
    return 1 if any(term in text.lower() for term in ['free prize','limited offer','win money']) else -1

def lf_trusted_sender(sender):
    return 0 if sender.split('@')[-1].lower() in ['company.com','university.edu'] else -1
~~~

Ces règles ne sont pas parfaites.

Elles peuvent :
- Se tromper.
- Se contredire.
- Ne couvrir qu'une partie du dataset.
C'est pourquoi nous devons les évaluer.


## 20. Combiner plusieurs sources faibles

Une approche simple consiste à utiliser un vote majoritaire.

Mais elle présente des limites.

Certaines règles sont plus fiables que d'autres.

De plus, plusieurs règles peuvent être fortement corrélées.

Des approches probabilistes, comme celles popularisées par Snorkel, cherchent à combiner les sources d'annotation en tenant compte de leur comportement.
Exemple conceptuel
Document	Règle 1	Règle 2	Règle 3	Label final
A	Spam	Spam	Abstention	Spam
B	Spam	Non-spam	Abstention	Incertain
C	Abstention	Non-spam	Non-spam	Non-spam

La qualité du label final dépend de la fiabilité des règles et de leur dépendance.

Important : le Weak Supervision ne remplace pas complètement les annotations humaines. Un ensemble de référence correctement annoté reste essentiel pour mesurer la qualité des labels générés.

## Section G du cours — Data Augmentation et données synthétiques

## 21. Pourquoi augmenter les données ?

Dans certains problèmes, les données disponibles sont insuffisantes.

Nous pouvons alors générer de nouvelles observations à partir des données existantes.

C'est le principe de la Data Augmentation.
Exemples
Images :
- Rotation.
- Recadrage.
- Variation de luminosité.
- Bruit léger.
Textes :
- Reformulation.
- Substitution contrôlée de mots.
- Traduction aller-retour, avec vérification du sens.
Données tabulaires :
- Génération synthétique sous contraintes.
- Méthodes de suréchantillonnage adaptées.
- Simulations métier.
Attention
Une augmentation doit préserver la validité du label.

Une transformation qui modifie la classe réelle peut dégrader le modèle.

Par exemple, modifier une valeur financière déterminante peut transformer une observation saine en observation à risque.


## 22. Données synthétiques

Les données synthétiques sont générées artificiellement afin de reproduire certaines propriétés des données réelles.

Elles peuvent servir à :
- Tester des pipelines.
- Simuler des cas rares.
- Compléter certains scénarios.
- Développer des prototypes.
- Étudier la robustesse.
Cependant, elles présentent plusieurs risques :
- Distribution différente des données réelles.
- Relations artificielles.
- Amplification de biais.
- Fuite possible d'informations privées si le générateur mémorise des exemples.
- Évaluation trompeuse si le test contient des données générées trop proches du train.
La qualité des données synthétiques doit être vérifiée à la fois statistiquement et métier.

## Section H du cours — Mini-projet complet

## 23. Améliorer un classifieur en corrigeant les labels

Nous allons construire un projet Data-Centric avec Scikit-learn et Cleanlab.
Objectif
Comparer deux stratégies :
- Modèle entraîné sur des labels bruités.
- Modèle entraîné après correction d'une partie des labels.
Nous utiliserons un dataset synthétique afin de connaître les vrais labels et de mesurer les effets des corrections.

## Étape 1 — Générer les données


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le test reste propre et indépendant.

## Étape 2 — Injecter du bruit dans le train


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Nous avons introduit 15 % d'erreurs d'annotation dans le train.

## Étape 3 — Entraîner un premier modèle


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 4 — Produire des probabilités OOF


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 5 — Identifier les labels suspects


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 6 — Simuler une vérification humaine

Dans cette simulation, nous connaissons les vrais labels.

Nous pouvons donc reproduire l'effet d'une vérification humaine parfaite sur les observations suspectes.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Dans un vrai projet, cette étape nécessiterait une réannotation humaine ou une autre source de vérité fiable.

## Étape 7 — Réentraîner le modèle


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Interprétation
Si le Macro-F1 augmente, cela suggère que les corrections ont amélioré la généralisation dans cette expérience.

Cependant, ce résultat n'est pas garanti.

La détection des labels suspects peut aussi sélectionner des exemples difficiles mais correctement annotés.

C'est pourquoi la vérification et l'évaluation restent indispensables.
Amélioration méthodologique
Pour comparer rigoureusement plusieurs stratégies de correction, nous devrions utiliser :
- Une validation indépendante pour choisir la stratégie.
- Un test final réservé à l'évaluation.
- Plusieurs seeds.
- Des intervalles de confiance.
- Une mesure du coût d'annotation.
- Une comparaison avec une sélection aléatoire d'exemples à corriger.
Le but n'est pas seulement d'améliorer le score, mais d'obtenir un gain fiable pour un coût raisonnable.

## Section I du cours — Mesurer le retour sur investissement

## 24. Performance vs coût d'annotation

Imaginons deux stratégies :
Stratégie	Documents annotés	Macro-F1
Annotation aléatoire	1 000	0,82
Active Learning	500	0,84
Active Learning + audit	650	0,87

Ces résultats sont fictifs.

L'approche Data-Centric pourrait permettre d'obtenir une meilleure performance avec moins d'annotations.

Mais nous devons également considérer :
- Le temps d'annotation.
- Le coût des experts.
- Le temps de calcul.
- Le coût des audits.
- La qualité finale des labels.
Learning Curves
Une courbe d'apprentissage peut représenter la performance en fonction du nombre d'observations annotées.

Annotation aléatoireActive Learning

Données fictives illustrant deux stratégies d'annotation.

L'Active Learning peut être plus efficace dans certains contextes, mais ce n'est pas une garantie universelle.

La performance dépend de la stratégie de sélection, du modèle, du dataset et de la qualité des annotations.


## 25. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Faire confiance à tous les labels	Apprentissage biaisé	Auditer les annotations
Corriger automatiquement les désaccords	Introduction de nouvelles erreurs	Vérification humaine
Détecter les erreurs avec des prédictions sur train	Résultats biaisés	Utiliser des prédictions OOF
Modifier le test pour améliorer le score	Évaluation non fiable	Conserver un test indépendant
Sélectionner uniquement les exemples incertains	Faible diversité	Combiner incertitude et diversité
Ignorer le coût d'annotation	Stratégie inefficace	Mesurer le ROI
Utiliser des règles faibles sans validation	Labels de mauvaise qualité	Évaluer les Labeling Functions
Ajouter des données synthétiques sans contrôle	Distribution artificielle	Vérifier validité et réalisme
Ignorer la provenance des données	Faible traçabilité	Versionner les datasets
Nettoyer sans mesurer l'effet	Effort non quantifié	Comparer avant/après

[[quiz]]


## 26. Synthèse du cours

Les concepts essentiels à retenir :
- Data-Centric ML : amélioration systématique des données.
- Label Noise : erreurs et incohérences d'annotation.
- Out-of-Fold Predictions : prédictions indépendantes des observations évaluées.
- Confident Learning : identification de labels potentiellement incorrects.
- Data Quality Audit : contrôle de la fiabilité, cohérence et représentativité des données.
- Active Learning : sélection intelligente des exemples à annoter.
- Uncertainty Sampling : sélection selon l'incertitude.
- Entropy Sampling : sélection selon l'entropie des probabilités.
- Margin Sampling : sélection selon l'écart entre les deux meilleures classes.
- Weak Supervision : production de labels approximatifs à partir de règles ou de sources faibles.
- Data Augmentation : enrichissement des données par transformations valides.
- Data Versioning : traçabilité des différentes versions du dataset.

## 27. Fin du module ML spécialisé

## Section B du cours — Machine Learning spécialisé terminée

Tu as désormais parcouru les cinq cours de cette partie : Stacking et Blending, Cost-Sensitive Learning, Survival Analysis, Recommender Systems et Data-Centric ML.

Ces méthodes complètent les bases du Machine Learning supervisé, non supervisé, de l'évaluation et des séries temporelles.

Prochaine étape — Deep Learning
Le prochain grand module de LearnLoop sera consacré au Deep Learning, en commençant par les réseaux de neurones artificiels, la rétropropagation et l'optimisation.
Deep Learning — Cours 01 : Introduction aux réseaux de neurones

Perceptron, neurones, couches, fonctions d'activation et premier réseau avec PyTorch.



## Notes sur l’implémentation et le protocole

Les probabilités de qualité de labels sont OOF et concernent seulement le train. Le budget de revue et les politiques sont fixés avant le test. La correction utilise un oracle connu uniquement dans cette simulation et est comparée à une revue aléatoire de même budget. En pratique, une revue humaine reste nécessaire. Une expérience unique ne garantit ni un gain ni un retour sur investissement.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np
import pandas as pd

def audit_dataset(df,target_col):
    if target_col not in df:raise ValueError('Colonne cible absente.')
    return {'n_rows':len(df),'n_columns':df.shape[1],'duplicates':int(df.duplicated().sum()),
            'missing_rate':(100*df.isna().mean()).round(2).to_dict(),
            'target_distribution':df[target_col].value_counts(normalize=True).to_dict()}

def label_quality(labels,oof_probabilities):
    labels=np.asarray(labels);p=np.asarray(oof_probabilities,dtype=float)
    if labels.ndim!=1 or p.ndim!=2 or len(p)!=len(labels) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=p.shape[1]) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Labels entiers et probabilités OOF normalisées alignées requis.')
    return p[np.arange(len(labels)),labels]

def select_for_annotation(probabilities,budget=20,strategy='entropy'):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or not len(p) or p.shape[1]<2 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1) or not isinstance(budget,int) or budget<0:
        raise ValueError('Probabilités multiclasses normalisées et budget entier positif ou nul requis.')
    if strategy=='entropy':
        terms=np.zeros_like(p);np.log(p,out=terms,where=p>0);score=-(p*terms).sum(axis=1)
    elif strategy=='uncertainty':score=1-p.max(axis=1)
    elif strategy=='margin':
        ordered=np.sort(p,axis=1);score=-(ordered[:,-1]-ordered[:,-2])
    else:raise ValueError('Stratégie inconnue.')
    return np.argsort(-score,kind='stable')[:budget]

def weak_majority(votes):
    votes=np.asarray(votes)
    if votes.ndim!=2 or not np.isin(votes,[-1,0,1]).all():raise ValueError('Votes -1 (abstention), 0 ou 1 requis.')
    positive=(votes==1).sum(axis=1);negative=(votes==0).sum(axis=1)
    return np.where(positive>negative,1,np.where(negative>positive,0,-1))

def lf_promotion(text):
    return 1 if any(term in text.lower() for term in ['free prize','limited offer','win money']) else -1

def lf_trusted_sender(sender):
    return 0 if sender.split('@')[-1].lower() in ['company.com','university.edu'] else -1

def run(plot=False):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from cleanlab.filter import find_label_issues
    X,y=make_classification(n_samples=2400,n_features=20,n_informative=12,n_redundant=4,n_classes=3,class_sep=1.2,random_state=42)
    train,test=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42)
    # Le test propre reste réservé. La politique de revue et son budget sont fixés à l'avance.
    truth=y[train].copy();noisy=truth.copy();rng=np.random.default_rng(42)
    corrupted=rng.choice(len(noisy),int(.15*len(noisy)),replace=False)
    noisy[corrupted]=(noisy[corrupted]+rng.integers(1,3,len(corrupted)))%3
    def estimator():return RandomForestClassifier(n_estimators=70,max_depth=10,random_state=42,n_jobs=1)
    cv=StratifiedKFold(3,shuffle=True,random_state=42)
    oof=cross_val_predict(estimator(),X[train],noisy,cv=cv,method='predict_proba',n_jobs=1)
    mask=find_label_issues(labels=noisy,pred_probs=oof)
    budget=100;quality=label_quality(noisy,oof)
    suspects=np.flatnonzero(mask);selected=suspects[np.argsort(quality[suspects],kind='stable')[:budget]]
    random_review=rng.choice(len(noisy),len(selected),replace=False)
    corrected=noisy.copy();corrected[selected]=truth[selected]
    random_corrected=noisy.copy();random_corrected[random_review]=truth[random_review]
    # Oracle simulé seulement pour les labels du train ; jamais pour une correction automatique réelle.
    results={}
    for name,labels in [('Bruit initial',noisy),('Revue OOF',corrected),('Revue aléatoire même budget',random_corrected)]:
        model=estimator().fit(X[train],labels);results[name]=f1_score(y[test],model.predict(X[test]),average='macro')
    print('Audit final, politiques fixées sans consulter le test :',results)
    print('Labels revus :',len(selected),'erreurs confirmées :',int(np.sum(noisy[selected]!=truth[selected])))
    print('Active Learning, entropie sur pool fictif :',select_for_annotation(oof[:20],5))
    print('Weak supervision :',weak_majority([[1,1,-1],[1,0,-1],[-1,-1,-1],[0,-1,0]]))
    frame=pd.DataFrame(X[train,:3],columns=['x1','x2','x3']);frame['label']=noisy
    print('Audit :',audit_dataset(frame,'label'))
    print('Une simulation unique ne garantit pas un gain ; répéter avec plusieurs seeds et validation indépendante pour choisir une politique.')
    return {'test_macro_F1':results,'reviewed':len(selected)}

if __name__=='__main__':run()
~~~

## Challenge Python

Compléter audit_dataset, select_for_annotation et weak_majority. Classer les probabilités par entropie, incertitude ou marge avec un budget fixé. Ignorer les abstentions et conserver l’abstention en cas d’égalité. Ne pas corriger automatiquement les labels suspects.

[[assessment]]

[[interview]]
