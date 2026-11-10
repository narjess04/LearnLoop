"""Projet complet local : XGBoost + SHAP + LIME.
Dépendances : numpy pandas scikit-learn matplotlib shap lime xgboost.
L'atelier navigateur propose séparément un exercice NumPy sur l'additivité.
"""
# TODO : séparer train/test, entraîner XGBoost et évaluer AUC/F1 au seuil fixé.
# TODO : TreeExplainer avec background train et model_output="probability".
# TODO : vérifier base_values + sum(values) == predict_proba pour la classe 1.
# TODO : produire beeswarm et waterfall pour trois clients.
# TODO : expliquer les mêmes clients avec LIME, labels=(1,), trois graines.
# TODO : comparer fidélité locale, stabilité, population de référence et limites.
def run():
    raise NotImplementedError("Complète le mini-projet puis compare au corrigé.")

if __name__ == "__main__":
    run()
