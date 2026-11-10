"""Behavioral checks, separate from the learner's code. Never compare code strings."""
results = []

def check(name, action):
    try:
        action()
        results.append({"name": name, "passed": True})
    except BaseException as error:
        results.append({"name": name, "passed": False,
                        "detail": type(error).__name__ + ": " + str(error)})

if exercise == "ch02_cours01":
    import numpy as np
    import pandas as pd
    def report():
        sample = pd.DataFrame({
            "age": [20., -1., 40., np.nan],
            "income": [100., np.nan, 300., np.nan],
            "debt_ratio": [.2, 1.8, .5, np.nan],
            "region": ["Tunis", "Sfax", "Tunis", None],
            "default": [0., 1., 0., 0.],
        })
        return namespace["data_quality_report"](sample)
    def numeric():
        r = report()
        for name, expected in {"mean": 200., "median": 200.,
                               "std": np.std([100., 300.], ddof=1),
                               "min": 100., "max": 300.}.items():
            assert name in r.columns, "Colonne manquante : " + name
            assert np.isclose(r.loc["income", name], expected), "Valeur incorrecte : income / " + name
        assert pd.isna(r.loc["region", "mean"]), "Une colonne texte n'a pas de moyenne numérique."
    def missing():
        r = report()
        assert r.loc["income", "missing_count"] == 2, "Deux revenus sont absents."
        assert np.isclose(r.loc["income", "missing_rate"], 50), "Le taux attendu est 50 %."
        assert r["missing_rate"].is_monotonic_decreasing, "Trie les taux par ordre décroissant."
    def immutable():
        sample = pd.DataFrame({"income": [1., np.nan, 3.]})
        before = sample.copy(deep=True)
        namespace["data_quality_report"](sample)
        pd.testing.assert_frame_equal(sample, before)
    check("Statistiques numériques, écart-type et colonnes texte", numeric)
    check("Valeurs absentes et tri décroissant", missing)
    # The exercise's optional business rules can require all five columns.
    def untouched():
        import copy
        sample = pd.DataFrame({"age": [20., 30.], "income": [100., np.nan],
            "debt_ratio": [.2, .3], "region": ["Tunis", "Sfax"], "default": [0, 1]})
        before = sample.copy(deep=True)
        namespace["data_quality_report"](sample)
        pd.testing.assert_frame_equal(sample, before)
    check("Le tableau d'origine reste intact", untouched)

if exercise == "ch03_cours01":
    import numpy as np
    def coefficients():
        for x, y, expected in [
            ([0, 1, 2], [3, 5, 7], [3, 2]),
            ([-2, 0, 2, 4], [5, 1, -3, -7], [1, -2]),
            ([1, 2, 3], [8, 8, 8], [8, 0]),
        ]:
            actual = namespace["fit_ols"](np.array(x, dtype=float), np.array(y, dtype=float))
            assert np.allclose(actual, expected), "Retourne (intercept, pente), valeurs attendues : " + str(expected)
    def rejects(x, y):
        try:
            namespace["fit_ols"](np.array(x, dtype=float), np.array(y, dtype=float))
        except ValueError:
            return
        raise AssertionError("Une ValueError explicite est attendue pour ces données.")
    check("Coefficients OLS sur trois jeux de données", coefficients)
    check("Refuser une variable x constante", lambda: rejects([1, 1, 1], [2, 3, 4]))
    check("Refuser des dimensions incompatibles", lambda: rejects([1, 2], [1, 2, 3]))
    check("Refuser les valeurs non finies", lambda: rejects([1, float("nan")], [2, 3]))

if exercise == "ch05_cours03_browser":
    import numpy as np
    check("Reconstruction additive : 50 % + 28 points = 78 %", lambda: np.testing.assert_allclose(namespace["reconstruct"](.5, [.25, .18, -.1, -.05]), .78))
    check("Ne pas tronquer artificiellement une somme additive", lambda: np.testing.assert_allclose(namespace["reconstruct"](2., [.5, -.1]), 2.4))
    check("Importance globale : moyenne des valeurs absolues par variable", lambda: np.testing.assert_allclose(namespace["global_importance"]([[.2, -.1], [-.4, .3]]), [.3, .2]))

def rejects_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Une ValueError explicite est attendue.")

if exercise == "ch05_cours04":
    import numpy as np
    f = namespace.get("perturb")
    sample = np.arange(24., dtype=float).reshape(8, 3)
    scale = np.array([1., 2., 3.])
    def unchanged():
        before = sample.copy()
        actual = f(sample, scale, seed=42)
        np.testing.assert_array_equal(actual, sample)
        np.testing.assert_array_equal(sample, before)
        assert actual is not sample, "Retourne une copie indépendante."
    def shifted():
        actual = f(sample, scale, shift=2.)
        expected = sample.copy()
        expected[:, 0] += 2*scale[0]
        np.testing.assert_allclose(actual, expected)
    def noise():
        first = f(sample, scale, level=.2, seed=7)
        second = f(sample, scale, level=.2, seed=7)
        np.testing.assert_allclose(first, second)
        assert not np.array_equal(first, sample), "Un niveau positif doit perturber les données."
        assert not np.array_equal(first, f(sample, scale, level=.2, seed=8)), "La graine doit influencer le bruit."
    def missing():
        assert np.isnan(f(sample, scale, missing_rate=1.)).all(), "Un taux de 100 % masque toutes les cellules."
    check("Niveau zéro : copie intacte sans modifier X", unchanged)
    check("Décalage de la première variable avec échelle train", shifted)
    check("Bruit reproductible et sensible à la graine", noise)
    check("Simulation des cellules manquantes", missing)
    check("Refuser un taux manquant invalide", lambda: rejects_value_error(lambda: f(sample, scale, missing_rate=1.2)))
    check("Refuser un bruit négatif", lambda: rejects_value_error(lambda: f(sample, scale, level=-.1)))

if exercise == "ch05_cours05":
    import numpy as np
    f = namespace.get("conformal_quantile")
    g = namespace.get("interval_metrics")
    scores = [5, 7, 8, 10, 11, 12, 14, 15, 18, 20]
    check("Rang fini : n=10 et alpha=0,1 donnent q=20", lambda: np.testing.assert_allclose(f(scores, .1), 20.))
    def infinite():
        assert np.isposinf(f(scores, .05)), "k=11 > n=10 impose +inf ; ne plafonne pas à max(scores)."
    check("Petit échantillon : borne infinie à 95 %", infinite)
    check("CQR : conserver les scores négatifs", lambda: np.testing.assert_allclose(f([-3, -2, -1, 1], .5), -1.))
    check("Refuser un jeu de scores vide", lambda: rejects_value_error(lambda: f([], .1)))
    check("Refuser alpha hors de ]0,1[", lambda: rejects_value_error(lambda: f(scores, 0)))
    def intervals():
        r = g([0, 1, 2], [0, 0, 3], [0, 2, 4])
        np.testing.assert_allclose(r["coverage"], 2/3)
        np.testing.assert_allclose(r["mean_width"], 1.)
    check("Couverture avec bornes incluses et largeur moyenne", intervals)

if exercise == "ch05_cours06":
    import numpy as np
    f = namespace.get("group_metrics")
    def rates():
        y = [1, 1, 0, 0, 1, 1, 0, 0]
        pred = [1, 0, 1, 0, 1, 1, 0, 0]
        r = f(y, pred, ["A"]*4+["B"]*4)
        for key in ["TP", "FP", "FN", "TN"]:
            assert r.loc["A", key] == 1, "Vérifie le décompte " + key
        for key in ["TPR", "FPR", "FNR", "precision", "selection_rate", "prevalence", "accuracy"]:
            np.testing.assert_allclose(r.loc["A", key], .5)
        assert r.loc["B", "TPR"] == 1 and r.loc["B", "FPR"] == 0
        assert r.loc["A", "n"] == 4
    def undefined():
        r = f([0, 0, 1, 1], [0, 0, 0, 0], ["A", "A", "B", "B"])
        assert np.isnan(r.loc["A", "TPR"]), "Sans vrais positifs possibles, TPR est non mesurable."
        assert np.isnan(r.loc["B", "FPR"]), "Sans négatifs réels, FPR est non mesurable."
        assert np.isnan(r.loc["A", "precision"]), "Sans prédiction positive, la précision est non mesurable."
    check("Effectifs et métriques des deux groupes", rates)
    check("Dénominateurs nuls : NaN, sans inventer zéro", undefined)
    check("Refuser des vecteurs désalignés", lambda: rejects_value_error(lambda: f([0, 1], [0], ["A", "B"])))
    check("Refuser des labels non binaires", lambda: rejects_value_error(lambda: f([0, 2], [0, 1], ["A", "B"])))

if exercise == "ch05_cours07":
    import numpy as np
    f = namespace.get("individual_log_loss")
    check("Perte individuelle binaire correcte", lambda: np.testing.assert_allclose(f([1, 0], [.8, .2]), [-np.log(.8)]*2))
    def endpoints():
        original = np.array([0., 1.])
        result = f([1, 0], original)
        assert np.isfinite(result).all(), "Évite log(0) avec un clipping numérique."
        assert (np.asarray(result) > 30).all(), "Une prédiction certaine et fausse doit coûter cher."
        np.testing.assert_array_equal(original, [0., 1.])
    check("Probabilités 0/1 : sortie finie et entrées intactes", endpoints)
    check("Refuser les probabilités hors domaine", lambda: rejects_value_error(lambda: f([1], [1.2])))
    check("Refuser des vecteurs désalignés", lambda: rejects_value_error(lambda: f([1, 0], [.2])))

if exercise == "ch05_cours08":
    import numpy as np
    f, g = namespace.get("calculate_psi"), namespace.get("monitoring_status")
    reference = np.linspace(0, 10, 100)
    check("PSI nul pour deux distributions identiques", lambda: np.testing.assert_allclose(f(reference, reference), 0., atol=1e-12))
    def outside():
        value = f(reference, reference+100)
        assert np.isfinite(value) and value > 0, "Les valeurs hors bornes de référence doivent être incluses."
    check("Bornes infinies : conserver les valeurs extrêmes", outside)
    check("Refuser une référence constante", lambda: rejects_value_error(lambda: f([1, 1], [2, 3])))
    check("Refuser un échantillon fini vide", lambda: rejects_value_error(lambda: f([np.nan], [1, 2])))
    def alerts():
        assert g(.10, .02, .05) == "GREEN", "Les bornes vertes sont incluses."
        assert g(.25, .05, .10) == "AMBER", "Les bornes orange sont incluses."
        assert g(.26, 0., 0.) == "RED"
        assert g(0., .06, 0.) == "RED"
        assert g(0., 0., .11) == "RED"
    check("Agrégation des alertes et limites des seuils", alerts)

if exercise == "ch05_cours09":
    import numpy as np
    f, g = namespace.get("compare_auc_bootstrap"), namespace.get("holm_adjust")
    y = np.tile([0, 1], 10)
    a = np.where(y == 1, .8, .2)
    def identical():
        r = f(y, a, a, n_bootstrap=40, seed=42)
        np.testing.assert_allclose([r["difference"], r["lower"], r["upper"]], 0.)
        np.testing.assert_allclose(r["samples"], 0.)
        assert 1 < r["n_valid"] <= 40
    def contrast():
        r = f(y, a, 1-a, n_bootstrap=40, seed=7)
        np.testing.assert_allclose([r["difference"], r["lower"], r["upper"]], 1.)
        repeated = f(y, a, 1-a, n_bootstrap=40, seed=7)
        np.testing.assert_array_equal(r["samples"], repeated["samples"])
    check("Bootstrap apparié : mêmes scores, différence toujours nulle", identical)
    check("Contraste parfait et reproductibilité", contrast)
    check("Refuser une cible mono-classe", lambda: rejects_value_error(lambda: f([0, 0], [.1, .2], [.2, .3], n_bootstrap=10)))
    check("Holm : valeurs attendues et ordre initial", lambda: np.testing.assert_allclose(g([.18, .001, .048, .012, .032]), [.18, .005, .096, .048, .096]))
    check("Holm : p-values invalides", lambda: rejects_value_error(lambda: g([.1, 1.2])))

if exercise == "ch05_cours10":
    import numpy as np
    f = namespace.get("evaluate_model")
    def green():
        r = f({"auc": .8, "brier": .05}, .10)
        assert r["status"] == "GREEN"
        assert r["requires_human_review"] is True
    def missing():
        for auc in [None, np.nan, np.inf, -.1, 1.1]:
            r = f({"auc": auc, "brier": .04}, .05)
            assert r["status"] == "RED", "Une métrique absente ou invalide doit bloquer."
            assert r["findings"], "La cause doit apparaître dans les findings."
    def amber_red():
        assert f({"auc": .79, "brier": .05}, .10)["status"] == "AMBER"
        assert f({"auc": .9, "brier": .09}, .10)["status"] == "RED"
        assert f({"auc": .9, "brier": .04}, .26)["status"] == "RED"
    def critical():
        r = f({"auc": .9, "brier": .03}, .01, critical_findings=["Split temporel invalide"])
        assert r["status"] == "RED" and any("Split temporel" in s for s in r["findings"])
    def configured():
        r = f({"auc": .79, "brier": .06}, .17,
            rules={"auc": (.75, .65), "brier": (.07, .10), "psi": (.20, .30)})
        assert r["status"] == "GREEN", "Les règles fournies doivent remplacer les seuils par défaut."
    check("Bornes vertes et revue humaine explicite", green)
    check("Métriques absentes, NaN, infinies ou hors domaine", missing)
    check("Le niveau le plus sévère l'emporte", amber_red)
    check("Un contrôle critique bloque et laisse un finding", critical)
    check("Configuration explicite des seuils", configured)
