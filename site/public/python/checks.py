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


if exercise == "ch06_cours01":
    import numpy as np
    def differences():
        fn = namespace['difference']
        for values, lag, expected in [([100,110,125,130],1,[10,15,5]), ([2,8,3,12,7],2,[1,4,4]), ([4,4,4],1,[0,0])]:
            original = list(values)
            np.testing.assert_allclose(fn(values, lag), expected)
            assert values == original, 'Ne pas modifier les entrées.'
    check('Différences simples, retard 2 et entrée intacte', differences)

if exercise == "ch06_cours02":
    import numpy as np
    def baselines():
        history = np.array([2.,7.,4.,10.,6.,15.])
        before = history.copy()
        pred = namespace['baseline_forecast'](history, 7, period=3, window=2)
        np.testing.assert_allclose(pred['Naive'], [15]*7)
        np.testing.assert_allclose(pred['Seasonal Naive'], [10,6,15,10,6,15,10])
        np.testing.assert_allclose(pred['Moving Average'], [10.5]*7)
        np.testing.assert_allclose(pred['Drift'], 15+np.arange(1,8)*13/5)
        np.testing.assert_array_equal(history, before)
    def metrics():
        fn = namespace['forecasting_metrics']
        result = fn([100,120,140],[110,115,130],[10,20,30,40],1)
        assert np.isclose(result['MAE'],25/3)
        assert np.isclose(result['RMSE'],np.sqrt(75))
        assert np.isclose(result['MASE'],5/6)
        assert np.isclose(result['sMAPE'],100*np.mean([20/210,10/235,20/270]))
        seasonal = fn([0,2],[0,0],[1,10,3,14],2)
        assert np.isclose(seasonal['MASE'],1/3)
        assert np.isclose(seasonal['sMAPE'],100)
        assert np.isnan(fn([1],[1],[4,4,4],1)['MASE'])
    check('Quatre baselines multi-horizon sans modification du passé', baselines)
    check('MAE, RMSE, MASE sur train, sMAPE et cas nuls', metrics)


if exercise == "ch06_cours03":
    import numpy as np
    def smoothing():
        fn=namespace['ses_forecast']
        assert np.isclose(fn([10,20,0],.5),7.5)
        assert fn([2,9,4],0)==2 and fn([2,9,4],1)==4
    def selection():
        rows=[{'MAE':.1,'converged':False},{'MAE':np.nan},{'MAE':9.,'id':'first'},{'MAE':2.,'id':'best'}]
        assert namespace['select_best'](rows)['id']=='best'
        assert rows[2]['id']=='first'
        assert namespace['residual_model_df']((2,1,3),(1,1,2,12))==8
    check('Lissage exponentiel et valeurs limites de alpha',smoothing)
    check('Choix sur MAE finie, convergence et paramètres saisonniers',selection)

if exercise == "ch06_cours04":
    import numpy as np
    import pandas as pd
    def frame():
        return pd.DataFrame({'sales':np.arange(60.)},index=pd.date_range('2025-01-01',periods=60))
    def causal_features():
        df=frame();before=df.copy(deep=True)
        result=namespace['create_features_for_future'](df)
        date=df.index[35]
        assert result.loc[date,'lag_7']==28
        assert np.isclose(result.loc[date,'rolling_mean_7'],31)
        assert np.isclose(result.loc[date,'rolling_std_7'],np.std(np.arange(28.,35.),ddof=1))
        changed=df.copy();changed.loc[date:,'sales']+=10000
        alt=namespace['create_features_for_future'](changed)
        features=['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_std_7']
        pd.testing.assert_frame_equal(result.loc[:date,features],alt.loc[:date,features])
        pd.testing.assert_frame_equal(df,before)
        future=pd.concat([df,pd.DataFrame({'sales':[np.nan]},index=[df.index[-1]+pd.Timedelta(days=1)])])
        assert namespace['create_features_for_future'](future).index[-1]==future.index[-1]
    def recursive():
        class Model:
            def predict(self,X):return X['lag_1'].to_numpy()+1
        history=frame().iloc[:40];before=history.copy(deep=True)
        result=namespace['recursive_forecast'](Model(),history,5)
        np.testing.assert_allclose(result,[40,41,42,43,44])
        assert result.index[0]==pd.Timestamp('2025-02-10')
        pd.testing.assert_frame_equal(history,before)
    check('Features causales, rolling décalé, ligne future conservée',causal_features)
    check('Récursion avec prédictions réinjectées et historique intact',recursive)

if exercise == "ch06_cours05":
    import numpy as np
    import pandas as pd
    def croston():
        fn=namespace['croston_forecast']
        assert fn([0,0,0])==0
        assert np.isclose(fn([0,0,4,0,8,0],.5),2.4)
        assert np.isclose(fn([0,0,4,0,8,0,0,0],.5),2.4)
        assert namespace['tsb_forecast']([0,0,4,0,8]+[0]*10)<namespace['tsb_forecast']([0,0,4,0,8])
    def hierarchy():
        S=np.array([[1.,1.],[1.,0.],[0.,1.]])
        y=namespace['mint_reconcile']([1000,600,450],S,np.diag([100.,64.,36.]))
        np.testing.assert_allclose(y,[1025,584,441])
        assert np.isclose(y[0],y[1]+y[2])
        consistent=np.array([900.,500.,400.])
        np.testing.assert_allclose(namespace['mint_reconcile'](consistent,S,np.eye(3)),consistent)
    def panels():
        dates=pd.date_range('2025-01-01',periods=40)
        df=pd.DataFrame({'series_id':['A']*40+['B']*40,'date':list(dates)*2,'sales':list(np.arange(40.)+100)+list(np.arange(40.)+1000)})
        data=namespace['create_panel_features'](df.sample(frac=1,random_state=42))
        row=data[(data['series_id']=='B')&(data['date']==dates[28])].iloc[0]
        assert row['lag_1']==1027 and row['lag_28']==1000
        assert np.isclose(row['rolling_mean_7'],1024)
    check('Croston, périodes sans demande et décroissance TSB',croston)
    check('Réconciliation pondérée et invariance des prévisions cohérentes',hierarchy)
    check('Features par série sans contamination entre produits',panels)


if exercise == "ch06_cours06":
    import numpy as np
    def quantiles():
        assert namespace['pinball_loss']([120],[100],.9)==18
        assert np.isclose(namespace['pinball_loss']([80],[100],.9),2)
        assert namespace['conformal_quantile'](np.arange(1,11),.1)==10
        assert np.isinf(namespace['conformal_quantile']([1,2,3],.1))
        assert namespace['conformal_quantile']([-3,-2,-1],.5)==-2
    def crps():
        fn=namespace['crps_ensemble'];samples=np.array([1.,1.,4.,9.]);y=3.
        expected=np.abs(samples-y).mean()-.5*np.abs(samples[:,None]-samples[None,:]).mean()
        assert np.isclose(fn(y,samples),expected)
        assert fn(5,[2])==3
        assert np.isclose(fn(y+20,samples+20),expected)
    check('Pinball asymétrique et rang conforme fini-échantillon',quantiles)
    check('CRPS, membres répétés et invariance par translation',crps)

if exercise == "ch06_cours07":
    import numpy as np
    import pandas as pd
    def rolling():
        values=pd.Series([1.,3.,2.,5.,100.,8.]);before=values.copy()
        z=namespace['rolling_zscore'](values,3)
        assert np.isclose(z.iloc[3],3.)
        changed=values.copy();changed.iloc[4:]+=1000
        pd.testing.assert_series_equal(z.iloc[:4],namespace['rolling_zscore'](changed,3).iloc[:4])
        pd.testing.assert_series_equal(values,before)
    def cusum():
        assert namespace['cusum_detector']([0,0,3,3,3,3],0,1,3)==[3,5]
        assert namespace['cusum_detector']([0,0,-3,-3],0,1,3)==[3]
    def drift():
        fn=namespace['population_stability_index'];reference=np.arange(100.)
        assert abs(fn(reference,reference))<1e-10
        value=fn(reference,np.arange(100.)+10000)
        assert np.isfinite(value) and value>1
    check('Rolling Z-score sans utilisation du futur',rolling)
    check('CUSUM bilatéral et réinitialisation après alerte',cusum)
    check('PSI, référence identique et valeurs nouvelles extrêmes',drift)

if exercise == "ch06_cours08":
    import numpy as np
    def voting():
        np.testing.assert_allclose(namespace['soft_vote']([[.8,.6,.7]],[1,2,3]),[4.1/6])
        np.testing.assert_allclose(namespace['soft_vote']([[.1,.9],[.8,.2]]),[.5,.5])
    def oof():
        from sklearn.base import BaseEstimator
        class Memorizer(BaseEstimator):
            def fit(self,X,y):self.seen=set(np.asarray(X)[:,0]);self.classes_=np.array([0,1]);return self
            def predict_proba(self,X):
                p=np.array([.99 if row[0] in self.seen else .2 for row in X]);return np.column_stack([1-p,p])
        X=np.arange(4.).reshape(-1,1);y=np.array([0,1,0,1]);splits=[([0,1],[2,3]),([2,3],[0,1])]
        np.testing.assert_allclose(namespace['generate_oof'](Memorizer(),X,y,splits),[.2]*4)
        try:namespace['generate_oof'](Memorizer(),X,y,[([0,1],[1,2])])
        except ValueError:pass
        else:raise AssertionError('Rejeter les folds qui partagent une ligne.')
    check('Soft Voting pondéré et poids égaux',voting)
    check('OOF avec modèle mémorisant, exclusion réelle des lignes',oof)

if exercise == "ch06_cours09":
    import numpy as np
    def costs():
        fn=namespace['compute_cost']
        assert fn([0,1,1,0],[.7,.9,.2,.1],.5)==530
        assert np.isclose(namespace['bayes_threshold'](),10/490)
        np.testing.assert_array_equal(namespace['optimal_actions']([.01,.03],[[0,500],[10,20]]),[0,1])
    def threshold():
        fn=namespace['select_threshold']
        t,cost=fn([1,1],[.01,.02]);assert t<=.01 and cost==40
        t,cost=fn([0,0],[.1,.9]);assert t>.9 and cost==0
        assert np.isclose(namespace['net_benefit']([1,0],[.8,.2],.5),.5)
    check('Matrice de coûts, risques attendus et seuil bayésien',costs)
    check('Validation incluant tout signaler / rien signaler',threshold)

if exercise == "ch06_cours10":
    import numpy as np
    def km():
        curve=namespace['kaplan_meier']([2,2,3],[1,0,1])
        np.testing.assert_allclose(curve['survival'],[2/3,0])
        assert curve.iloc[0]['at_risk']==3
        assert namespace['survival_at'](curve,1)==1
        assert np.isclose(namespace['survival_at'](curve,2),2/3)
        assert (namespace['kaplan_meier']([1,2,3],[0,0,0])['survival']==1).all()
    def concordance():
        fn=namespace['risk_concordance']
        assert fn([2,4,6],[1,0,1],[3,2,1])==1
        assert fn([2,4,6],[1,0,1],[1,2,3])==0
        assert fn([2,4],[1,1],[1,1])==.5
        assert np.isnan(fn([2,4,6],[0,0,1],[3,2,1]))
    check('Kaplan-Meier, événements et censures au même instant',km)
    check('Concordance, orientation du risque et paires censurées',concordance)

if exercise == "ch06_cours11":
    import numpy as np
    def ranking():
        fn=namespace['ranking_metrics']
        assert np.isclose(fn([3,2,1,0],[4,3,2,1],4)['NDCG'],1)
        assert fn([3,2,1,0],[1,2,3,4],4)['NDCG']<.7
        r=fn([1,0,1],[3,2,1],3);assert np.isclose(r['AP'],5/6) and r['recall']==1
        assert fn([0,0],[2,1],2)['NDCG']==0
        assert fn([1,0],[2,1],4)['precision']==.25
    def unseen():
        fn=namespace['recommend_unseen']
        np.testing.assert_array_equal(fn([.9,.8,.7,.6],[True,False,True,False],3),[1,3])
        assert len(fn([.9,.8],[True,True],2))==0
    check('NDCG exponentiel, AP@K et absence de pertinence',ranking)
    check('Exclusion des items consommés et catalogue épuisé',unseen)

if exercise == "ch06_cours12":
    import numpy as np
    import pandas as pd
    def audit():
        frame=pd.DataFrame({'x':[1.,1.,np.nan],'y':[0,0,1]});before=frame.copy(deep=True)
        report=namespace['audit_dataset'](frame,'y')
        assert report['duplicates']==1 and np.isclose(report['missing_rate']['x'],33.33)
        assert np.isclose(report['target_distribution'][0],2/3)
        pd.testing.assert_frame_equal(frame,before)
    def annotation():
        p=np.array([[1.,0.,0.],[.34,.33,.33],[.48,.47,.05]])
        assert namespace['select_for_annotation'](p,1,'entropy')[0]==1
        assert namespace['select_for_annotation'](p,1,'margin')[0]==1
        assert len(namespace['select_for_annotation'](p,0))==0
        np.testing.assert_array_equal(namespace['weak_majority']([[1,1,-1],[1,0,-1],[-1,-1,-1],[0,-1,0]]),[1,-1,-1,0])
    check('Audit de qualité et conservation des données',audit)
    check('Budget d’annotation, probabilités nulles et abstentions',annotation)
