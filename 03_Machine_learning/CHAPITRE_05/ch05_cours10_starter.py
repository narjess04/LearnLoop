"""Complète le moteur de triage. Une métrique absente/invalide doit bloquer."""
def evaluate_model(metrics, psi, rules=None, critical_findings=None):
    # TODO : config par défaut : auc (.80,.70), brier (.05,.08), psi (.10,.25).
    # TODO : valider les données et les seuils ; absent/None/NaN => RED.
    # TODO : agréger GREEN < AMBER < RED et enregistrer les findings.
    # TODO : critical_findings non vide => RED.
    # Retourner status, criteria, findings, requires_human_review=True.
    raise NotImplementedError("Complète evaluate_model.")

def run():
    import ch05_cours10_solution as benchmark
    benchmark.evaluate_model = evaluate_model
    return benchmark.run()

if __name__ == "__main__":
    run()
