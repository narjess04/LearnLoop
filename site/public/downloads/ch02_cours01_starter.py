import pandas as pd
from ch02_cours01_lab import df


def data_quality_report(df):
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean() * 100,
        "unique_values": df.nunique(dropna=True),
    })
    # TODO : moyenne, médiane, écart-type, minimum, maximum numériques.
    # TODO : trier par missing_rate décroissant.
    # BONUS : compter séparément valeurs absentes et violations métier.
    return report


if __name__ == "__main__":
    print(data_quality_report(df))
