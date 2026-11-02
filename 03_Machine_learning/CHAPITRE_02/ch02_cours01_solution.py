import pandas as pd
from ch02_cours01_lab import df


def data_quality_report(df):
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean() * 100,
        "unique_values": df.nunique(dropna=True),
    })
    numeric = df.select_dtypes(include="number")
    for name, values in {
        "mean": numeric.mean(), "median": numeric.median(),
        "std": numeric.std(), "min": numeric.min(), "max": numeric.max(),
    }.items():
        report[name] = values.reindex(report.index)
    # Les absences sont déjà comptées ; elles ne sont pas des violations ici.
    rules = {
        "age": df["age"].notna() & (df["age"] < 0),
        "debt_ratio": df["debt_ratio"].notna() & ~df["debt_ratio"].between(0, 1),
        "default": df["default"].notna() & ~df["default"].isin([0, 1]),
    }
    report["rule_violations"] = pd.Series({k: int(v.sum()) for k, v in rules.items()}).reindex(report.index).astype("Int64")
    return report.sort_values("missing_rate", ascending=False)


if __name__ == "__main__":
    report = data_quality_report(df)
    print(report.to_string())
    assert report.loc["income", "missing_rate"] == 10
    assert report.loc["age", "rule_violations"] == 1
    assert report.loc["debt_ratio", "rule_violations"] == 1
    assert report.loc["default", "rule_violations"] == 0
    assert pd.isna(report.loc["region", "mean"])
    assert report["missing_rate"].is_monotonic_decreasing
