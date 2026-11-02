import numpy as np
import pandas as pd


def feature_engineering(df):
    df = df.copy()
    # Colonnes : income, expenses, debt, age, account_open_date, observation_date.
    # TODO : disposable_income, expense_ratio, debt_to_income, income_log.
    # TODO : account_tenure_days, age_group, income_debt_interaction.
    # TODO : traiter les dénominateurs nuls et les dates incohérentes.
    return df

# BONUS : vérifier absence de ratios infinis, durées cohérentes et absence de mutation.
