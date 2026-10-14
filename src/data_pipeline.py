import pandas as pd
import os

def load_fund_profiles(path=None):
    if path is None:
        if os.path.exists('data/processed/fund_universe_profiles_real.csv'):
            path = 'data/processed/fund_universe_profiles_real.csv'
        else:
            path = 'data/synthetic/fund_universe_profiles.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    return pd.read_csv(path)

def load_13f_holdings(path=None):
    if path is None:
        if os.path.exists('data/processed/sec_13f_ria_ownership_real.csv'):
            path = 'data/processed/sec_13f_ria_ownership_real.csv'
        else:
            path = 'data/synthetic/sec_13f_ria_ownership.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    return pd.read_csv(path)

if __name__ == '__main__':
    df_f = load_fund_profiles()
    df_13 = load_13f_holdings()
    print(f"Data Pipeline loaded {len(df_f)} funds and {len(df_13)} 13F RIA holdings.")
