import pandas as pd

def extract_features(df):
    features = [
        'expense_ratio_bps', 'beta_market', 'beta_size_smb',
        'beta_value_hml', 'beta_momentum', 'pure_alpha_pct',
        'tracking_error_pct', 'cosine_similarity'
    ]
    return df[features], df['is_prime_substitute']

if __name__ == '__main__':
    from data_pipeline import load_fund_profiles
    df = load_fund_profiles()
    X, y = extract_features(df)
    print(f"Feature matrix shape: {X.shape}, labels: {y.shape}")
