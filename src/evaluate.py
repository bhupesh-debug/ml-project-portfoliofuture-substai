import os, pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_pipeline import load_fund_profiles, load_13f_holdings
from feature_engineering import extract_features

def evaluate():
    with open('models/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    df = load_fund_profiles()
    X, y = extract_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("=" * 55)
    print("  PORTFOLIOFUTURE-SUBSTAI EVALUATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=['Regular Fund', 'Prime Substitute']))
    print(f"  Holdout ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("=" * 55)
    
    df_13 = load_13f_holdings()
    total_switching_aum = df_13['position_value_usd'].sum()
    total_fee_drag = df_13['annual_fee_drag_usd'].sum()
    print("\n[INSTITUTIONAL 13F ADDRESSABLE MARKET ANALYSIS]")
    print(f"  Total RIA Position Capital Mapped:  ${total_switching_aum:,.2f}")
    print(f"  Annual Active Fee Drag on RIAs:     ${total_fee_drag:,.2f}")
    print(f"  Annual Addressable Client Savings:   ${total_fee_drag * 0.70:,.2f} (assuming 70% fee reduction)")
    print("=" * 55)

if __name__ == '__main__':
    evaluate()
