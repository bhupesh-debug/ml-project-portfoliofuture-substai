import os, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_pipeline import load_fund_profiles
from feature_engineering import extract_features

def train():
    df = load_fund_profiles()
    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Random Forest Substitute Classifier -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    out_path = 'models/best_model.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {out_path}")
    return clf

if __name__ == '__main__':
    train()
