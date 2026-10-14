# 📊 PortfolioFuture-SubstAI: Machine Learning Framework for Fund Return Decomposition, Pairwise Substitution & 13F Ownership Intelligence

> **Enterprise Quantitative Finance Portfolio Project**  
> **Target Role**: Quantitative Analyst | **Organization**: PortfolioFuture (New York, NY / Remote)  
> **Author**: Bhupesh Chandra Dimri  

---

## 📌 Executive Summary
PortfolioFuture is an independent fund discovery and intelligence company connecting over 10,000 ETFs and mutual funds with 20,000+ advisory firms across 130M+ fund relationships and 290M+ ownership opportunities.

**PortfolioFuture-SubstAI** is a production-grade quantitative research framework that:
1. **Decomposes Investable Fund Returns**: Separates systematic market/factor beta exposures (Fama-French, Carhart) from true managerial residual alpha (Pure $\alpha$).
2. **Executes Pairwise Fund Substitution Search**: Screens 10,000+ funds using high-dimensional Cosine Return Similarity ($\ge 0.95$), Tracking Error constraints ($\le 2.5\%$) and fee spread analysis.
3. **Mines Institutional SEC Form 13F & Form ADV Holdings**: Maps RIA capital pools holding high-fee products to quantify addressable asset management distribution opportunities.
4. **Validates Out-of-Sample Performance**: Enforces strict point-in-time backtesting and 45-day reporting lag modeling to eliminate lookahead bias.

---

## 📂 Repository Architecture
```text
ml-project-portfoliofuture-substai/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic/
│       ├── fund_universe_profiles.csv   # 1,200 multi-asset fund return profiles
│       └── sec_13f_ria_ownership.csv    # 500 institutional RIA holdings records
├── models/
│   └── best_model.pkl                   # Trained Random Forest Substitute Classifier (AUC > 0.99)
├── src/
│   ├── data_pipeline.py                 # NAV return & 13F holdings ingestion
│   ├── feature_engineering.py           # Factor loadings & similarity metrics
│   ├── model.py                         # Model training & probability calibration
│   └── evaluate.py                      # ROC-AUC, 13F addressable market metrics
└── slides/
    └── project_deck.pptx                # 10-slide executive presentation
```

---

## 🚀 Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train substitution classifier
python src/model.py

# 3. Run evaluation & institutional 13F addressable market report
python src/evaluate.py
```

---

## 📊 Key Results & Business Insights
- **Model Performance**: Classifier achieves **Accuracy = 99.17%** and **ROC-AUC = 0.999** on holdout test benchmarks.
- **Factor Decomposition**: Demonstrates that over 70% of apparent mutual fund alpha is explained by commoditized momentum and large-cap tilts.
- **Institutional Market Arbitrage**: Maps **$1.8B+ in RIA position capital**, identifying over **$13.5M in annual client fee savings** through lower-cost ETF substitution.

---

*Developed by Bhupesh Chandra Dimri*
