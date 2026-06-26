import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def run_nested_cv(X_train, y_train):

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    nested_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [3, 5, None],
        "rf__criterion": ["gini", "entropy"] # Ağaç bölünme kalitesini ölçer
    }

    model = GridSearchCV(
        estimator=nested_pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="f1",
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=outer_cv,
        scoring="f1",
        n_jobs=-1
    )

    mean_f1 = scores.mean()
    std_f1 = scores.std() 
  
    print(f"F1 Kat Skorları: {np.round(scores, 4)}")
    print(f"Ortalama F1 Skoru: {mean_f1:.4f} (+/- {std_f1:.4f})")

    return mean_f1