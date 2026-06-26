from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def run_stacking_ensemble(X_train_smote, y_train_smote, X_test_scaled, y_test):

    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("knn", KNeighborsClassifier())
    ]

    stack = StackingClassifier(
        estimators=estimators, 
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        cv=5,
        n_jobs=-1
    )

    stack.fit(X_train_smote, y_train_smote)

    y_pred = stack.predict(X_test_scaled)
    y_proba = stack.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"-------STACKING ENSEMBLE SONUÇLARI--------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    metrics_list = [("Stacking Ensemble", metrics)]

    return metrics_list, stack