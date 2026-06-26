from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, max_depth=None):

    rf = RandomForestClassifier(n_estimators=200, max_depth=max_depth, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def evaluate_rf(rf, X_test, y_test, title="Random Forest"):

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Auc  : {auc:.4f}")

    metrics = {
         "accuracy": accuracy,
         "precision": precision,
         "recall": recall,
         "f1": f1,
         "auc": auc,
    }

    return y_pred, y_prob, metrics

def run_random_forest(X_train, y_train, X_test, y_test):
    
    max_depth = [None, 5, 10]
    metrics_list = []

    print("\n----------RANDOM FOREST----------")
    for m in max_depth:
         model = train_random_forest(X_train, y_train, m)
         print(f"\nMax Depth : {m}")
         y_pred, y_prob, metrics = evaluate_rf(model, X_test, y_test)
         metrics_list.append((f"Random Forest (max depth = {m})", metrics))
   
    return metrics_list
