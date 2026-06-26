from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

def train_smote(X_train, y_train):
    
    smote = SMOTE(random_state=42) 
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

def plot_smote_balance(y_train, y_train_smote):
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    counts_before = y_train.value_counts()
    bars1 = counts_before.plot(kind='bar', color=['darkblue', 'darkorange'])
    plt.title("SMOTE Öncesi Sınıf Dağılımı")
    plt.xlabel("Diyabet Durumu (0: Yok, 1: Var)")
    plt.ylabel("Örnek Sayısı")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    for bar in bars1.patches:
        plt.annotate(f'{int(bar.get_height())}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height() - 25),
                     ha='center', va='center', color='white', fontweight='bold')

    plt.subplot(1, 2, 2)
    counts_after = y_train_smote.value_counts()
    bars2 = counts_after.plot(kind='bar', color=['darkblue', 'darkorange'])
    plt.title("SMOTE Sonrası Sınıf Dağılımı")
    plt.xlabel("Diyabet Durumu (0: Yok, 1: Var)")
    plt.ylabel("Örnek Sayısı")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    for bar in bars2.patches:
        plt.annotate(f'{int(bar.get_height())}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height() - 25),
                     ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()
