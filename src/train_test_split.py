from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import clean_data

def split(data_clean, out_dir=None):
   X = data_clean.drop("Outcome", axis=1) 
   y = data_clean["Outcome"]              
   print("X shape:", X.shape)
   print("y shape:", y.shape) 

   print("X data types:",X.dtypes) 
   print("X summary",X.describe().T[["mean", "std", "min", "max"]])

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)

   print("Eğitim seti boyutu:", X_train.shape)
   print("Test seti boyutu:", X_test.shape)

   print("\nEğitim setinde sınıf dağılımı:")
   print(y_train.value_counts(normalize=True))

   print("\nTest setinde sınıf dağılımı:")
   print(y_test.value_counts(normalize=True)) 
   
   if out_dir is None:
       base_dir = Path(__file__).resolve().parent.parent 
       out_dir = base_dir/"data"/"splits"
   else:
       out_dir = Path(out_dir)
   
   out_dir.mkdir(parents=True, exist_ok=True) 

   X_train.to_csv(out_dir / "X_train.csv", index=False)
   X_test.to_csv(out_dir / "X_test.csv", index=False)
   y_train.to_csv(out_dir / "y_train.csv", index=False)
   y_test.to_csv(out_dir / "y_test.csv", index=False)
   print("\nDosyalar kaydedildi.")

   return X_train, X_test, y_train, y_test 

def show_plot_distribution(y_train, y_test):
    plt.figure(figsize=(10, 4))

    # 1. Grafik: Eğitim Seti Dağılımı 
    plt.subplot(1, 2, 1)
    counts_train = y_train.value_counts()
    bars1 = counts_train.plot(kind='bar', color=['darkblue', 'salmon'])
    plt.title("Eğitim Seti Sınıf Dağılımı")
    plt.xlabel("Diyabet Durumu (0: Yok, 1: Var)")
    plt.ylabel("Hasta Sayısı")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    for bar in bars1.patches:
        plt.annotate(f'{int(bar.get_height())}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height() - 30),
                     ha='center', va='center', color='white', fontweight='bold')

    # 2. Grafik: Test Seti Dağılımı 
    plt.subplot(1, 2, 2)
    counts_test = y_test.value_counts()
    bars2 = counts_test.plot(kind='bar', color=['darkblue', 'salmon'])
    plt.title("Test Seti Sınıf Dağılımı")
    plt.xlabel("Diyabet Durumu (0: Yok, 1: Var)")
    plt.ylabel("Hasta Sayısı") # İkinci grafiğe de eksen ismi eklendi
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    for bar in bars2.patches:
        plt.annotate(f'{int(bar.get_height())}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height() - 10),
                     ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()