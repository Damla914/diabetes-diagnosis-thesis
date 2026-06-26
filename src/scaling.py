import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print("Eğitim seti yeni ortalama:", X_train_scaled.mean(axis=0).round(4))
    print("Eğitim seti yeni standart sapma:", X_train_scaled.std(axis=0).round(4)) 
  
    return X_train_scaled, X_test_scaled, scaler

def plot_scaling_boxplots(x_train: pd.DataFrame, x_train_scaled):
   
   plt.figure(figsize=(12,5))
   sns.boxplot(data=x_train, palette="Set3")
   plt.title("Ölçekleme Öncesi Özelliklerin Dağılımı", fontsize=12, pad=10)
   plt.xticks(rotation=45)
   plt.grid(axis='y', linestyle=':', alpha=0.5)
   plt.tight_layout()
   plt.show()

   plt.figure(figsize=(12,5))
   sns.boxplot(data=x_train_scaled, palette="Set2")
   plt.title("Ölçekleme Sonrası Özelliklerin Dağılımı", fontsize=12, pad=10)
   plt.xticks(rotation=45)
   plt.grid(axis='y', linestyle=':', alpha=0.5)
   plt.tight_layout()
   plt.show()
