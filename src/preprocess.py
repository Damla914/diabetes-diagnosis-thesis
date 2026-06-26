import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from data_check import load_data

def clean_data(save_path=None):

    data = load_data()
    data_clean = data.copy()
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_cols:
        data_clean[col] = data_clean[col].replace(0, np.nan)

    for col in zero_cols:
        median_value = data_clean[col].median()
        data_clean[col] = data_clean[col].fillna(median_value)

    print("\nNull values:")
    print(data_clean.isnull().sum())

    upper_limit = 400
    data_clean["Insulin"] = np.where(data_clean["Insulin"] > upper_limit, upper_limit, data_clean["Insulin"])

    print("\nClean Data Describe:")
    print(data_clean.describe())
   
    if save_path is None:
         base_dir = Path(__file__).resolve().parent.parent
         save_path = base_dir/"data"/"diabetes_clean.csv"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data_clean.to_csv(save_path,index=False)
    print("Temiz veri kaydedildi.")

    return data_clean

def data_clean_basic_info(data_clean):
    print("Veri Bilgisi:")
    print(data_clean.info())
    print("Veri Boyutu:")
    print(data_clean.shape)
    print("Null Değerler:")
    print(data_clean.isnull().sum())

def plot_data_clean_hist(data_clean):
    data_clean.hist(figsize=(12,8), color='orange', edgecolor='black', alpha=0.7)
    plt.suptitle("Sayısal Özelliklerin Dağılımları", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_data_clean_corr(data_clean):
    plt.figure(figsize=(10,6))
    sns.heatmap(data_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Ön İşleme Sonrası Değişkenler Arası Korelasyon Matrisi")
    plt.tight_layout()
    plt.show()

def plot_feature_boxplots(data_clean):
    features = ["Glucose", "BMI", "Age", "Insulin"]
    plt.figure(figsize=(12,8))
    for i, col in enumerate(features, 1):
        plt.subplot(2,2,i)
        sns.boxplot(x="Outcome", y=col, data=data_clean, palette="Set2")
        plt.title(f"{col} vs Outcome")
    plt.tight_layout()
    plt.show()

def print_data_groupby_summary(data_clean):
    print("Diyabet Durumuna Göre Klinitk Ortalamalar:")
    group_summary = data_clean.groupby("Outcome")[["Glucose", "BMI", "Age", "Insulin"]].mean()
    print(group_summary)
    plt.figure(figsize=(8, 5))
    sns.histplot(data_clean["Insulin"], kde=True, bins=30, color='orange')
    plt.title("Insulin Değerlerinin Dağılımı")
    plt.xlabel("Insulin Seviyesi")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()
