import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "diabetes.csv"
    return pd.read_csv(data_path)

def basic_info(data):
    print("Veri Bilgileri:")
    data.info()
    print("Veri Boyutu:")
    print(f"Satır Sayısı: {data.shape[0]} | Sütun Sayısı: {data.shape[1]}")
    print("Null Değerler:")
    print(data.isnull().sum())
    print("Geçersiz Sıfır Değerler:")
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    for col in invalid_zero_cols:
        zero_count = (data[col] == 0).sum()
        zero_percentage = (zero_count / len(data)) * 100
        print(f"{col:<20}: {zero_count:<5} adet (%{zero_percentage:.2f})")

def plot_outcome_distribution(data):
    plt.figure(figsize=(6, 5))
    counts = data["Outcome"].value_counts()
    bars = counts.plot(kind='bar', color=['darkgreen', 'purple'])
    for bar in bars.patches:
        plt.annotate(f'{int(bar.get_height())}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height() - 40),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12) 
    plt.title("Diyabet Sınıf Dağılımı (Sınıf Dengesizliği Analizi)", fontsize=12, pad=15)
    plt.xticks([0, 1], ['Sağlıklı (0)', 'Diyabet hastası (1)'], rotation=0)
    plt.ylabel("Hasta Sayısı")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_corr(data):
    corr = data.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Korelasyon Matrisi (Özellikler Arası İlişki Analizi)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.show()

def plot_box_insulin(data):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data["Insulin"], color='skyblue')
    plt.title("Insulin Değişkeni Aykırı Değer (Outlier) İncelemesi", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_box_dbf(data):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data["DiabetesPedigreeFunction"], color='plum')
    plt.title("Diabetes Pedigree Function Aykırı Değer İncelemesi", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()