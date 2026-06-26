import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_models(metrics_list, sort_by="f1"):

    formatted_list = []
    for name, metrics in metrics_list:
        row = {"model": name}
        for metric_name in ["accuracy", "precision", "recall", "f1", "auc"]:
            row[metric_name] = metrics.get(metric_name, np.nan)
        formatted_list.append(row)

    df = pd.DataFrame(formatted_list)

    col_order = ["model", "accuracy", "precision", "recall", "f1", "auc"]
    df = df[col_order]

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    print("\n========== MODEL KARŞILAŞTIRMA TABLOSU ==========")
    print(df.fillna("-").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    best_row = df.iloc[0].to_dict()
    print(f" TEŞHİS BAŞARISI EN YÜKSEK MODEL (Sıralama: {sort_by.upper()})")
    print(f"Model Adı : {best_row['model']}")
    print(f"F1-Skoru  : {best_row['f1']:.4f} | AUC: {best_row['auc']:.4f} | Doğruluk: {best_row['accuracy']:.4f}")

    return df, best_row

def plot_model_comparison(df, metric_to_plot="f1"):
    plt.figure(figsize=(12, 6))
    df_plot = df.dropna(subset=[metric_to_plot]).copy()
    ax = sns.barplot(
        x=metric_to_plot, 
        y="model", 
        data=df_plot, 
        palette="viridis", 
        hue="model", 
        legend=False
    )
    for p in ax.patches:
        width = p.get_width()
        if width > 0: # Boş barları atla
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.4f}', 
                    va='center', ha='left', fontsize=10, fontweight='bold')
    plt.title(f"Modellerin {metric_to_plot.upper()} Skorlarına Göre Karşılaştırmalı Analizi", fontsize=13, pad=15)
    plt.xlabel(metric_to_plot.upper(), fontsize=11)
    plt.ylabel("Uygulanan Algoritma ve Strateji", fontsize=11)
    plt.xlim(0, 1.1) # Sayıların sığması için sınırı biraz büyüttük
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()