import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

def run_comparative_tuning(lr_model, stack_model, X_test_scaled, y_test):

    y_probs_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_probs_stack = stack_model.predict_proba(X_test_scaled)[:, 1]

    precisions_lr, recalls_lr, thresholds_lr = precision_recall_curve(y_test, y_probs_lr)
    precisions_stack, recalls_stack, thresholds_stack = precision_recall_curve(y_test, y_probs_stack)

    f1_lr = 2 * (precisions_lr[:-1] * recalls_lr[:-1]) / (precisions_lr[:-1] + recalls_lr[:-1] + 1e-10)
    best_idx_lr = np.argmax(f1_lr)
    opt_thresh_lr = thresholds_lr[best_idx_lr]

    f1_stack = 2 * (precisions_stack[:-1] * recalls_stack[:-1]) / (precisions_stack[:-1] + recalls_stack[:-1] + 1e-10)
    best_idx_stack = np.argmax(f1_stack)
    opt_thresh_stack = thresholds_stack[best_idx_stack]

    print("\n=== KLİNİK EŞİK OPTİMİZASYON SONUÇLARI ===")
    print(f"Logistic Regression -> Optimal Eşik: {opt_thresh_lr:.4f} | En Yüksek F1: {f1_lr[best_idx_lr]:.4f}")
    print(f"Stacking Ensemble   -> Optimal Eşik: {opt_thresh_stack:.4f} | En Yüksek F1: {f1_stack[best_idx_stack]:.4f}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(recalls_lr, precisions_lr, label='Logistic Regression (Baseline)', color='darkblue', lw=2, alpha=0.8)
    plt.scatter(recalls_lr[best_idx_lr], precisions_lr[best_idx_lr], color='blue', s=120, zorder=5,
                edgecolors='black', label=f'LR Opt. Eşik ({opt_thresh_lr:.2f})')
    
    plt.plot(recalls_stack, precisions_stack, label='Stacking Ensemble (Şampiyon)', color='crimson', lw=2)
    plt.scatter(recalls_stack[best_idx_stack], precisions_stack[best_idx_stack], color='red', s=120, zorder=5,
                edgecolors='black', label=f'Stacking Opt. Eşik ({opt_thresh_stack:.2f})')
    
    plt.xlabel('Recall / Duyarlılık (Gerçek Hastaları Yakalama Oranı)', fontsize=11)
    plt.ylabel('Precision / Kesinlik (Pozitif Tahminlerin Doğruluk Oranı)', fontsize=11)
    plt.title('Modellerin Precision-Recall Eğrileri ve Optimal Karar Noktaları', fontsize=12, pad=12)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)

    test_thresholds = np.linspace(0.1, 0.9, 9)
    fn_lr_counts, fp_lr_counts = [], []
    fn_stack_counts, fp_stack_counts = [], []

    total_positives = np.sum(y_test == 1)

    for t in test_thresholds:
        cm_lr = confusion_matrix(y_test, (y_probs_lr >= t).astype(int))
        if cm_lr.shape == (2, 2):
            fp_lr_counts.append(cm_lr[0, 1])
            fn_lr_counts.append(cm_lr[1, 0])
        else:
            fp_lr_counts.append(0 if t > 0.5 else len(y_test) - total_positives)
            fn_lr_counts.append(total_positives if t > 0.5 else 0)

        cm_stack = confusion_matrix(y_test, (y_probs_stack >= t).astype(int))
        if cm_stack.shape == (2, 2):
            fp_stack_counts.append(cm_stack[0, 1])
            fn_stack_counts.append(cm_stack[1, 0])
        else:
            fp_stack_counts.append(0 if t > 0.5 else len(y_test) - total_positives)
            fn_stack_counts.append(total_positives if t > 0.5 else 0)

    plt.subplot(1, 2, 2)
    plt.plot(test_thresholds, fn_lr_counts, '--o', color='navy', label='LR Gözden Kaçan Hasta (FN)', lw=1.5)
    plt.plot(test_thresholds, fn_stack_counts, '-o', color='crimson', label='Stacking Gözden Kaçan Hasta (FN)', lw=2)
    
    plt.plot(test_thresholds, fp_lr_counts, '--s', color='lightblue', label='LR Yalancı Alarm (FP)', lw=1.5)
    plt.plot(test_thresholds, fp_stack_counts, '-s', color='orange', label='Stacking Yalancı Alarm (FP)', lw=2)

    plt.xlabel('Olasılık Karar Eşik Değeri (Threshold)', fontsize=11)
    plt.ylabel('Etkilenen Birey Sayısı', fontsize=11)
    plt.title('Eşik Değişimine Göre Klinik Hataların (FN vs FP) Ticari-Tıbbi Seyri', fontsize=12, pad=12)
    plt.legend(ncol=2, loc='upper center') 
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    return opt_thresh_lr, opt_thresh_stack