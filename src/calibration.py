import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

def run_calibration_analysis(lr_model, stack_model, X_test_scaled, y_test):

    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_stack = stack_model.predict_proba(X_test_scaled)[:, 1]

    brier_lr = brier_score_loss(y_test, y_prob_lr)
    brier_stack = brier_score_loss(y_test, y_prob_stack)

    print("\n=== KALİBRASYON VE GÜVENİLİRLİK ANALİZİ ===")
    print(f"Logistic Regression Brier Skoru : {brier_lr:.4f}")
    print(f"Stacking Ensemble Brier Skoru   : {brier_stack:.4f}")

    prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_prob_lr, n_bins=5, strategy='uniform')
    prob_true_stack, prob_pred_stack = calibration_curve(y_test, y_prob_stack, n_bins=5, strategy='uniform')

    fig, ax1 = plt.subplots(figsize=(7, 7))
    
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Mükemmel Kalibre Edilmiş (Teorik)')
    ax1.plot(prob_pred_lr, prob_true_lr, marker='o', linewidth=2, color='darkblue', 
             label=f'Logistic Regression (Brier: {brier_lr:.4f})')
    ax1.plot(prob_pred_stack, prob_true_stack, marker='s', linewidth=2, color='crimson', 
             label=f'Stacking Ensemble (Brier: {brier_stack:.4f})')

    ax1.set_xlabel("Tahmin Edilen Diyabet Olasılığı (Predicted Probability)", fontsize=11)
    ax1.set_ylabel("Gerçek Diyabet Oranı (True Probability)", fontsize=11)
    ax1.set_title("Modellerin Tahmin Güvenilirliği ve Kalibrasyon Eğrisi Karşılaştırması", fontsize=12, pad=12)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10)
    
    ax2 = ax1.inset_axes([0.05, 0.65, 0.35, 0.25])
    ax2.hist(y_prob_stack, bins=10, color='crimson', alpha=0.4, label='Stacking')
    ax2.hist(y_prob_lr, bins=10, color='darkblue', alpha=0.3, label='LR')
    ax2.set_title("Tahmin Yoğunluğu", fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    
    plt.tight_layout()
    plt.show()

    return brier_lr, brier_stack