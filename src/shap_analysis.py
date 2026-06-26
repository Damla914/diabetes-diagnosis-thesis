import matplotlib.pyplot as plt
import shap

def run_shap_analysis(lr_model, X_train_scaled, X_test_scaled, feature_names):

    explainer = shap.LinearExplainer(lr_model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    shap_values.feature_names = list(feature_names)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    
    plt.gca().set_title("Klinik Özelliklerin Diyabet Teşhisine Etki Dağılımı", fontsize=12, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

    shap.plots.bar(shap_values, show=False)
    
    plt.gca().set_title("Diyabet Teşhisinde En Önemli Klinik Faktörler", fontsize=12, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.show()