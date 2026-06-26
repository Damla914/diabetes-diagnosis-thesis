import warnings
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
plt.rcParams['figure.max_open_warning'] = 100 

from data_check import load_data, basic_info, plot_outcome_distribution, plot_corr, plot_box_insulin, plot_box_dbf
from decision_tree import run_decision_tree
from knn import run_knn
from comparative_tuning import run_comparative_tuning
from random_forest import run_random_forest
from nested_cv import run_nested_cv
from preprocess import clean_data, data_clean_basic_info, plot_data_clean_hist, plot_data_clean_corr, plot_feature_boxplots, print_data_groupby_summary
from scaling import  scale_data, plot_scaling_boxplots
from train_test_split import split, show_plot_distribution
from logistic_regression import run_logistic_regression
from smote import train_smote, plot_smote_balance
from stacking import run_stacking_ensemble
from compare_models import compare_models, plot_model_comparison
from calibration import run_calibration_analysis
from shap_analysis import run_shap_analysis


all_metrics = []

print("----------VERİ KONTROLÜ (DATA CHECK)----------")
data=load_data()
basic_info(data)
plot_outcome_distribution(data)
plot_corr(data)
plot_box_insulin(data)
plot_box_dbf(data)

print("\n----------ÖN İŞLEME (PREPROCESSING)----------")
data_clean=clean_data()
data_clean_basic_info(data_clean)
print_data_groupby_summary(data_clean)
plot_data_clean_hist(data_clean)
plot_data_clean_corr(data_clean)
plot_feature_boxplots(data_clean)

print("\n----------VERİ KÜMESİ BÖLÜNMESİ (TRAIN TEST SPLIT)----------")
X_train, X_test, y_train, y_test = split(data_clean)
show_plot_distribution(y_train, y_test)

print("\n----------ÖLÇEKLEME (SCALING)----------")
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
plot_scaling_boxplots(X_train, X_train_scaled)

print("\n----------SINIF DENGELEME (SMOTE)------------")
X_train_smote, y_train_smote = train_smote(X_train_scaled, y_train)
plot_smote_balance(y_train, y_train_smote)

print("\n----------İÇ İÇE ÇAPRAZ DOĞRULAMA (NESTED CV)------------")
mean_f1 = run_nested_cv(X_train_smote, y_train_smote)

print("\n---------- MODEL KARŞILAŞTIRMALARI (MODEL COMPARISON) ------------")

all_metrics = []


print("Baseline Modelleri:")

lr_metrics_list, ham_lr_model, balanced_lr_model = run_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
for name, metrics in lr_metrics_list: 
    all_metrics.append((f"{name} (Baseline)", metrics))

for name, metrics in run_knn(X_train_scaled, y_train, X_test_scaled, y_test):
    all_metrics.append((f"{name} (Baseline)", metrics))

for name, metrics in run_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test):
    all_metrics.append((f"{name} (Baseline)", metrics))

for name, metrics in run_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    all_metrics.append((f"{name} (Baseline)", metrics))


print("SMOTE Entegre Edilmiş Modeller:")

lr_smote_list, _, _ = run_logistic_regression(X_train_smote, y_train_smote, X_test_scaled, y_test)
for name, metrics in lr_smote_list: 
    all_metrics.append((f"{name} (SMOTE)", metrics))

for name, metrics in run_knn(X_train_smote, y_train_smote, X_test_scaled, y_test):
    all_metrics.append((f"{name} (SMOTE)", metrics))

for name, metrics in run_decision_tree(X_train_smote, y_train_smote, X_test_scaled, y_test):
    all_metrics.append((f"{name} (SMOTE)", metrics))

for name, metrics in run_random_forest(X_train_smote, y_train_smote, X_test_scaled, y_test):
    all_metrics.append((f"{name} (SMOTE)", metrics))

print("Stacking Ensemble Modeli:")
stack_metrics_list, stack_model = run_stacking_ensemble(X_train_smote, y_train_smote, X_test_scaled, y_test)
for name, metrics in stack_metrics_list: 
    all_metrics.append((f"{name} (SMOTE)", metrics))

print("----------NİHAİ MODEL KARŞILAŞTIRMA VE RAPORLAMA (ULTIMATE COMPARISON & REPORTING)------------")
df_comparison, best_model = compare_models(all_metrics, sort_by="f1")
plot_model_comparison(df_comparison, metric_to_plot="f1")

print("\n----------KARŞILAŞTIRMALI KLİNİK ANALİZİ (COMPARATIVE CLINICAL ANALYSIS)---------")
_ = run_comparative_tuning(balanced_lr_model, stack_model, X_test_scaled, y_test)

print("\n---------- KALİBRASYON ANALİZİ (CALIBRATION ANALYSIS)------------")
_ = run_calibration_analysis(balanced_lr_model, stack_model, X_test_scaled, y_test)

print("\n---------- SHAP ANALİZİ (SHAP ANALYSIS)------------")
feature_names = data_clean.drop(columns=['Outcome']).columns.tolist()
run_shap_analysis(balanced_lr_model, X_train_scaled, X_test_scaled, feature_names)
