import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import optuna

print("--- Loading Processed Data for LightGBM Ranker ---")
try:
    train_df_orig = pd.read_csv('data/processed_training_set.csv')
    test_df_orig = pd.read_csv('data/processed_test_set.csv')
except FileNotFoundError:
    print("Error: Processed data files not found. Please run data_process.py first.")
    exit()

train_df = train_df_orig.copy()
test_df = test_df_orig.copy()


# 1. 定义特征列和目标列
Y_train = train_df['relevance_score'].copy() # 创建副本以防后续意外修改

# 排除ID类、目标、原始目标/泄漏列和低重要性特征
# 根据最新的特征重要性图 (你提供的第二个图) 和之前分析，重新选择要排除的特征
# 目标是保留贡献大的，移除贡献小的，以简化模型并可能减少噪声
cols_to_exclude_from_features = [
    'srch_id', 'prop_id', 'relevance_score', 'click_bool', 'booking_bool',
    'gross_bookings_usd', 'position', 'date_time', 'date_time_dt',
    'total_guests', 'comp5_inv', 'comp8_rate', 'comp2_rate',
    'count_comp_rate_percent_diff_valid', 'comp2_inv', 'visitor_hist_starrating',
    'comp6_rate_percent_diff', 'comp4_rate_percent_diff', 'comp3_rate_percent_diff',
    'exp_historical_price', 'num_comp_cheaper', 'num_comp_more_expensive',
    'comp5_rate_percent_diff', 'orig_destination_distance',
    'sum_comp_rate_percent_diff_valid', 'srch_children_count'
]

feature_columns = [col for col in train_df.columns if col not in cols_to_exclude_from_features]

common_features = list(set(feature_columns) & set(test_df.columns))
print(f"Initial common features: {len(common_features)}")
if len(common_features) < len(feature_columns):
    missing_in_test = list(set(feature_columns) - set(common_features))
    print(f"Warning: Features in training but not in test (will be dropped from training features): {missing_in_test}")
    feature_columns = common_features

X_train = train_df[feature_columns].copy()
train_srch_ids_for_grouping = train_df['srch_id'].copy()
train_groups = train_df.groupby('srch_id', sort=False).size().to_numpy()

X_test = test_df[feature_columns].copy()
test_srch_ids_for_submission = test_df['srch_id'].copy()
test_prop_ids_for_submission = test_df['prop_id'].copy()

categorical_cols_to_encode = [
    'site_id', 'visitor_location_country_id', 'prop_country_id',
    'srch_destination_id', 'prop_starrating', 'prop_brand_bool',
    'dt_year', 'dt_month', 'dt_dayofweek', 'dt_hour',
    'promotion_flag', 'srch_saturday_night_bool', 'random_bool'
]

categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col in X_train.columns]

label_encoders = {}
for col in categorical_cols_to_encode:
    print(f"Label encoding column: {col}")
    le = LabelEncoder()
    # 合并训练集和测试集的特定列以确保编码一致性
    combined_col_data = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(combined_col_data)
    X_train.loc[:, col] = le.transform(X_train[col].astype(str))
    X_test.loc[:, col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

print(f"X_train shape after feature selection and encoding: {X_train.shape}")
print(f"X_test shape after feature selection and encoding: {X_test.shape}")


# 2. 使用 Optuna 进行超参数调优
print("\n--- Hyperparameter Tuning with Optuna ---")

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'label_gain': [0, 1, 5],
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'importance_type': 'gain',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100), # 增加迭代次数上限
        'num_leaves': trial.suggest_int('num_leaves', 20, 100, step=5),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50, step=5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True), # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True), # L2
    }

    # 定义类别特征 (在 objective 函数内部，因为 X_train 可能在外部被修改)
    current_categorical_features_for_lgbm = [col for col in categorical_cols_to_encode if col in X_train.columns]

    fold_ndcg_scores = []
    gkf = GroupKFold(n_splits=3) # 使用3折CV进行调优，可以减少时间

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, Y_train, groups=train_srch_ids_for_grouping.values)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

        train_fold_srch_ids = train_srch_ids_for_grouping.iloc[train_idx]
        val_fold_srch_ids = train_srch_ids_for_grouping.iloc[val_idx]
        train_fold_groups = train_fold_srch_ids.groupby(train_fold_srch_ids, sort=False).size().to_numpy()
        val_fold_groups = val_fold_srch_ids.groupby(val_fold_srch_ids, sort=False).size().to_numpy()

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(X_train_fold, Y_train_fold, group=train_fold_groups,
                   eval_set=[(X_val_fold, Y_val_fold)],
                   eval_group=[val_fold_groups],
                   eval_metric='ndcg',
                   eval_at=5,
                   callbacks=[lgb.early_stopping(100, verbose=False)], # 在调优时关闭详细日志
                   categorical_feature=current_categorical_features_for_lgbm)

        if ranker.best_score_ and 'valid_0' in ranker.best_score_ and 'ndcg@5' in ranker.best_score_['valid_0']:
            fold_ndcg_scores.append(ranker.best_score_['valid_0']['ndcg@5'])
        else:
            fold_ndcg_scores.append(0) # 惩罚无法获取分数的尝试

    avg_ndcg = np.mean(fold_ndcg_scores)
    print(f"Trial finished with avg_ndcg@5: {avg_ndcg:.4f} for params: {trial.params}")
    return avg_ndcg # Optuna 会最大化这个返回值

# # 运行 Optuna 研究
# study = optuna.create_study(direction='maximize', study_name='lgbm_ranker_tuning')
# study.optimize(objective, n_trials=10)

# print("\nBest trial for Optuna:")
# best_trial = study.best_trial
# print(f"  Value (avg NDCG@5): {best_trial.value:.4f}")
# print("  Params: ")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")

# best_params_from_optuna = best_trial.params

print("Skipping Optuna tuning, using pre-defined or improved parameters.")
best_params_from_optuna = { # 这是一个示例，你应该使用Optuna找到的参数
    'learning_rate': 0.02319,
    'n_estimators': 1300,
    'num_leaves': 90,
    'max_depth': 15,
    'min_child_samples': 15,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.05,
    'reg_lambda': 0.55,
    
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'label_gain': [0, 1, 5],
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'importance_type': 'gain'
}


# 3. 使用找到的最佳参数或一组改进的参数进行最终模型训练
print("\n--- Training Final LightGBM Ranker with Best/Improved Params ---")
final_categorical_features = [col for col in categorical_cols_to_encode if col in X_train.columns] # 确保使用正确的特征集
print(f"Final categorical features for LGBM: {final_categorical_features}")

final_n_splits = 3 # 可以使用更多的折数进行最终训练评估，例如5
final_group_kfold = GroupKFold(n_splits=final_n_splits)
final_models = []
final_oof_ndcg_scores = []

print(f"Starting {final_n_splits}-Fold Cross-Validation for final model...")
for fold, (train_idx, val_idx) in enumerate(final_group_kfold.split(X_train, Y_train, groups=train_srch_ids_for_grouping.values)):
    print(f"--- Final Fold {fold+1}/{final_n_splits} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

    train_fold_srch_ids = train_srch_ids_for_grouping.iloc[train_idx]
    val_fold_srch_ids = train_srch_ids_for_grouping.iloc[val_idx]
    train_fold_groups = train_fold_srch_ids.groupby(train_fold_srch_ids, sort=False).size().to_numpy()
    val_fold_groups = val_fold_srch_ids.groupby(val_fold_srch_ids, sort=False).size().to_numpy()

    ranker = lgb.LGBMRanker(**best_params_from_optuna) # 使用最佳参数
    ranker.fit(X_train_fold, Y_train_fold, group=train_fold_groups,
               eval_set=[(X_val_fold, Y_val_fold)],
               eval_group=[val_fold_groups],
               eval_metric='ndcg',
               eval_at=5,
               callbacks=[lgb.early_stopping(150, verbose=50)], # 可以增加耐心
               categorical_feature=final_categorical_features)
    final_models.append(ranker)
    if ranker.best_score_ and 'valid_0' in ranker.best_score_ and 'ndcg@5' in ranker.best_score_['valid_0']:
        best_ndcg = ranker.best_score_['valid_0']['ndcg@5']
        final_oof_ndcg_scores.append(best_ndcg)
        print(f"Fold {fold+1} Best NDCG@5: {best_ndcg:.4f} at iteration {ranker.best_iteration_}")

print(f"\nFinal Model CV Finished. Average OOF NDCG@5: {np.mean(final_oof_ndcg_scores):.4f} (+/- {np.std(final_oof_ndcg_scores):.4f})")

# 4. 特征重要性
if final_models:
    lgb.plot_importance(final_models[0], figsize=(10, 12), max_num_features=len(X_train.columns), importance_type='gain') # 显示所有特征
    plt.title("LightGBM Feature Importance (Final Model - Fold 1)")
    plt.tight_layout()
    plt.show()

# 5. 进行预测和生成提交文件
print("\n--- Making Predictions on Test Set (LightGBM Final Model) ---")
test_scores_lgbm_final = np.zeros(len(X_test))
if final_models:
    for model in final_models:
        test_scores_lgbm_final += model.predict(X_test) / final_n_splits
else:
    print("Error: No final models were trained.")
    exit()

submission_df_lgbm_final = pd.DataFrame({
    'srch_id': test_srch_ids_for_submission,
    'prop_id': test_prop_ids_for_submission,
    'score': test_scores_lgbm_final
})
submission_df_lgbm_final = submission_df_lgbm_final.sort_values(['srch_id', 'score'], ascending=[True, False])

final_submission_list_lgbm_final = [{'srch_id': int(r['srch_id']), 'prop_id': int(r['prop_id'])} for _, r in submission_df_lgbm_final.iterrows()]
final_submission_df_lgbm_final = pd.DataFrame(final_submission_list_lgbm_final)


submission_filename_lgbm_final = 'submission_lightgbm_ranker_tuned.csv'
final_submission_df_lgbm_final.to_csv(submission_filename_lgbm_final, index=False)
print(f"LightGBM Tuned Submission file '{submission_filename_lgbm_final}' created successfully.")