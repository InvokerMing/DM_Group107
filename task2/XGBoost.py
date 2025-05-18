import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import re

print("--- Loading Processed Data for XGBoost Ranker ---")
try:
    train_df = pd.read_csv('data/processed_training_set_2.csv')
    test_df = pd.read_csv('data/processed_test_set_2.csv')
except FileNotFoundError:
    print("Error: Processed data files not found. Please run data_process.py first.")
    exit()

# 1. 定义特征列和目标列
Y_train = train_df['relevance_score']

# 初始排除列列表
cols_to_exclude_initial = [
    'srch_id', 'prop_id', 'relevance_score', 'click_bool', 'booking_bool',
    'gross_bookings_usd', 'position', 'date_time', # 确保原始date_time被排除
    # 'date_time_dt' # 这个列似乎不存在，移除
    # 任何在data_process.py中创建的、但明确不想作为特征的中间列也应在此添加
]

# --- 特征选择 ---
# 基于EDA和上一轮的特征重要性图，选择约50-60个特征
# （注意：这个列表是根据提供的图和通用实践选择的，可能需要根据实际运行和进一步分析调整）
selected_features = [
    # 从特征重要性图 (Top ~40)
    'prop_location_score2_missing', 'promotion_flag', 'prop_location_score2', 'price_ratio_hist',
    'prop_location_score1', 'prop_starrating_x_review_score', 'prop_starrating', 'random_bool',
    'price_usd', 'price_x_prop_location_score2', 'comp8_inv_missing', 'price_per_night',
    'prop_review_score', 'location_score_combined', 'prop_country_id', 'srch_room_count',
    'prop_brand_bool', 'comp8_rate_missing', 'visitor_hist_adr_usd', 'visitor_hist_adr_usd_missing',
    'srch_query_affinity_score', 'prop_log_historical_price', 'starrating_diff_from_hist',
    'visitor_hist_starrating_missing', 'comp8_rate', 'exp_historical_price', 'comp1_rate_missing',
    'visitor_hist_starrating', 'comp5_rate', 'comp3_rate_missing', 'srch_children_count',
    'comp5_inv_missing', 'num_comp_cheaper', 'avg_comp_rate_percent_diff_abs',
    'sum_comp_rate_percent_diff_abs_valid', 'srch_booking_window', 'comp1_inv_missing', 'site_id',
    'num_comp_more_expensive', # 约39个来自图示

    # 添加其他重要特征
    'dt_month', 'dt_dayofweek', 'dt_hour',               # 时间特征
    'srch_length_of_stay', 'srch_adults_count',         # 搜索参数
    'total_guests',                                      # 衍生搜索参数
    'prop_review_score_missing',                         # 其他重要缺失指标
    'srch_query_affinity_score_missing',
    'orig_destination_distance_missing',
    'prop_log_historical_price_missing',
    'visitor_location_country_id',                       # 用户位置
    'orig_destination_distance',                         # 原始特征（填充后）
    'price_diff_from_hist_abs',                          # 价格差异
    'num_comp_available_info',                           # 竞争对手信息计数
    'num_comp_inv_unavailable',
    'comp2_rate', 'comp2_rate_missing',                  # 其他竞争对手特征样例
    'comp3_rate', 'comp3_rate_missing',
    'comp2_inv_missing', 'comp3_inv_missing',
    'prop_starrating_missing', # 如果在data_process中添加了这个，则包含，否则移除
    'comp1_rate', 'comp4_rate', 'comp6_rate', 'comp7_rate', # 其他comp rates
]

# 确保selected_features中的列存在于train_df中，并移除cols_to_exclude_initial中的列
feature_columns = [col for col in selected_features if col in train_df.columns and col not in cols_to_exclude_initial]
# 去重，以防selected_features中有重复
feature_columns = sorted(list(set(feature_columns)))


# 使用共同特征 (以防测试集缺少某些列，尽管data_process应该保持一致)
common_features = list(set(feature_columns) & set(test_df.columns))
feature_columns = common_features

X_train = train_df[feature_columns]
train_srch_ids_for_grouping = train_df['srch_id']
train_groups = train_df.groupby('srch_id', sort=False).size().to_numpy()

if not set(feature_columns).issubset(set(test_df.columns)):
    print("Warning: Test set is missing some feature columns present in training set after selection.")
    # Fallback or error, here we'll use only common features available in test_df for X_test
    X_test = test_df[list(set(feature_columns) & set(test_df.columns))]
else:
    X_test = test_df[feature_columns]

test_srch_ids_for_submission = test_df['srch_id']
test_prop_ids_for_submission = test_df['prop_id']

# --- 内存优化：数据类型转换 ---
print("Optimizing memory by converting data types...")
for col in X_train.columns:
    if X_train[col].dtype == 'float64':
        X_train[col] = X_train[col].astype('float32')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('float32')
    elif X_train[col].dtype == 'int64':
        # 检查是否可以安全转换为int32
        if X_train[col].min() >= np.iinfo(np.int32).min and X_train[col].max() <= np.iinfo(np.int32).max:
            X_train[col] = X_train[col].astype('int32')
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('int32')

print(f"X_train shape after selection & type conversion: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape after selection & type conversion: {X_test.shape}")
if X_train.shape[1] != X_test.shape[1]:
    print(f"CRITICAL WARNING: X_train and X_test have different number of features: {X_train.shape[1]} vs {X_test.shape[1]}")
    # This can happen if common_features logic above results in different feature sets.
    # Re-align columns to ensure they are identical for DMatrix creation.
    shared_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[shared_cols]
    X_test = X_test[shared_cols]
    feature_columns = shared_cols # Update feature_columns to the truly shared ones
    print(f"Re-aligned X_train and X_test to {len(shared_cols)} features.")


# XGBoost 对列名中的特殊字符敏感，进行清理
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns]


# 2. 模型训练
print("\n--- Training XGBoost Ranker ---")
# XGBoost Ranker 参数
# 注意：XGBoost 的 'rank:ndcg' 通常需要 Y 是非负整数，且它内部会处理 gain
# 它不像 LightGBM 那样直接有 label_gain 参数，但可以通过设置权重或自定义目标函数来模拟
# 对于 rank:pairwise, Y 通常是0/1。对于 rank:ndcg, Y 是相关性等级。
xgb_ranker_params = {
    'objective': 'rank:ndcg',    # 或者 'rank:pairwise' (Y需要是0/1)
    'eval_metric': 'ndcg@5',     # XGBoost 使用 '@'
    'eta': 0.025,                 # learning_rate
    'max_depth': 6,              # 略微增加
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'gamma': 0.1,
    'lambda': 2,                 # L2 regularization
    'alpha': 0.1,                  # L1 regularization
    'seed': 42,
    'tree_method': 'hist',       # 使用 hist 加速，如果数据量大
    # 'enable_categorical': True # 如果使用XGBoost >= 1.3.0 并且有类别特征(需要数值编码)
}


n_splits = 2
group_kfold = GroupKFold(n_splits=n_splits)
models_xgb = []
oof_ndcg_scores_xgb = []

print(f"Starting {n_splits}-Fold Cross-Validation for XGBoost...")
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, Y_train, groups=train_srch_ids_for_grouping.values)):
    print(f"--- XGBoost Fold {fold+1}/{n_splits} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

    # XGBoost 需要 DMatrix，并且 group 信息在 DMatrix 中设置
    dtrain = xgb.DMatrix(X_train_fold, label=Y_train_fold, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val_fold, label=Y_val_fold, feature_names=feature_columns)

    # 设置 group 信息
    train_fold_srch_ids = train_srch_ids_for_grouping.iloc[train_idx]
    val_fold_srch_ids = train_srch_ids_for_grouping.iloc[val_idx]
    train_fold_groups_xgb = train_fold_srch_ids.groupby(train_fold_srch_ids, sort=False).size().to_numpy()
    val_fold_groups_xgb = val_fold_srch_ids.groupby(val_fold_srch_ids, sort=False).size().to_numpy()
    dtrain.set_group(train_fold_groups_xgb)
    dval.set_group(val_fold_groups_xgb)

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    eval_results_history = {} # Initialize dict to store evaluation results

    model = xgb.train(
        xgb_ranker_params,
        dtrain,
        num_boost_round=2500, # 增加轮数
        evals=watchlist,
        early_stopping_rounds=150, # 增加早停轮数
        verbose_eval=50,
        evals_result=eval_results_history # 传递字典以捕获历史记录
    )
    models_xgb.append(model)

    # 从eval_results_history中获取最佳NDCG@5
    metric_name = xgb_ranker_params.get('eval_metric', 'ndcg@5') # 确保使用配置的指标
    if model.best_iteration > 0 : # best_iteration is 1-based
        if 'eval' in eval_results_history and metric_name in eval_results_history['eval']:
            best_ndcg_val = eval_results_history['eval'][metric_name][model.best_iteration - 1]
            oof_ndcg_scores_xgb.append(best_ndcg_val)
            print(f"Fold {fold+1} Best Eval {metric_name}: {best_ndcg_val:.4f} at iteration {model.best_iteration}")
        elif model.best_score is not None: # Fallback to model.best_score if specific metric not in history (should not happen)
            print(f"Warning: Metric '{metric_name}' not found directly in evals_result for fold {fold+1}. Using model.best_score.")
            oof_ndcg_scores_xgb.append(model.best_score)
            print(f"Fold {fold+1} Best Eval {metric_name} (from model.best_score): {model.best_score:.4f} at iteration {model.best_iteration}")
        else:
            print(f"Error: No best score or specific metric value found for fold {fold+1} after early stopping.")
            # Append NaN or handle as error, e.g. by not appending, which will affect np.mean later
    else:
        # This case means early stopping might not have triggered effectively, or num_boost_round was hit.
        # Use the last available score.
        print(f"Warning: Early stopping may not have triggered or num_boost_round reached for fold {fold+1}. Using last score.")
        if 'eval' in eval_results_history and metric_name in eval_results_history['eval'] and len(eval_results_history['eval'][metric_name]) > 0:
            last_ndcg_val = eval_results_history['eval'][metric_name][-1]
            oof_ndcg_scores_xgb.append(last_ndcg_val)
            print(f"Fold {fold+1} Last Eval {metric_name}: {last_ndcg_val:.4f} at last iteration {len(eval_results_history['eval'][metric_name])}.")
        else:
            print(f"Error: No evaluation results found in history for fold {fold+1} for fallback.")


print(f"\nXGBoost CV Finished. Average OOF NDCG@5: {np.mean(oof_ndcg_scores_xgb):.4f} (+/- {np.std(oof_ndcg_scores_xgb):.4f})")


# 3. 特征重要性
if models_xgb:
    fig, ax = plt.subplots(figsize=(10, 15))
    xgb.plot_importance(models_xgb[0], ax=ax, max_num_features=min(40, len(feature_columns)), importance_type='gain', show_values=False)
    plt.title("XGBoost Feature Importance (Ranker - Fold 1)")
    plt.tight_layout()
    plt.show()

# 4. 进行预测和生成提交文件
print("\n--- Making Predictions on Test Set ---")
dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
test_scores_xgb = np.zeros(len(X_test))

if models_xgb:
    for model in models_xgb:
        test_scores_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration)) / n_splits
else:
    print("Warning: No CV models for XGBoost found, training a single model.")
    dtrain_full = xgb.DMatrix(X_train, label=Y_train, feature_names=feature_columns)
    dtrain_full.set_group(train_groups)
    single_model_xgb = xgb.train(xgb_ranker_params, dtrain_full, num_boost_round=500)
    test_scores_xgb = single_model_xgb.predict(dtest)


submission_df_xgb = pd.DataFrame({
    'srch_id': test_srch_ids_for_submission,
    'prop_id': test_prop_ids_for_submission,
    'score': test_scores_xgb
})
submission_df_xgb = submission_df_xgb.sort_values(['srch_id', 'score'], ascending=[True, False])
final_submission_list_xgb = [{'srch_id': int(r['srch_id']), 'prop_id': int(r['prop_id'])} for _, r in submission_df_xgb.iterrows()]
final_submission_df_xgb = pd.DataFrame(final_submission_list_xgb)

submission_filename_xgb = 'submission_xgboost_ranker_2.csv'
final_submission_df_xgb.to_csv(submission_filename_xgb, index=False)
print(f"XGBoost Submission file '{submission_filename_xgb}' created successfully.")