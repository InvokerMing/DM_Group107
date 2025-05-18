import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import re

print("--- Loading Processed Data for XGBoost Ranker ---")
try:
    train_df = pd.read_csv('data/processed_training_set.csv')
    test_df = pd.read_csv('data/processed_test_set.csv')
except FileNotFoundError:
    print("Error: Processed data files not found. Please run data_process.py first.")
    exit()

# 1. 定义特征列和目标列
Y_train = train_df['relevance_score']

cols_to_exclude_from_features = [
    'srch_id', 'prop_id', 'relevance_score', 'click_bool', 'booking_bool',
    'gross_bookings_usd', 'position', 'date_time', 'date_time_dt'
]

feature_columns = [col for col in train_df.columns if col not in cols_to_exclude_from_features]
common_features = list(set(feature_columns) & set(test_df.columns))
feature_columns = common_features # 使用共同特征

X_train = train_df[feature_columns]
train_srch_ids_for_grouping = train_df['srch_id']
train_groups = train_df.groupby('srch_id', sort=False).size().to_numpy()

X_test = test_df[feature_columns]
test_srch_ids_for_submission = test_df['srch_id']
test_prop_ids_for_submission = test_df['prop_id']

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")

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
    'max_depth': 5,              # 比LGBM稍浅一些开始
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


n_splits = 3
group_kfold = GroupKFold(n_splits=n_splits)
models_xgb = []
oof_ndcg_scores_xgb = []

print(f"Starting {n_splits}-Fold Cross-Validation for XGBoost...")
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, Y_train, groups=train_srch_ids_for_grouping.values)):
    print(f"--- XGBoost Fold {fold+1}/{n_splits} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

    # XGBoost 需要 DMatrix，并且 group 信息在 DMatrix 中设置
    dtrain = xgb.DMatrix(X_train_fold, label=Y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=Y_val_fold)

    # 设置 group 信息
    train_fold_srch_ids = train_srch_ids_for_grouping.iloc[train_idx]
    val_fold_srch_ids = train_srch_ids_for_grouping.iloc[val_idx]
    train_fold_groups_xgb = train_fold_srch_ids.groupby(train_fold_srch_ids, sort=False).size().to_numpy()
    val_fold_groups_xgb = val_fold_srch_ids.groupby(val_fold_srch_ids, sort=False).size().to_numpy()
    dtrain.set_group(train_fold_groups_xgb)
    dval.set_group(val_fold_groups_xgb)

    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    model = xgb.train(
        xgb_ranker_params,
        dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=50
    )
    models_xgb.append(model)
    # XGBoost 的 eval history 比较复杂，这里简化
    # 通常看最后一次 eval 的 ndcg@5
    # best_ndcg = model.best_score # 这是 best_iteration 时的 eval metric
    # print(f"Fold {fold+1} Best NDCG@5 (approx from best_score): {best_ndcg:.4f} at iteration {model.best_iteration}")
    # oof_ndcg_scores_xgb.append(best_ndcg)
    # 更准确地获取特定指标:
    try:
        results = model.evals_result()
    except:
        print(f"Error: No eval results for fold {fold+1}")
        continue
    if 'eval' in results and 'ndcg@5' in results['eval']:
       best_ndcg_val = results['eval']['ndcg@5'][model.best_iteration -1] # -1因为迭代从1开始，列表索引从0
       oof_ndcg_scores_xgb.append(best_ndcg_val)
       print(f"Fold {fold+1} Best Eval NDCG@5: {best_ndcg_val:.4f} at iteration {model.best_iteration}")


print(f"\nXGBoost CV Finished. Average OOF NDCG@5: {np.mean(oof_ndcg_scores_xgb):.4f} (+/- {np.std(oof_ndcg_scores_xgb):.4f})")


# 3. 特征重要性
if models_xgb:
    fig, ax = plt.subplots(figsize=(10, 15))
    xgb.plot_importance(models_xgb[0], ax=ax, max_num_features=40, importance_type='gain', show_values=False)
    plt.title("XGBoost Feature Importance (Ranker - Fold 1)")
    plt.tight_layout()
    plt.show()

# 4. 进行预测和生成提交文件
print("\n--- Making Predictions on Test Set ---")
dtest = xgb.DMatrix(X_test)
test_scores_xgb = np.zeros(len(X_test))

if models_xgb:
    for model in models_xgb:
        test_scores_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration)) / n_splits
else:
    print("Warning: No CV models for XGBoost found, training a single model.")
    dtrain_full = xgb.DMatrix(X_train, label=Y_train)
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

submission_filename_xgb = 'submission_xgboost_ranker.csv'
final_submission_df_xgb.to_csv(submission_filename_xgb, index=False)
print(f"XGBoost Submission file '{submission_filename_xgb}' created successfully.")