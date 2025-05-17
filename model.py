import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ndcg_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ----------------------------
# Load and split data
# ----------------------------
train_df = pd.read_csv("train_prepared.csv")

def get_features(df):
    drop_cols = ['srch_id', 'prop_id', 'date_time', 'click_bool', 'booking_bool', 'label']
    return [col for col in df.columns if col not in drop_cols]

features = get_features(train_df)
X = train_df[features]
y = train_df['label']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_ids = train_df.loc[X_train.index, 'srch_id']
val_ids = train_df.loc[X_val.index, 'srch_id']
group_train = train_ids.groupby(train_ids).size().to_list()
group_val = val_ids.groupby(val_ids).size().to_list()

# ----------------------------
# KNN Step: sample subset of validation set
# ----------------------------
print("\n🔄 Running Item-based KNN (Top-K)...")
val_subset = train_df.loc[X_val.index].copy()
val_subset = val_subset.groupby('srch_id').head(10).reset_index(drop=True)  # ✅ 抽样每个搜索前10条

prop_cat = val_subset['prop_id'].astype('category')
srch_cat = val_subset['srch_id'].astype('category')

ratings = val_subset['booking_bool'].values
row = prop_cat.cat.codes
col = srch_cat.cat.codes

item_user_sparse = csr_matrix((ratings, (row, col)))
item_ids = prop_cat.cat.categories

knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=21, n_jobs=-1)
knn_model.fit(item_user_sparse)

distances, indices = knn_model.kneighbors(item_user_sparse, return_distance=True)

item_sim_dict = {}
for idx, (dists, nbrs) in enumerate(zip(distances, indices)):
    src_item = item_ids[idx]
    sim_scores = 1 - dists[1:]  # skip self-match
    sim_items = item_ids[nbrs[1:]]
    item_sim_dict[src_item] = dict(zip(sim_items, sim_scores))

val_knn_records = []
for srch_id, group in val_subset.groupby('srch_id'):
    prop_ids = group['prop_id'].tolist()
    for pid in prop_ids:
        score = sum(item_sim_dict.get(pid, {}).get(other_pid, 0) for other_pid in prop_ids if other_pid != pid)
        label = group[group['prop_id'] == pid]['label'].values[0]
        val_knn_records.append({'srch_id': srch_id, 'prop_id': pid, 'label': label, 'score': score})
val_knn = pd.DataFrame(val_knn_records)


# ----------------------------
# LightGBM Model (tunable)
# ----------------------------
print("\n🔄 Training LightGBM...")
lgb_train = lgb.Dataset(X_train, label=y_train, group=group_train)
lgb_val = lgb.Dataset(X_val, label=y_val, group=group_val)

params_lgb = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'learning_rate': 0.1,   # 👈 可调参
    'num_leaves': 31,       # 👈 可调参
    'verbose': -1
}

# 若想进行参数搜索，可以使用 sklearn.model_selection.GridSearchCV
# TODO: 添加 GridSearchCV 逻辑（如需调参）

lgb_model = lgb.train(params_lgb, lgb_train, num_boost_round=100)
val_lgb = train_df.loc[X_val.index, ['srch_id', 'prop_id', 'label']].copy()
val_lgb['score'] = lgb_model.predict(X_val)

# ----------------------------
# XGBoost Model (tunable)
# ----------------------------
print("\n🔄 Training XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)
dval = xgb.DMatrix(X_val)

params_xgb = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'learning_rate': 0.1,       # 👈 可调参
    'gamma': 1.0,               # 👈 可调参
    'min_child_weight': 0.1,    # 👈 可调参
    'max_depth': 6              # 👈 可调参
}

# 若想使用 GridSearchCV + xgb.cv 实现超参搜索，可参考官方文档

xgb_model = xgb.train(params=params_xgb, dtrain=dtrain, num_boost_round=100)
val_xgb = train_df.loc[X_val.index, ['srch_id', 'prop_id', 'label']].copy()
val_xgb['score'] = xgb_model.predict(dval)

# ----------------------------
# Evaluate NDCG@5
# ----------------------------
def compute_grouped_ndcg(df, score_col, k=5):
    ndcgs = []
    for srch_id, group in df.groupby("srch_id"):
        if len(group) <= 1:
            continue
        if group['label'].sum() == 0:
            continue
        ndcg = ndcg_score([group['label']], [group[score_col]], k=k)
        ndcgs.append(ndcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0

ndcg_lgb = compute_grouped_ndcg(val_lgb, "score")
ndcg_xgb = compute_grouped_ndcg(val_xgb, "score")
ndcg_knn = compute_grouped_ndcg(val_knn, "score")

# ➕ 计算训练集上的 NDCG@5
train_lgb = train_df.loc[X_train.index, ['srch_id', 'prop_id', 'label']].copy()
train_lgb['score'] = lgb_model.predict(X_train)
ndcg_train_lgb = compute_grouped_ndcg(train_lgb, 'score')

train_xgb = train_df.loc[X_train.index, ['srch_id', 'prop_id', 'label']].copy()
train_xgb['score'] = xgb_model.predict(xgb.DMatrix(X_train))
ndcg_train_xgb = compute_grouped_ndcg(train_xgb, 'score')

# ----------------------------
# Plot results
# ----------------------------
print("\n📊 NDCG@5 Comparison:")
print(f"LightGBM - Train: {ndcg_train_lgb:.4f}, Validation: {ndcg_lgb:.4f}")
print(f"XGBoost  - Train: {ndcg_train_xgb:.4f}, Validation: {ndcg_xgb:.4f}")
print(f"ItemKNN : {ndcg_knn:.4f}")

models = ['LightGBM (Val)', 'XGBoost (Val)', 'ItemKNN']
scores = [ndcg_lgb, ndcg_xgb, ndcg_knn]
plt.bar(models, scores, color=['green', 'orange', 'blue'])
plt.title("NDCG@5 Comparison (Validation Set)")
plt.ylabel("NDCG@5 Score")
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("ndcg_validation_comparison.png")
plt.show()
