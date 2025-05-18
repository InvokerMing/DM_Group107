# data_process.py

import pandas as pd
import numpy as np

def create_relevance_score_mapped(row):
    """创建映射后的相关性标签 (0: 其他, 1: 点击, 2: 预订)"""
    if row['booking_bool'] == 1:
        return 2
    elif row['click_bool'] == 1:
        return 1
    else:
        return 0

def process_data(df_original, is_train=True, comp_missing_fill_value=0, hist_missing_fill_value=0):
    """
    对输入的数据集进行预处理和特征工程。

    Args:
        df_original (pd.DataFrame): 原始输入 DataFrame。
        is_train (bool): 指示是否为训练集。
        comp_missing_fill_value (int/float): 竞争对手信息缺失时的填充值。
        hist_missing_fill_value (int/float): 用户历史信息缺失时的填充值。

    Returns:
        pd.DataFrame: 处理后的 DataFrame。
    """
    print(f"--- Starting Data Processing {'for Training Set' if is_train else 'for Test Set'} ---")
    df = df_original.copy()

    # 1. 时间特征处理
    try:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['dt_year'] = df['date_time'].dt.year
        df['dt_month'] = df['date_time'].dt.month
        df['dt_dayofweek'] = df['date_time'].dt.dayofweek
        df['dt_hour'] = df['date_time'].dt.hour
        # df = df.drop('date_time', axis=1) # 暂时保留原始列，如果需要
    except KeyError:
        print("Column 'date_time' not found.")
    except Exception as e:
        print(f"Error processing date_time: {e}")

    # 2. 缺失值处理
    print("Handling missing values...")
    # 用户历史信息
    df['visitor_hist_starrating'].fillna(hist_missing_fill_value, inplace=True)
    df['visitor_hist_adr_usd'].fillna(hist_missing_fill_value, inplace=True)

    # 酒店评分和位置评分2
    df['prop_review_score'].fillna(df['prop_review_score'].median(), inplace=True) # 少量缺失，用中位数
    df['prop_location_score2'].fillna(df['prop_location_score2'].median(), inplace=True) # 较多缺失，用中位数

    # 搜索亲和度和原始目的地距离
    df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].median(), inplace=True)
    df['orig_destination_distance'].fillna(df['orig_destination_distance'].median(), inplace=True)

    # 竞争对手信息 (缺失率很高)
    comp_cols = []
    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'
        comp_cols.extend([rate_col, inv_col, diff_col])

        # 对于 rate 和 inv，缺失可能表示无竞争或信息不可用，用指定值填充
        if rate_col in df.columns:
            df[rate_col].fillna(comp_missing_fill_value, inplace=True)
        if inv_col in df.columns:
            df[inv_col].fillna(comp_missing_fill_value, inplace=True)
        # 对于 diff，缺失用0填充可能表示无差异或无信息
        if diff_col in df.columns:
            df[diff_col].fillna(0, inplace=True) # 0 表示无价格差异百分比

    # gross_bookings_usd: 训练集特有，预测时不应使用，如果作为特征，需在训练前移除
    if 'gross_bookings_usd' in df.columns and is_train:
        # 可以考虑用0填充，但更好的做法是不作为特征直接输入模型
        df['gross_bookings_usd'].fillna(0, inplace=True)


    # 3. 特征工程
    print("Performing feature engineering...")

    # 价格相关
    df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay'].replace(0, 1) # 防止除以0
    # 处理 prop_log_historical_price 对数转换和缺失
    df['prop_log_historical_price'].fillna(0, inplace=True) # 缺失或0表示无历史价格记录
    df['exp_historical_price'] = np.expm1(df['prop_log_historical_price']) # 反对数，np.expm1(0)=0
    df['price_diff_from_hist_abs'] = (df['price_usd'] - df['exp_historical_price']).abs()
    df['price_ratio_hist'] = (df['price_usd'] + 1e-6) / (df['exp_historical_price'] + 1e-6) # 加小量防止除0

    # 位置和评分相关
    df['location_score_combined'] = df['prop_location_score1'] + df['prop_location_score2'].fillna(0) # 填充score2的缺失
    df['starrating_diff_from_hist'] = df['prop_starrating'] - df['visitor_hist_starrating']

    # 搜索上下文
    df['total_guests'] = df['srch_adults_count'] + df['srch_children_count']

    # 交互特征 (基于重要性图)
    df['price_x_prop_location_score2'] = df['price_usd'] * df['prop_location_score2'].fillna(df['prop_location_score2'].median())
    df['prop_starrating_x_review_score'] = df['prop_starrating'] * df['prop_review_score'].fillna(df['prop_review_score'].median())

    # 竞争对手聚合特征
    df['num_comp_cheaper'] = 0
    df['num_comp_more_expensive'] = 0
    df['num_comp_unavailable'] = 0
    df['sum_comp_rate_percent_diff_valid'] = 0
    df['count_comp_rate_percent_diff_valid'] = 0

    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'

        if rate_col in df.columns:
            df['num_comp_cheaper'] += (df[rate_col] == 1).astype(int)
            df['num_comp_more_expensive'] += (df[rate_col] == -1).astype(int)
        if inv_col in df.columns:
            df['num_comp_unavailable'] += (df[inv_col] == 1).astype(int)
        if diff_col in df.columns:
            # 仅当 compX_rate 不是0 (即有竞争对手报价) 时，diff 才更有意义
            # 这里简化：假设填充后的0 diff 是有效的（表示价格相同或无信息）
            is_valid_diff = df[rate_col] != comp_missing_fill_value # 假设 comp_missing_fill_value 表示无数据
            df['sum_comp_rate_percent_diff_valid'] += df[diff_col] * is_valid_diff.astype(int)
            df['count_comp_rate_percent_diff_valid'] += is_valid_diff.astype(int)

    df['avg_comp_rate_percent_diff'] = (df['sum_comp_rate_percent_diff_valid'] / \
                                        df['count_comp_rate_percent_diff_valid'].replace(0, 1))


    # 4. 布尔值转换 (假设原始数据中部分bool列已是0/1 int)
    # srch_saturday_night_bool, random_bool, promotion_flag, prop_brand_bool
    # 如果不是，需要转换: df[col] = df[col].astype(int)
    # 在示例数据中，它们已经是int64

    # 5. 创建相关性标签 (如果需要，且是训练集)
    if is_train:
        df['relevance_score'] = df.apply(create_relevance_score_mapped, axis=1)

    # 6. 移除不必要的列 (在模型训练脚本中进行，这里仅作数据转换)
    # 例如，原始的 compX_* 列，如果已经创建了聚合特征，可能可以移除
    # date_time 如果不再需要，也可以移除

    print("Final check for NaN values in a few key engineered features:")
    key_engineered_cols = ['price_per_night', 'price_x_prop_location_score2', 'avg_comp_rate_percent_diff']
    for col in key_engineered_cols:
        if col in df.columns:
            print(f"NaNs in {col}: {df[col].isnull().sum()}")

    print(f"--- Data Processing Finished. Shape: {df.shape} ---")
    return df

if __name__ == '__main__':
    # --- 使用示例 ---
    print("Running data processing example...")

    # 加载数据 (请确保文件路径正确)
    try:
        train_df_raw = pd.read_csv('data/training_set.csv', na_values=['NULL'])
        test_df_raw = pd.read_csv('data/test_set.csv', na_values=['NULL']) # 假设测试集文件名
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'data/training_set.csv' and 'data/test_set.csv' exist.")
        exit()

    # 处理训练集
    processed_train_df = process_data(train_df_raw, is_train=True, comp_missing_fill_value=0, hist_missing_fill_value=0)
    print("\nProcessed Training Data Head:")
    print(processed_train_df.head())
    print(f"\nUnique relevance scores in training: {processed_train_df['relevance_score'].unique()}")

    # 处理测试集
    processed_test_df = process_data(test_df_raw, is_train=False, comp_missing_fill_value=0, hist_missing_fill_value=0)
    print("\nProcessed Test Data Head:")
    print(processed_test_df.head())

    # 保存处理后的数据 (可选)
    processed_train_df.to_csv('data/processed_training_set.csv', index=False)
    processed_test_df.to_csv('data/processed_test_set.csv', index=False)
    print("\nProcessed datasets saved (optional).")
