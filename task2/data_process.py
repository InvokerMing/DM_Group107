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
                                             0 表示没有竞争或信息不可用 (用于 rate, inv)。
                                             对于 rate_percent_diff，0 表示价格相同或无信息。
        hist_missing_fill_value (int/float): 用户历史信息缺失时的填充值。
                                             对于 starrating 和 adr_usd，0 或中位数/均值是可选项。
                                             EDA 显示这些特征缺失率极高。

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
    print("Handling missing values with improved strategies...")

    # 用户历史信息 (高缺失率) - 创建缺失指示，然后填充
    # visitor_hist_starrating: EDA显示大部分是NaN。填充值可以是0或者全局平均（如果计算的话）
    # visitor_hist_adr_usd: EDA显示大部分是NaN，且有极端值。先处理极端值，再填充。
    df['visitor_hist_starrating_missing'] = df['visitor_hist_starrating'].isnull().astype(int)
    df['visitor_hist_starrating'].fillna(hist_missing_fill_value, inplace=True) # 0作为通用填充

    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isnull().astype(int)
    # EDA提示visitor_hist_adr_usd有潜在的极大值，填充前先处理
    # 如果hist_missing_fill_value不是0，需要确保它是一个合理的值
    # 这里我们用0填充，因为EDA显示其分布，如果填充均值或中位数会引入偏差
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].replace([np.inf, -np.inf], np.nan)
    # 可以考虑对非缺失值进行clip，但由于缺失率太高，直接填充
    df['visitor_hist_adr_usd'].fillna(hist_missing_fill_value, inplace=True)


    # 酒店评分和位置评分
    # prop_review_score: 少量缺失，用中位数填充是合理的
    median_prop_review_score = df['prop_review_score'].median() # 先计算中位数
    df['prop_review_score_missing'] = df['prop_review_score'].isnull().astype(int)
    df['prop_review_score'].fillna(median_prop_review_score, inplace=True)

    # prop_location_score2: 较多缺失，用中位数填充，并创建缺失指示
    median_prop_loc_score2 = df['prop_location_score2'].median() # 先计算中位数
    df['prop_location_score2_missing'] = df['prop_location_score2'].isnull().astype(int)
    df['prop_location_score2'].fillna(median_prop_loc_score2, inplace=True)

    # 搜索亲和度和原始目的地距离
    # srch_query_affinity_score: EDA显示高缺失，用中位数填充，并创建缺失指示
    median_affinity_score = df['srch_query_affinity_score'].median()
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isnull().astype(int)
    df['srch_query_affinity_score'].fillna(median_affinity_score, inplace=True)

    # orig_destination_distance: 较多缺失，用中位数填充，并创建缺失指示
    # EDA图未直接显示此列，但通常距离特征填充中位数是稳健的
    median_orig_dest_dist = df['orig_destination_distance'].median()
    df['orig_destination_distance_missing'] = df['orig_destination_distance'].isnull().astype(int)
    df['orig_destination_distance'].fillna(median_orig_dest_dist, inplace=True)

    # 竞争对手信息 (缺失率极高) - 创建缺失指示，然后用指定值填充
    # comp_missing_fill_value (默认0)
    # rate: 0可以表示价格相同或无竞争对手。 -1表示更贵, 1表示更便宜。
    # inv: 0可以表示有库存或无竞争对手。 1表示无库存。
    # rate_percent_diff: 0可以表示价格差异为0或无信息。
    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'

        if rate_col in df.columns:
            df[f'{rate_col}_missing'] = df[rate_col].isnull().astype(int)
            df[rate_col].fillna(comp_missing_fill_value, inplace=True)
        if inv_col in df.columns:
            df[f'{inv_col}_missing'] = df[inv_col].isnull().astype(int)
            df[inv_col].fillna(comp_missing_fill_value, inplace=True) # 0表示有库存或无竞争
        if diff_col in df.columns:
            df[f'{diff_col}_missing'] = df[diff_col].isnull().astype(int)
            df[diff_col].fillna(0, inplace=True) # 0 表示无价格差异百分比或无信息

    # gross_bookings_usd: 训练集特有，高缺失。创建指示，然后填充0。
    if 'gross_bookings_usd' in df.columns and is_train:
        df['gross_bookings_usd_missing'] = df['gross_bookings_usd'].isnull().astype(int)
        df['gross_bookings_usd'].fillna(0, inplace=True)


    # 3. 特征工程
    print("Performing enhanced feature engineering...")

    # 价格相关
    # EDA显示 price_usd 分布可能也需要注意异常值，这里先进行简单clip（基于99.9%分位数）
    # 仅在训练集上计算分位数，应用到测试集 (或全局计算)
    # 为简化，这里不对price_usd做clip，但实际项目中会考虑
    df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay'].replace(0, 1) # 防止除以0

    # 处理 prop_log_historical_price
    df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isnull().astype(int)
    df['prop_log_historical_price'].fillna(0, inplace=True) # 缺失或0表示无历史价格记录
    df['exp_historical_price'] = np.expm1(df['prop_log_historical_price'])
    df['price_diff_from_hist_abs'] = (df['price_usd'] - df['exp_historical_price']).abs()
    # 确保 hist_price 不为0，加小量修正
    df['price_ratio_hist'] = (df['price_usd'] + 1e-6) / (df['exp_historical_price'] + 1e-6)

    # 位置和评分相关
    # 使用已经填充过的 prop_location_score2 (之前用的是 .fillna(0) )
    df['location_score_combined'] = df['prop_location_score1'] + df['prop_location_score2']
    # 使用已经填充过的 visitor_hist_starrating
    df['starrating_diff_from_hist'] = df['prop_starrating'] - df['visitor_hist_starrating']
    # 确保 visitor_hist_starrating 不是填充的0，否则差异无意义，或者只在非缺失时计算
    # 考虑到我们创建了missing indicator, 模型可以学习这种差异
    df.loc[df['visitor_hist_starrating_missing'] == 1, 'starrating_diff_from_hist'] = 0 # 如果历史评分缺失，则差异设为0

    # 搜索上下文
    df['total_guests'] = df['srch_adults_count'] + df['srch_children_count']

    # 交互特征 (使用填充后的特征)
    # EDA 未直接显示这些交互的重要性，但基于常见实践保留
    df['price_x_prop_location_score2'] = df['price_usd'] * df['prop_location_score2'] # 使用已填充的score2
    df['prop_starrating_x_review_score'] = df['prop_starrating'] * df['prop_review_score'] # 使用已填充的review_score

    # 竞争对手聚合特征 (优化逻辑)
    df['num_comp_cheaper'] = 0      # compX_rate == 1
    df['num_comp_more_expensive'] = 0 # compX_rate == -1
    df['num_comp_same_price'] = 0     # compX_rate == 0 (且非缺失)
    df['num_comp_inv_unavailable'] = 0 # compX_inv == 1 (无库存)
    df['num_comp_available_info'] = 0 # 有费率信息的竞争对手数量
    df['sum_comp_rate_percent_diff_abs_valid'] = 0 # 对有效的diff取绝对值求和

    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'
        rate_missing_col = f'{rate_col}_missing' # 我们创建的缺失指示列

        if rate_col in df.columns:
            # 只有当原始竞争对手费率信息存在时，才计数
            has_rate_info = (df[rate_missing_col] == 0)
            df['num_comp_available_info'] += has_rate_info.astype(int)

            df['num_comp_cheaper'] += ((df[rate_col] == 1) & has_rate_info).astype(int)
            df['num_comp_more_expensive'] += ((df[rate_col] == -1) & has_rate_info).astype(int)
            # comp_missing_fill_value (默认0) 表示价格相同或无信息
            # 我们只在有明确费率信息 (非缺失) 且rate为0时，认为是价格相同
            df['num_comp_same_price'] += ((df[rate_col] == 0) & has_rate_info).astype(int)

        if inv_col in df.columns:
            # 只有当原始竞争对手库存信息存在时 (非缺失)，才计数其不可用状态
            has_inv_info = (df[f'{inv_col}_missing'] == 0)
            df['num_comp_inv_unavailable'] += ((df[inv_col] == 1) & has_inv_info).astype(int)

        if diff_col in df.columns:
            # 只有当原始竞争对手费率信息存在时，百分比差异才有意义
            # diff列本身也可能有原始缺失，我们用diff_missing_col检查
            has_diff_info = (df[f'{diff_col}_missing'] == 0)
            is_valid_competitor = has_rate_info # 必须要有竞争对手的rate信息
            
            # 我们只累加那些 原始diff存在 且 对应rate信息也存在的 diff
            valid_diff_values = df[diff_col].where(is_valid_competitor & has_diff_info, 0)
            df['sum_comp_rate_percent_diff_abs_valid'] += valid_diff_values.abs()


    # 平均绝对价格差异百分比（仅基于有明确竞争对手报价的情况）
    df['avg_comp_rate_percent_diff_abs'] = (df['sum_comp_rate_percent_diff_abs_valid'] / \
                                           df['num_comp_available_info'].replace(0, 1)) # 防止除以0

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
    key_engineered_cols = ['price_per_night', 'price_x_prop_location_score2', 'avg_comp_rate_percent_diff_abs', 'location_score_combined']
    for col in key_engineered_cols:
        if col in df.columns:
            print(f"NaNs in {col}: {df[col].isnull().sum()}")
            # 额外检查 inf
            if df[col].isin([np.inf, -np.inf]).any():
                print(f"Inf values found in {col}")

    print(f"--- Data Processing Finished. Shape: {df.shape} ---")
    return df

if __name__ == '__main__':
    # --- 使用示例 ---
    print("Running data processing example...")

    # 加载数据 (请确保文件路径正确)
    try:
        train_df_raw = pd.read_csv('data/training_set_VU_DM.csv', na_values=['NULL'])
        test_df_raw = pd.read_csv('data/test_set_VU_DM.csv', na_values=['NULL']) # 假设测试集文件名
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

    # 保存处理后的数据
    processed_train_df.to_csv('data/processed_training_set_2.csv', index=False)
    processed_test_df.to_csv('data/processed_test_set_2.csv', index=False)
    print("\nProcessed datasets saved (optional).")
