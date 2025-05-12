import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # 用于处理可能的 inf 值

# --- 配置 ---
plt.style.use('ggplot')

# --- 加载数据 ---
file_path = 'data/training_set_VU_DM.csv'
print(f"Loading data from: {file_path}")
# na_values=['NULL'] 确保文件中的 'NULL' 字符串被识别为 NaN
df = pd.read_csv(file_path, na_values=['NULL'])
print("Data loaded successfully.")

# --- 1. 初步检查 ---
print("\n--- Basic Info ---")
df.info()

print("\n--- Descriptive Statistics (Numerical Columns) ---")
# 选择数值类型的列进行描述性统计
numerical_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd',
                  'prop_starrating', 'prop_review_score']
print(df[numerical_cols].describe())

print("\n--- Example Data Rows ---")
print(df.head())

# --- 2. 缺失值分析 ---
print("\n--- Missing Value Analysis ---")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': missing_percent})
missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)
print(missing_data)

# 可视化缺失值百分比
if not missing_data.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data['Missing Percentage'])
    plt.title('Percentage of Missing Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found in the dataset.")


# --- 3. 分布分析 (数值型特征) ---
print("\n--- Distribution Analysis (Numerical) ---")

numerical_cols_to_plot = ['prop_starrating', 'prop_review_score'] # 这些列通常都有值
# visitor_hist_* 列缺失较多，绘制前可以先填充或移除NaN，这里先只绘制非缺失部分
visitor_hist_cols_to_plot = ['visitor_hist_starrating', 'visitor_hist_adr_usd']

# 绘制直方图和箱线图
for col in numerical_cols_to_plot + visitor_hist_cols_to_plot:
    if col in df.columns:
        plt.figure(figsize=(12, 5))

        # 直方图
        plt.subplot(1, 2, 1)
        # 使用 dropna() 避免 NaN 影响绘图，对于 hist_adr 可能有极大值，先处理inf和极大值
        data_to_plot = df[col].dropna()
        if col == 'visitor_hist_adr_usd':
             # 简单处理：替换 inf 并限制一个较合理的上限（例如基于 99% 分位数）
             data_to_plot = data_to_plot.replace([np.inf, -np.inf], np.nan).dropna()
             upper_limit = data_to_plot.quantile(0.99) if not data_to_plot.empty else 1000 # 设置一个上限
             data_to_plot = data_to_plot[data_to_plot <= upper_limit]

        sns.histplot(data_to_plot, kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # 箱线图
        plt.subplot(1, 2, 2)
        sns.boxplot(y=data_to_plot) # 使用处理过的数据
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)

        plt.tight_layout()
        plt.show()


# --- 4. 分布分析 (类别型/ID型特征) ---
print("\n--- Distribution Analysis (Categorical/ID) ---")

# 选择基数相对较低的类别型特征进行可视化
categorical_cols_low_cardinality = ['site_id', 'prop_starrating', 'prop_country_id', 'visitor_location_country_id']

for col in categorical_cols_low_cardinality:
    if col in df.columns:
        plt.figure(figsize=(12, 6))
        # 获取 Top N 类别进行展示，避免类别过多导致图形混乱
        top_n = 20
        # 计算频率并取 Top N
        value_counts = df[col].value_counts()
        top_categories = value_counts.nlargest(top_n).index

        # 仅绘制 Top N 类别的条形图
        sns.countplot(data=df[df[col].isin(top_categories)], x=col, order=top_categories)
        plt.title(f'Frequency of Top {top_n} {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# 对于高基数 ID (如 prop_id, srch_id)，统计唯一值数量
print(f"Number of unique srch_id: {df['srch_id'].nunique()}")
print(f"Number of unique prop_id: {df['prop_id'].nunique()}")


# --- 5. 时间特征分析 (`date_time`) ---
print("\n--- Time Feature Analysis ---")
if 'date_time' in df.columns:
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['dt_year'] = df['date_time'].dt.year
    df['dt_month'] = df['date_time'].dt.month
    df['dt_dayofweek'] = df['date_time'].dt.dayofweek # Monday=0, Sunday=6
    df['dt_hour'] = df['date_time'].dt.hour

    print("Time components extracted: year, month, dayofweek, hour.")

    # 可视化搜索量随时间的变化
    plt.figure(figsize=(12, 5))
    df.groupby('dt_month')['srch_id'].count().plot(kind='bar')
    plt.title('Number of Searches per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Searches')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    df.groupby('dt_dayofweek')['srch_id'].count().plot(kind='bar')
    plt.title('Number of Searches per Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Number of Searches')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    df.groupby('dt_hour')['srch_id'].count().plot(kind='bar')
    plt.title('Number of Searches per Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Searches')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

else:
    print("Column 'date_time' not found, skipping time analysis.")

print("\n--- EDA Finished ---")