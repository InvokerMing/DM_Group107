import pandas as pd


def handle_missing_values(df):
    """
    Fill missing values in the Expedia dataset.
    This should be applied equally to both train and test sets.
    """
    df['visitor_hist_starrating'].fillna(-1, inplace=True)
    df['visitor_hist_adr_usd'].fillna(-1, inplace=True)

    if 'prop_review_score' in df.columns:
        df['prop_review_score'].fillna(df['prop_review_score'].median(), inplace=True)

    df['orig_destination_distance'].fillna(-1, inplace=True)
    df['prop_location_score2'].fillna(0, inplace=True)
    df['srch_query_affinity_score'].fillna(0, inplace=True)

    if 'gross_bookings_usd' in df.columns:
        df.drop(columns=['gross_bookings_usd'], inplace=True)

    comp_cols = [col for col in df.columns if col.startswith('comp')]
    df.drop(columns=comp_cols, inplace=True)

    return df


def feature_engineering(df, is_train=True):
    """
    Perform feature engineering on Expedia dataset.
    Applies both to training and test sets.
    """
    df['price_rank'] = df.groupby('srch_id')['price_usd'].rank(method='min')
    df['score_rank'] = df.groupby('srch_id')['prop_review_score'].rank(ascending=False, method='min')

    df['price_diff_vs_mean'] = df['price_usd'] - df.groupby('srch_id')['price_usd'].transform('mean')
    df['score_diff_vs_mean'] = df['prop_review_score'] - df.groupby('srch_id')['prop_review_score'].transform('mean')

    df['price_diff_vs_hist'] = df['price_usd'] - df['visitor_hist_adr_usd']
    df['score_diff_vs_hist'] = df['prop_review_score'] - df['visitor_hist_starrating']

    df['is_same_country'] = (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)

    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['search_month'] = df['date_time'].dt.month
        df['search_dayofweek'] = df['date_time'].dt.dayofweek
        df['search_hour'] = df['date_time'].dt.hour

    if is_train and 'booking_bool' in df.columns and 'click_bool' in df.columns:
        df['label'] = 5 * df['booking_bool'] + df['click_bool']

    return df


if __name__ == "__main__":
    # Load raw data
    train = pd.read_csv(r"C:\Users\Clanc\Desktop\DMT\A2\dmt-2025-2nd-assignment\training_set_VU_DM.csv", na_values=['NULL'])
    test = pd.read_csv(r"C:\Users\Clanc\Desktop\DMT\A2\dmt-2025-2nd-assignment\test_set_VU_DM.csv", na_values=['NULL'])

    # Handle missing values
    train = handle_missing_values(train)
    test = handle_missing_values(test)

    # Feature engineering
    train = feature_engineering(train, is_train=True)
    test = feature_engineering(test, is_train=False)

    # Export processed data
    train.to_csv("train_prepared.csv", index=False)
    test.to_csv("test_prepared.csv", index=False)

    print("✅ Data preparation completed and files saved:")
    print("- train_prepared.csv")
    print("- test_prepared.csv")