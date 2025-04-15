import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


TIME_COLUMN = 'time'
ID_COLUMN = 'id'
VARIABLE_COLUMN = 'variable'
VALUE_COLUMN = 'value'
DATA_FILE = 'data/dataset_mood_smartphone.csv'
CLEANED_DATA_FILE = 'data/cleaned_dataset_mood_smartphone.csv'


# --- Load Data ---
print("--- Loading Data ---")
df = pd.read_csv(DATA_FILE)
print(f"Successfully loaded data from: {DATA_FILE}, Shape: {df.shape}")

# --- Preprocessing: Handle Duplicates and Convert Value ---
df['value_numeric'] = pd.to_numeric(df[VALUE_COLUMN], errors='coerce')
print("\nConverted 'value' to numeric 'value_numeric'; conversion errors became NaN.")

# Handle potential duplicate entries for the same id, time (as string/object), and variable by aggregation
print(f"\nChecking for and handling duplicate ({ID_COLUMN}, {TIME_COLUMN}, {VARIABLE_COLUMN}) entries by averaging numeric values...")
agg_funcs = {
    'value_numeric': 'mean', # Average numeric values
    VALUE_COLUMN: 'first'    # Keep the first original value for reference
}
original_rows = df.shape[0]
# Group by the identifying columns and aggregate
df = df.groupby([ID_COLUMN, TIME_COLUMN, VARIABLE_COLUMN], as_index=False).agg(agg_funcs)
processed_rows = df.shape[0]
if processed_rows < original_rows:
    print(f"Processed {original_rows - processed_rows} duplicate entries via aggregation.")
else:
    print("No duplicate entries requiring aggregation were found.")

# Sort data primarily by ID. The order within ID will be based on the aggregation/original file order.
df.sort_values(by=[ID_COLUMN], inplace=True)
df.reset_index(drop=True, inplace=True) # Reset index after sorting

print("\n--- Task 1A: Exploratory Data Analysis (EDA) ---")

# 1. Basic Properties
print("\n--- 1A.1: Basic Properties ---")
print(f"Number of records (after aggregation): {df.shape[0]}, Number of columns: {df.shape[1]}")
df.info() # Check column names and types.
unique_ids = df[ID_COLUMN].unique()
unique_variables = df[VARIABLE_COLUMN].unique()
print(f"\nNumber of unique IDs: {len(unique_ids)}")
print(f"Number of unique Variables: {len(unique_variables)}")
try:
    time_deltas = pd.to_timedelta(df[TIME_COLUMN].astype(str))
    print(f"Time-of-day range (approx): {time_deltas.min()} to {time_deltas.max()}")
except ValueError:
    print(f"Could not interpret '{TIME_COLUMN}' column as consistent time duration/time of day for range calculation.")


# 2. Value Distributions (Grouped by Variable)
print("\n--- 1A.2: Value Distributions ---")
numeric_variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms',
                     'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                     'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
                     'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
                     'appCat.weather']
numeric_variables_present = [v for v in unique_variables if v in numeric_variables]
df_numeric_view = df[df[VARIABLE_COLUMN].isin(numeric_variables_present)]

if not df_numeric_view.empty:
    print("\nDescriptive Statistics for Numeric Variables (Grouped by Variable):")
    print(df_numeric_view.groupby(VARIABLE_COLUMN)['value_numeric'].describe())
else:
    print("\nNo specified numeric variables found in the data.")

# Plot distributions for a few key numeric variables (example)
key_numeric_vars_plot = ['mood', 'activity', 'screen']
print(f"\nPlotting distributions for example variables: {key_numeric_vars_plot}")
plt.figure(figsize=(12, 4 * ((len(key_numeric_vars_plot) // 2) + (len(key_numeric_vars_plot) % 2))))
plot_num = 1
for var in key_numeric_vars_plot:
    if var in unique_variables:
        plt.subplot(((len(key_numeric_vars_plot) // 2) + (len(key_numeric_vars_plot) % 2)), 2, plot_num)
        subset = df[df[VARIABLE_COLUMN] == var]['value_numeric'].dropna()
        if not subset.empty:
            sns.histplot(subset, kde=True)
            plt.title(f'Distribution of {var}')
        else:
            plt.title(f'Distribution of {var} (No Data)')
        plot_num += 1
plt.tight_layout()
plt.show()

# 3. Missing Values Analysis
print("\n--- 1A.3: Missing Values Analysis ---")
print("Missing values count in 'value_numeric' (after aggregation):")
print(df['value_numeric'].isnull().sum())

# Visualize missing data patterns - Cannot use time axis effectively
# Alternative: Plot presence against record index for a user
if len(unique_ids) > 0:
    example_id = unique_ids[0] # Pick one ID as an example
    print(f"\nVisualizing data presence against record index for ID: {example_id} (selected variables: {key_numeric_vars_plot})")

    # Select data for the example user and key variables
    df_example = df[(df[ID_COLUMN] == example_id) & (df[VARIABLE_COLUMN].isin(key_numeric_vars_plot))].reset_index() # Get record index
    
    # --- Debug Print 1 ---
    print(f"Debug: Shape of df_example for ID {example_id}: {df_example.shape}")
    if df_example.empty:
        print(f"Debug: df_example is empty. No data for this ID and variable combination.")
    else:
        print(f"Debug: df_example 'value_numeric' NaNs: {df_example['value_numeric'].isnull().sum()} out of {df_example.shape[0]}")

        # Pivot for plotting presence
        # Use fill_value=0 to ensure combinations not present become 0 instead of NaN in the pivot table
        df_pivot_presence = df_example.pivot_table(index='index', columns=VARIABLE_COLUMN, values='value_numeric',
                                                   aggfunc=lambda x: 1 if x.notna().any() else 0, # Check if any value in the group is not NaN
                                                   fill_value=0) # Rows/columns with no data will be filled with 0

        # --- Debug Print 2 ---
        print(f"Debug: Shape of df_pivot_presence: {df_pivot_presence.shape}")
        if not df_pivot_presence.empty:
            print("Debug: Head of df_pivot_presence:")
            print(df_pivot_presence.head())
            print(f"Debug: Sum of presence values in df_pivot_presence: {df_pivot_presence.sum().sum()}") # Check if any '1's exist

            # Check if pivot table contains only zeros
            if df_pivot_presence.sum().sum() == 0:
                print("Debug: df_pivot_presence contains only 0s (no data presence detected for these variables/index). Heatmap will be blank or uniform.")

            # --- Plotting ---
            plt.figure(figsize=(15, 3))
            sns.heatmap(df_pivot_presence.transpose(), cmap="gray_r", cbar=False, xticklabels=50) # Show fewer x-labels
            plt.title(f'Data Presence (1=Present) vs Record Index for ID {example_id}')
            plt.xlabel('Record Index (within user)')
            plt.show()
        else:
            # This case might be hit if df_example was not empty but pivot failed
            print(f"Debug: df_pivot_presence is empty after pivot operation for ID {example_id}.")
else:
    print("No unique IDs found in the data.")


# 4. Relationships & Sequence Plots (Not strictly Time Series)
print("\n--- 1A.4: Sequence Plots ---")
# Plot values against their sequence index for one example user
if len(unique_ids) > 0:
    example_id = unique_ids[0]
    print(f"\nPlotting example sequence plots for ID: {example_id}")
    df_user_subset = df[df[ID_COLUMN] == example_id].reset_index() # Get index for plotting sequence

    plt.figure(figsize=(15, 6))
    # Mood
    plt.subplot(2, 1, 1)
    mood_data = df_user_subset[df_user_subset[VARIABLE_COLUMN] == 'mood'].dropna(subset=['value_numeric'])
    if not mood_data.empty:
        plt.plot(mood_data.index, mood_data['value_numeric'], marker='.', linestyle='-', label='Mood')
        plt.title(f'Mood Sequence Plot for ID {example_id}')
        plt.ylabel('Mood')
        plt.xlabel('Record Index (within user)')
        plt.legend()
        plt.grid(True)
    # Activity
    plt.subplot(2, 1, 2)
    activity_data = df_user_subset[df_user_subset[VARIABLE_COLUMN] == 'activity'].dropna(subset=['value_numeric'])
    if not activity_data.empty:
        plt.plot(activity_data.index, activity_data['value_numeric'], marker='.', linestyle='-', label='Activity', color='orange')
        plt.title(f'Activity Sequence Plot for ID {example_id}')
        plt.ylabel('Activity')
        plt.xlabel('Record Index (within user)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n--- Task 1B: Data Cleaning ---")

# Create a copy for cleaning operations
df_clean = df.copy()

# 1. Outlier Removal [source: 47]
# Example Approach: Apply IQR method within each ID for each numeric Variable.
numeric_variables_to_clean = [v for v in unique_variables if v in numeric_variables]
print("\n--- 1B.1: Outlier Removal (IQR Method per ID/Variable) ---")

# Define IQR outlier detection function
def detect_outliers_iqr(group):
    if group['value_numeric'].isnull().all():
        return group.assign(Is_Outlier=False)
    q1 = group['value_numeric'].quantile(0.25)
    q3 = group['value_numeric'].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        is_outlier_series = pd.Series(False, index=group.index)
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        is_outlier_series = ~group['value_numeric'].between(lower_bound, upper_bound, inclusive='both') & ~group['value_numeric'].isnull()
    return group.assign(Is_Outlier=is_outlier_series)

# Apply IQR detection grouped by ID
df_clean['Is_Outlier'] = False
for var in numeric_variables_to_clean:
    var_mask = df_clean[VARIABLE_COLUMN] == var
    if var_mask.any():
        df_clean.loc[var_mask] = df_clean.loc[var_mask].groupby(ID_COLUMN, group_keys=False).apply(detect_outliers_iqr)

# Apply domain-specific rules (e.g., mood range 1-10) [source: 31]
mood_mask = df_clean[VARIABLE_COLUMN] == 'mood'
if mood_mask.any():
    domain_outliers_mood = ~df_clean.loc[mood_mask, 'value_numeric'].between(1, 10, inclusive='both') & ~df_clean.loc[mood_mask, 'value_numeric'].isnull()
    df_clean.loc[mood_mask & domain_outliers_mood, 'Is_Outlier'] = True
# ... Add similar checks for other variables with known ranges ...

outliers_indices = df_clean[df_clean['Is_Outlier']].index
print(f"Identified {len(outliers_indices)} potential outliers using IQR (per ID/Variable) and domain rules.")

# Handle outliers: Replace with NaN
print("Replacing identified outliers with NaN.")
df_clean.loc[outliers_indices, 'value_numeric'] = np.nan
if 'Is_Outlier' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Is_Outlier'])

# 2. Missing Value Imputation (Adapting for lack of proper time index) [source: 53]
print("\n--- 1B.2: Missing Value Imputation ---")
print("Warning: Due to lack of date info, time-based interpolation is not applicable.")
print("Using methods based on record order within ID/Variable groups.")

# Ensure data is sorted by ID, Variable, and original index/time-of-day if meaningful for order
# Sorting just by ID was done earlier. Let's assume the order within ID is somewhat sequential.
df_clean.sort_values(by=[ID_COLUMN, VARIABLE_COLUMN], inplace=True) # Keep variable records together within user

# --- Method 1: Linear Interpolation (based on index/order) ---
print("Applying Imputation Method 1: Linear Interpolation (order-based)")
df_imputed_linear = df_clean.copy()
# Group by ID and Variable, then apply interpolation based on existing order
df_imputed_linear['value_imputed_linear'] = df_imputed_linear.groupby([ID_COLUMN, VARIABLE_COLUMN])['value_numeric']\
                                                        .transform(lambda series: series.interpolate(method='linear', limit_direction='both', limit_area=None))

# --- Method 2: Forward Fill + Backward Fill ---
print("Applying Imputation Method 2: Forward Fill (LOCF) + Backward Fill (BOCF)")
df_imputed_ffill = df_clean.copy()
df_imputed_ffill['value_imputed_ffill'] = df_imputed_ffill.groupby([ID_COLUMN, VARIABLE_COLUMN])['value_numeric']\
                                                      .transform(lambda series: series.fillna(method='ffill').fillna(method='bfill'))

# --- Comparison & Justification ---
print("\nComparing Imputation Methods (Example: Checking remaining NaNs):")
nans_original = df_clean['value_numeric'].isnull().sum()
nans_linear = df_imputed_linear['value_imputed_linear'].isnull().sum()
nans_ffill = df_imputed_ffill['value_imputed_ffill'].isnull().sum()
print(f"NaNs count after outlier removal: {nans_original}")
print(f"NaNs count after Linear Interpolation (order-based): {nans_linear}")
print(f"NaNs count after Forward/Backward Fill: {nans_ffill}")

# --- STUDENT ACTION REQUIRED ---
# TODO: Add more sophisticated comparison between methods here.
#       (e.g., visualize imputed vs. original series for samples, compare distributions)
# TODO: Based on your analysis, data characteristics (e.g., gap lengths [source: 54]),
#       task goals (considering the limitations), and literature [source: 53], choose ONE final imputation method.
# TODO: Clearly justify your choice in the report. Acknowledge the limitations imposed by the lack of proper timestamps.
#       How prolonged missing periods [source: 54] are handled needs careful consideration (e.g., ffill might be inappropriate).

# --- Apply Final Chosen Imputation Method (Example: Choosing Ffill/Bfill) ---
chosen_imputation_method = 'ffill' # <<< Modify this based on your analysis
print(f"\nApplying final chosen imputation method: '{chosen_imputation_method}' (Example Choice).")

df_final_cleaned = df_clean.copy()
if chosen_imputation_method == 'linear':
    df_final_cleaned['value_numeric'] = df_imputed_linear['value_imputed_linear']
elif chosen_imputation_method == 'ffill':
    df_final_cleaned['value_numeric'] = df_imputed_ffill['value_imputed_ffill']
else:
    print(f"Warning: Chosen imputation method '{chosen_imputation_method}' is not implemented in this example!")

# Check for any remaining NaNs after the chosen imputation
remaining_nans_final = df_final_cleaned['value_numeric'].isnull().sum()
if remaining_nans_final > 0:
    print(f"Warning: {remaining_nans_final} NaNs still remain after final imputation.")
    # TODO: Decide how to handle these remaining NaNs (e.g., drop rows, impute with group/global median/mean)
    # Example: Fill remaining NaNs with the median of that specific variable across all users
    # variable_medians = df_final_cleaned.groupby(VARIABLE_COLUMN)['value_numeric'].transform('median')
    # df_final_cleaned['value_numeric'].fillna(variable_medians, inplace=True)
    # print("Filled remaining NaNs using variable-specific medians.")



print("\n--- Task 1B Data Cleaning Finished ---")

# --- Save Cleaned Data & Final Quality Check ---
print("\n--- Saving Cleaned Data & Final Check ---")
try:
    df_final_cleaned.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"Cleaned data saved successfully to: {CLEANED_DATA_FILE}")
except Exception as e:
    print(f"Error saving cleaned data to {CLEANED_DATA_FILE}: {e}")

# Final check on missing values in the cleaned dataset
final_nan_check = df_final_cleaned['value_numeric'].isnull().sum()
print(f"\nFinal check: Missing values in 'value_numeric' of cleaned data: {final_nan_check}")
if final_nan_check == 0:
    print("Cleaned data has no missing numeric values.")
else:
    print("Warning: Cleaned data still contains missing numeric values. Review imputation strategy or handling of remaining NaNs.")