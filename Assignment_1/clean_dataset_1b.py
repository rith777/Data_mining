import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

import os

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    elif 18 <= hour <= 24:
        return 3
    else:
        return 4
    
def ensure_correct_data_types(df):
    if 'month' in df.columns:
        df['month'] = df['month'].astype(int)
    if 'day' in df.columns:
        df['day'] = df['day'].astype(int)
    if 'hour' in df.columns:
        df['hour'] = df['hour'].astype(int)
    if 'minute' in df.columns:
        df['minute'] = df['minute'].astype(int)
    if 'second' in df.columns:
        df['second'] = df['second'].astype(int)
    if 'day_of_week' in df.columns:
        df['day_of_week'] = df['day_of_week'].astype(int)
    if 'time_of_day' in df.columns:
        df['time_of_day'] = df['time_of_day'].astype(int)

    return df

def clean_data(df):
    unique_ids = df['id'].unique()
    id_to_int = {id_val: i+1 for i, id_val in enumerate(unique_ids)}
    print("Id to int mapping: ", id_to_int)
    
    df['id_int'] = df['id'].map(id_to_int)
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second

    df['datetime'] = df['time']

    # 1-7 (monday - sunday)
    df['day_of_week'] = df['datetime'].dt.dayofweek + 1

    # look at function above, we can change definition based on the if statements
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 6 else 0)

    # Drop year? all is 2014 so its useless:
    df = df.drop(['id', 'time', 'Unnamed: 0', 'year'], axis=1, errors='ignore')
    df = df.rename(columns={'id_int': 'id'})

    df = ensure_correct_data_types(df)

    return df

# https://medium.com/@piyushkashyap045/handling-missing-values-in-data-a-beginner-guide-to-knn-imputation-30d37cc7a5b7
def deal_with_missing_data(df):
    # Calculate neighbours: (i just used 10 but we need to search which one is actually best based on our dataset)
    # It calculates the missing ones based on 10 nearest neigbours now
    missing_data_dfs = []

    # Group by id and variable since variable and value are correlated to each other
    for _, group in df.groupby(['id', 'variable']):
        features = ['month', 'day', 'hour', 'minute', 'second', 'value']

        if group['value'].isna().sum() > 0:
            imputer = KNNImputer(n_neighbors=10)
            group[features] = imputer.fit_transform(group[features])

        missing_data_dfs.append(group)

    result_df = pd.concat(missing_data_dfs)
    result_df.reset_index(drop=True, inplace=True)

    result_df = ensure_correct_data_types(result_df)

    return result_df

def handle_outliers(df):
    result_dfs = []

    var_stats = {}
    for var, group in df.groupby('variable'):
        var_stats[var] = {
            'median': group['value'].median(),
            'iqr': group['value'].quantile(0.75) - group['value'].quantile(0.25),
            'min': group['value'].min(),
            'max': group['value'].max()
        }


        unique_vals = group['value'].nunique()
        total_vals = len(group)
        var_stats[var]['is_discrete'] = (unique_vals / total_vals < 0.05) or (unique_vals < 10)

    for (id_val, var), group in df.groupby(['id', 'variable']):
        if len(group) <= 1:
            result_dfs.append(group)
            continue

        is_discrete = var_stats[var]['is_discrete']

        if is_discrete:

            median = group['value'].median()

            q1 = group['value'].quantile(0.25)
            q3 = group['value'].quantile(0.75)
            iqr = q3 - q1


            lower_bound = q1 - 2.5 * iqr if iqr > 0 else q1 - 1
            upper_bound = q3 + 2.5 * iqr if iqr > 0 else q3 + 1

            outliers = (group['value'] < lower_bound) | (group['value'] > upper_bound)
            group.loc[outliers, 'value'] = median

        else:
            group = group.sort_values('datetime')


            window_size = max(5, len(group) // 10)


            group['rolling_median'] = group['value'].rolling(window=window_size, min_periods=1).median()
            group['rolling_q1'] = group['value'].rolling(window=window_size, min_periods=1).quantile(0.25)
            group['rolling_q3'] = group['value'].rolling(window=window_size, min_periods=1).quantile(0.75)
            group['rolling_iqr'] = group['rolling_q3'] - group['rolling_q1']


            group['rolling_iqr'] = group['rolling_iqr'].fillna(var_stats[var]['iqr'])


            group['lower_bound'] = group['rolling_median'] - 1.5 * group['rolling_iqr']
            group['upper_bound'] = group['rolling_median'] + 1.5 * group['rolling_iqr']


            outliers = (group['value'] < group['lower_bound']) | (group['value'] > group['upper_bound'])


            group.loc[outliers, 'value'] = group.loc[outliers, 'rolling_median']

            group = group.drop(['rolling_median', 'rolling_q1', 'rolling_q3', 'rolling_iqr',
                               'lower_bound', 'upper_bound'], axis=1)

        result_dfs.append(group)

    result_df = pd.concat(result_dfs)
    return result_df.sort_index()


# Part of feature engineering
def normalize_or_scale():
    pass

# Doesnt show much correlation
def plot_corr_diagram(df):
    plot_df = df.copy()

    non_numeric_cols = []
    for col in plot_df.columns:
        if plot_df[col].dtype == 'object' or plot_df[col].dtype == 'datetime64[ns]':
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"Dropping non-numeric columns for correlation analysis: {non_numeric_cols}")
        plot_df = plot_df.drop(columns=non_numeric_cols)

    corr_matrix = plot_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, annot_kws={"size": 8})
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    print("Correlation matrix saved as 'correlation_matrix.png'")
    plt.show()

def main():
    folder_path = "data_sets_after_cleaning"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df = pd.read_csv("dataset_mood_smartphone.csv", parse_dates=['time'])

    cleaned_df = clean_data(df)
    cleaned_df.to_csv(folder_path + "/cleaned_with_missing.csv", index=False)

    print(f"Missing values per column: \n{cleaned_df.isnull().sum()}\n")
    print(f"Missing values by id: {df[df['value'].isna()].groupby('id').size()}")

    no_missing_data_df_knn = deal_with_missing_data(cleaned_df)
    no_outliers_df = handle_outliers(no_missing_data_df_knn)
    final_df = pd.get_dummies(no_outliers_df, columns=['variable'], prefix='variable') # One hot encoding variable
    final_df.to_csv(folder_path + "/cleaned_without_missing.csv", index=False)

    plot_corr_diagram(no_missing_data_df_knn)

if __name__ == '__main__':
    main()
