import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
# command for me please do not delete: conda activate datamining_env
# Configuration
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# Load and prepare data
def load_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath, parse_dates=['time'])
    df['date'] = df['time'].dt.date
    df['day_of_week'] = df['time'].dt.day_name()
    df['hour'] = df['time'].dt.hour
    return df


# Dataset summary statistics
def get_dataset_summary(df):
    """Generate overall dataset statistics"""
    return {
        'Total Records': len(df),
        'Unique Patients': df['id'].nunique(),
        'Time Range Start': df['time'].min(),
        'Time Range End': df['time'].max(),
        'Total Variables': df['variable'].nunique(),
        'Total Missing Values': df['value'].isna().sum(),
        'Missing Percentage': f"{df['value'].isna().mean() * 100:.1f}%",
        'Recording Days': df['date'].nunique(),
        'Avg Records per Day': f"{len(df) / df['date'].nunique():.1f}"
    }


# Variable-level statistics
def get_variable_stats(df):
    """Generate detailed statistics for each variable"""
    stats = []
    for var in df['variable'].unique():
        var_data = df[df['variable'] == var]['value'].dropna()
        is_continuous = var_data.nunique() > 10

        stats.append({
            'Variable': var,
            'Type': 'Continuous' if is_continuous else 'Categorical',
            'Records': len(var_data),
            'Missing (%)': f"{(1 - len(var_data) / len(df[df['variable'] == var])) * 100:.1f}",
            'Min': var_data.min() if is_continuous else '-',
            'Max': var_data.max() if is_continuous else '-',
            'Mean': f"{var_data.mean():.2f}" if is_continuous else '-',
            'Median': f"{var_data.median():.2f}" if is_continuous else '-',
            'Std Dev': f"{var_data.std():.2f}" if is_continuous else '-',
            'Unique Values': var_data.nunique(),
            'Most Frequent': var_data.mode().values[0] if not is_continuous else '-'
        })
    return pd.DataFrame(stats)


# Visualization functions
def plot_distributions(df, variables, plot_dir=PLOT_DIR):
    """Plot distributions for all variables"""
    for var in variables:
        var_data = df[df['variable'] == var]['value'].dropna()
        plt.figure(figsize=(10, 5))

        if var_data.nunique() > 10:
            sns.histplot(var_data, bins=20, kde=True)
            plt.title(f'Distribution of {var}\n(Skewness: {var_data.skew():.2f})')
        else:
            value_counts = var_data.value_counts().sort_index()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Value Counts for {var}')

        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/distribution_{var}.png')
        plt.close()


def plot_correlations(df, numeric_vars, plot_dir=PLOT_DIR):
    """Plot correlation matrix for numeric variables"""
    print(f"Original numeric_vars: {numeric_vars}")

    filtered_df = df[df['variable'].isin(numeric_vars)]
    print(f"Unique variables after filtering: {filtered_df['variable'].unique()}")

    pivot_df = filtered_df.pivot(index=['id', 'time'], columns='variable', values='value')

    print(f"Columns in pivoted DataFrame: {pivot_df.columns.tolist()}")
    print(f"Shape of pivoted DataFrame: {pivot_df.shape}")

    missing_vars = set(numeric_vars) - set(pivot_df.columns)
    if missing_vars:
        print(f"Warning: Missing variables in pivoted data: {missing_vars}")

    # Calculate correlations
    corr_matrix = pivot_df.corr()
    # Check the correlation matrix
    print("Correlation matrix:")
    print(corr_matrix)

    # Plotting
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, fmt='.2f', annot_kws={'size': 8})
    plt.title('Correlation Between Continuous Variables', pad=20)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/correlation_matrix.png')
    plt.close()

def plot_temporal_patterns(df, target_var='mood', plot_dir=PLOT_DIR):
    """Plot temporal patterns for target variable"""
    target_data = df[df['variable'] == target_var].copy()

    # Daily patterns
    plt.figure(figsize=(12, 5))
    daily_avg = target_data.groupby('date')['value'].mean()
    plt.plot(daily_avg.index, daily_avg.values)
    plt.title(f'Daily Average {target_var.capitalize()} Trend')
    plt.xlabel('Date')
    plt.ylabel(f'{target_var.capitalize()} Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/daily_{target_var}_trend.png')
    plt.close()

    # Weekly patterns
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='day_of_week', y='value', data=target_data,
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                       'Friday', 'Saturday', 'Sunday'])
    plt.title(f'{target_var.capitalize()} Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel(f'{target_var.capitalize()} Score')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/weekly_{target_var}_pattern.png')
    plt.close()


# Main analysis
def main():
    # Load data
    df = load_data('dataset_mood_smartphone.csv')

    # Generate statistics
    dataset_summary = pd.DataFrame([get_dataset_summary(df)]).T
    variable_stats = get_variable_stats(df)

    # Save statistics
    dataset_summary.to_csv('dataset_overview.csv', header=False)
    variable_stats.to_csv('variable_statistics.csv', index=False)

    # Generate visualizations
    numeric_vars = variable_stats[variable_stats['Type'] == 'Continuous']['Variable']
    plot_distributions(df, df['variable'].unique())
    plot_correlations(df, numeric_vars)
    plot_temporal_patterns(df)

    # Print summary
    print("=== Dataset Overview ===")
    print(dataset_summary.to_markdown())

    print("\n=== Variable Statistics ===")
    print(variable_stats.to_markdown(index=False))


if __name__ == '__main__':
    main()