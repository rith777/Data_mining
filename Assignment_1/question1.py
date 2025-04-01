import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# conda activate datamining_env

# Load the dataset
df = pd.read_csv('dataset_mood_smartphone.csv', parse_dates=['time'])

# Basic Dataset Properties
print("Basic Dataset Properties")
print(f"Total records: {len(df)}")
print(f"Unique patients: {df['id'].nunique()}")
print(f"Time range: {df['time'].min()} to {df['time'].max()}")
print(f"Unique variables: {df['variable'].unique()}")

# Data Quality Check
print("\nMissing Values")
print(df.isnull().sum())

# Value Distribution for Each Variable
print("\nValue Distributions")
for var in df['variable'].unique():
    var_data = df[df['variable'] == var]['value']
    print(f"\nVariable: {var}")
    print(var_data.describe())

    # Plot histograms for continuous variables
    if var_data.nunique() > 10:
        plt.figure()
        sns.histplot(var_data.dropna(), bins=20)
        plt.title(f'Distribution of {var}')
        plt.show()
    else:
        plt.figure()
        var_data.value_counts().sort_index().plot(kind='bar')
        plt.title(f'Counts of {var} values')
        plt.show()

# Temporal Analysis
print("\n=== Temporal Analysis ===")
df['date'] = df['time'].dt.date
daily_counts = df.groupby('date').size()
plt.figure(figsize=(12, 6))
daily_counts.plot()
plt.title('Number of Records per Day')
plt.ylabel('Count')
plt.show()

# Mood Analysis (assuming mood is the key variable)
mood_data = df[df['variable'] == 'mood']
daily_mood = mood_data.groupby('date')['value'].mean()
plt.figure(figsize=(12, 6))
daily_mood.plot()
plt.title('Daily Average Mood')
plt.ylabel('Mood Score (1-10)')
plt.show()

# Correlation between variables (pivot to wide format)
pivot_df = df.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()
correlation_matrix = pivot_df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Between Variables')
plt.show()

# Save summary statistics to CSV for reporting
summary_stats = df.groupby('variable')['value'].describe()
summary_stats.to_csv('variable_summary_stats.csv')