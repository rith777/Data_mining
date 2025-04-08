import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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


def create_temporal_features(df, window_size='5D'):
    """
    Transforms raw time-series into a prediction-ready dataset with:
    - Features aggregated from 5-day windows
    - Targets aligned to next-day values
    """
    # Filter only mood data
    mood_df = df[df['variable_mood'] == 1].copy()
    mood_df = mood_df.sort_values(['id', 'datetime'])

    # Initialize feature storage
    features = []

    # Process each patient separately
    for patient_id, patient_data in mood_df.groupby('id'):
        patient_data = patient_data.set_index('datetime')

        # 5-day window
        patient_data['mood_5day_mean'] = patient_data['value'].rolling(window_size).mean()
        patient_data['mood_5day_std'] = patient_data['value'].rolling(window_size).std()

        # Lag features
        for lag in [1, 2, 3]:
            patient_data[f'mood_lag_{lag}'] = patient_data['value'].shift(lag)

        # Time-based features
        patient_data['target_mood'] = patient_data['value'].shift(-1)  

        # Store results
        features.append(patient_data.reset_index())

    # Combine all patients
    temporal_df = pd.concat(features)
    temporal_df = temporal_df.dropna()
    keep_cols = ['id', 'datetime', 'mood_5day_mean', 'mood_5day_std',
                 'mood_lag_1', 'mood_lag_2', 'mood_lag_3',
                 'day_of_week', 'time_of_day', 'is_weekend', 'target_mood']

    return temporal_df[keep_cols]

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

def discretize_mood(value):
    if value < 4:
        return 0  # Low mood
    elif 4 <= value <= 6:
        return 1  # Medium mood
    else:
        return 2  # High mood

def train_random_forest(df):
    # Drop datetime and target_mood for training
    X = df.drop(['datetime', 'target_mood', 'mood_class', 'id'], axis=1)
    y = df['mood_class']

    # Time-based split (because we have temporal data)
    df_sorted = df.sort_values('datetime')
    split_idx = int(len(df_sorted) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def prepare_rnn_data(df):
    # Convert target to mood class
    df['mood_class'] = df['target_mood'].apply(discretize_mood)

    # Feature selection from the temporal dataset
    feature_cols = ['mood_lag_1', 'mood_lag_2', 'mood_lag_3',
                    'mood_5day_mean', 'mood_5day_std',
                    'day_of_week', 'time_of_day', 'is_weekend']

    X = df[feature_cols].values
    y = df['mood_class'].values

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Treat each row as 1 time step with all features from temporal dataset
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # One-hot encode the labels
    y = to_categorical(y, num_classes=3)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_rnn_model(temporal_df):
    X_train, X_test, y_train, y_test = prepare_rnn_data(temporal_df)
    model = build_lstm_model(X_train.shape[1:])

    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nLSTM Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

    return model
# Part of feature engineering
def handle_outliers():
    pass

# Part of feature engineering
def normalize_or_scale():
    pass

# Doesnt show much correlation
def plot_corr_diagram(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
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
    no_missing_data_df_knn = pd.get_dummies(no_missing_data_df_knn, columns=['variable'], prefix='variable') # One hot encoding variable
    no_missing_data_df_knn.to_csv(folder_path + "/cleaned_without_missing.csv", index=False)

    print(no_missing_data_df_knn.describe)
    print(f"Missing values per column: \n{no_missing_data_df_knn.isnull().sum()}\n")
    print(f"Missing values by id: {no_missing_data_df_knn[no_missing_data_df_knn['value'].isna()].groupby('id').size()}")

    plot_corr_diagram(no_missing_data_df_knn)

    temporal_df = create_temporal_features(no_missing_data_df_knn)

    # Save results
    temporal_df.to_csv(folder_path + "/temporal_features_5day.csv", index=False)
    print("Temporal features saved. Sample:")
    print(temporal_df.head())
    temporal_df = discretize_mood(temporal_df)
    temporal_df.to_csv(folder_path + "/temporal_features_5day_class.csv", index=False)

    # Random Forest training (use the below vars to evaluate later)
    random_forest = train_random_forest(temporal_df)
    # RNN training
    lstm_model = train_rnn_model(temporal_df)


if __name__ == '__main__':
    main()
