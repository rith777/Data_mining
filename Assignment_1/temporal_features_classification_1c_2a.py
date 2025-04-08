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

def main():
    folder_path = "data_sets_after_cleaning"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df = pd.read_csv("data_sets_after_cleaning/cleaned_without_missing.csv", parse_dates=['time'])

    temporal_df = create_temporal_features(df)
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
