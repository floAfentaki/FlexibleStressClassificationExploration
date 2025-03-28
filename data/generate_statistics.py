import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew
from scipy import integrate, stats
import neurokit2 as nk
import biosppy
import pyhrv
import sys
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from numpy import trapz  # Import trapz from numpy


def compute_statistics_for_column(column, df, window_size, sampling_rate=64):
    rolling_stats = df[column].rolling(window=window_size)
    stats_dict = {
        f'{column}_mean': rolling_stats.mean(),
        f'{column}_std': rolling_stats.std(),
        f'{column}_sum': rolling_stats.sum(),
        f'{column}_min': rolling_stats.min(),
        f'{column}_max': rolling_stats.max(),
        f'{column}_median': rolling_stats.median(),
        f'{column}_range': rolling_stats.max() - rolling_stats.min(),
        f'{column}_kurtosis': rolling_stats.apply(lambda x: kurtosis(x, nan_policy='omit') if len(x) > 1 else np.nan, raw=True),
        f'{column}_skewness': rolling_stats.apply(lambda x: skew(x, nan_policy='omit') if len(x) > 1 else np.nan, raw=True)
    }
    # X,Y,Z,EDA,BVP,TEMP,id,datetime,label

    if column == 'TEMP':
        stats_dict[f'{column}_slope'] = rolling_stats.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_range'] = rolling_stats.max() - rolling_stats.min()
    elif column in ['X', 'Y', 'Z']:
        stats_dict[f'{column}_auc'] = rolling_stats.apply(lambda x: trapz(x) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_mean_peaks'] = rolling_stats.apply(lambda x: np.mean(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
    elif column in ['ACC_X', 'ACC_Y', 'ACC_Z']:
        stats_dict[f'{column}_auc'] = rolling_stats.apply(lambda x: trapz(x) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_mean_peaks'] = rolling_stats.apply(lambda x: np.mean(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
    elif 'EDA' in column:
        stats_dict[f'{column}_slope'] = rolling_stats.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_range'] = rolling_stats.max() - rolling_stats.min()
        stats_dict[f'{column}_auc'] = rolling_stats.apply(lambda x: trapz(x) if len(x) > 1 else np.nan, raw=True)
    elif column == 'EMG':
        stats_dict[f'{column}_range'] = rolling_stats.max() - rolling_stats.min()
        stats_dict[f'{column}_auc'] = rolling_stats.apply(lambda x: trapz(x) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_median'] = rolling_stats.median()
        stats_dict[f'{column}_10p'] = rolling_stats.apply(lambda x: np.percentile(x, 10) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_90p'] = rolling_stats.apply(lambda x: np.percentile(x, 90) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_mu_peak'] = rolling_stats.apply(lambda x: np.mean(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_std_peak'] = rolling_stats.apply(lambda x: np.std(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_n_peak'] = rolling_stats.apply(lambda x: len(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_sum_peak'] = rolling_stats.apply(lambda x: np.sum(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
    elif column == 'ECG':
        stats_dict[f'{column}_sum_peak'] = rolling_stats.apply(lambda x: np.sum(find_peaks(x)[0]) if len(x) > 1 else np.nan, raw=True)
        if dataset != 'DriveDB':
            stats_dict[f'{column}_LF/HF_ratio'] = rolling_stats.apply(lambda x: nk.hrv(x, sampling_rate=sampling_rate)['HRV_LF/HF'] if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_sum_power'] = rolling_stats.apply(lambda x: np.sum(np.abs(np.fft.fft(x))**2) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_relative_power'] = rolling_stats.apply(lambda x: np.sum(np.abs(np.fft.fft(x))**2) / np.sum(np.abs(np.fft.fft(df[column]))**2) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_sum_norm'] = rolling_stats.apply(lambda x: np.sum(x / np.linalg.norm(x)) if len(x) > 1 else np.nan, raw=True)
        stats_dict[f'{column}_rmssd'] = rolling_stats.apply(lambda x: np.sqrt(np.mean(np.diff(x)**2)) if len(x) > 1 else np.nan, raw=True)
    
    return stats_dict

def majority_label(x):
    return pd.Series(x).mode()[0] if not pd.Series(x).empty else np.nan

def compute_statistics(df, window_size, sampling_rate):
    statistics_df = pd.DataFrame()
    
    for column in df.columns:
        if column != 'label':
            stats_dict = compute_statistics_for_column(column, df, window_size)
            for key, value in stats_dict.items():
                statistics_df[key] = value
    
    statistics_df['label'] = df['label'].rolling(window=window_size).apply(majority_label, raw=False)
    statistics_df.dropna(inplace=True)
    return statistics_df

def process_csv_files(file_name, window_size, sampling_rate):
    if not os.path.exists(file_name):
        print(f'File {file_name} not found')
        return None
    
    print(f'Processing {file_name}')
    df = pd.read_csv(file_name, delimiter=',')
    # df = df.iloc[:1000]
    df = df.iloc[:50000]
    
    # df.to_csv('stress_in_nurses_test.csv', index=False)
    # print(df)

    if 'Timestamp' in df.columns:
        df.drop(columns=['Timestamp'], inplace=True)
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'datetime' in df.columns:
        df.drop(columns=['datetime'], inplace=True)

    # print(df)
    # df=df.astype(float)
    
    # for column in df.columns:
    #     if column not in ['label', 'Timestamp']:
    #         typ = column.lower()
    #         extracted_features = compute_complex_statistics(df[column], typ, sampling_rate)
    #         df = pd.concat([df, extracted_features], axis=1)
    
    statistics_df = compute_statistics(df, window_size, sampling_rate)
    output_file_name = file_name.replace('.csv', f"_statistics_{window_size}.csv")
    statistics_df.to_csv(output_file_name, index=False)
    print(f'Statistics saved to {output_file_name}')
    return output_file_name

def train_mlp(file_path):
    if file_path is None:
        return
    df = pd.read_csv(file_path)
    
    if 'label' not in df.columns:
        print(f'label column missing in {file_path}')
        return
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'MLP Accuracy: {accuracy}')

def process_and_train(dataset, sampling_rate, window_size):
   
    samples = int(window_size * sampling_rate)
    print(f'Processing dataset: {dataset} with window size: {window_size} ({samples} samples)')
    file_path = f'{dataset}.csv'
    statistics_file_path = process_csv_files(file_path, samples, sampling_rate)
    train_mlp(statistics_file_path)

if __name__ == "__main__":
    # datasets = ['AffectiveROAD', 'spd', 'DriveDB']
    # datasets = ['stress_in_nurses_test']
    datasets = ['DriveDB']
    # sampling_rates = [64, 64, 16]
    sampling_rates = [16]
    window_sizes = [1, 1.5, 3, 5]

    

    for dataset, sampling_rate in zip(datasets, sampling_rates):
        # log_file = f'{dataset}.log'
        # bkstdout = sys.stdout
        # with open(log_file, 'w') as f:
        #     sys.stdout = f
            for window_size in window_sizes:
                process_and_train(dataset, sampling_rate, window_size)
        # sys.stdout = bkstdout