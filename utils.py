# utils.py
import pandas as pd
import numpy as np
import boto3
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from io import BytesIO
import streamlit as st

class DataLoader:
    def __init__(self):
        # Initialize the S3 client using default credentials from the environment
        self.s3 = boto3.client('s3')
    
    @st.cache_data
    def load_data(_self, file_path=None, s3_url=None):
        if file_path:
            return pd.read_parquet(file_path)
        elif s3_url:
            return _self.load_data_from_s3(s3_url)
        return None

    @st.cache_data
    def load_data_from_s3(_self, s3_url):
        # Split the S3 URL into bucket and key
        s3_bucket, s3_key = s3_url.replace("s3://", "").split("/", 1)
        response = _self.s3.get_object(Bucket=s3_bucket, Key=s3_key)
        return pd.read_parquet(BytesIO(response['Body'].read()))

class DataProcessor:
    def __init__(self):
        # Initialize any attributes if necessary
        pass
    
    @st.cache_data
    def preprocess_data(_self, data, method, sampling_frequency=None, window=None, lowcut=None, highcut=None):
        if method == 'Derivative':
            return data.diff().fillna(0), 'Derivative'
        elif method == 'Z-Score':
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(data), columns=data.columns), 'Z-Score'
        elif method == 'Smoothing':
            return data.rolling(window=window).mean().fillna(method='bfill'), f'Smoothing (Window: {window})'
        elif method == 'Band-Pass Filter':
            nyquist = 0.5 * sampling_frequency
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(4, [low, high], btype='band')
            return pd.DataFrame(filtfilt(b, a, data, axis=0), columns=data.columns), f'Band-Pass Filter ({lowcut}-{highcut} Hz)'
        return data, 'None'

class PathReconstructor:
    def __init__(self, wheel_base=2.5):
        self.wheel_base = wheel_base

    def calculate_path(self, wheel_angle, speed, sampling_frequency):
        """
        Reconstruct the path of the car using a bicycle model.
        
        :param wheel_angle: Array of steering angles (in radians).
        :param speed: Array of speed values (in m/s).
        :param sampling_frequency: Sampling frequency of the data (in Hz).
        :return: x_path, y_path: Arrays of x and y positions.
        """
        x, y, theta = 0, 0, 0
        dt = 1 / sampling_frequency
        
        x_path, y_path = [x], [y]
        
        for angle, spd in zip(wheel_angle, speed):
            if spd == 0:
                x_path.append(x)
                y_path.append(y)
                continue

            if angle != 0:
                R = self.wheel_base / np.tan(angle)
            else:
                R = np.inf
            
            if R != np.inf:
                d_theta = spd * dt / R
            else:
                d_theta = 0
            
            theta += d_theta
            x += spd * np.cos(theta) * dt
            y += spd * np.sin(theta) * dt
            
            x_path.append(x)
            y_path.append(y)
        
        return np.array(x_path), np.array(y_path)

    def calculate_similarity(self, path1, path2):
        """
        Calculate a similarity index between two paths.
        
        :param path1: Tuple of (x_path, y_path) for the first path.
        :param path2: Tuple of (x_path, y_path) for the second path.
        :return: similarity_index: A value between 0 and 1.
        """
        x1, y1 = path1
        x2, y2 = path2

        # Normalize paths to account for differences in starting points
        x1 -= x1[0]
        y1 -= y1[0]
        x2 -= x2[0]
        y2 -= y2[0]

        # Resample paths to the same length for comparison
        len1 = len(x1)
        len2 = len(x2)
        length = max(len1, len2)

        x1_resampled = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len1), x1)
        y1_resampled = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len1), y1)
        x2_resampled = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len2), x2)
        y2_resampled = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len2), y2)

        # Calculate Euclidean distance between resampled paths
        distance = np.sqrt((x1_resampled - x2_resampled) ** 2 + (y1_resampled - y2_resampled) ** 2)
        
        # Normalize distance to a similarity index between 0 and 1
        max_distance = np.sqrt((x1_resampled.max() - x1_resampled.min())**2 + (y1_resampled.max() - y1_resampled.min())**2)
        similarity_index = 1 - np.clip(np.mean(distance / max_distance), 0, 1)
        
        return similarity_index

    def calculate_similarity_matrix(self, paths):
        """
        Calculate a similarity matrix for multiple paths.
        
        :param paths: Dictionary of {name: (x_path, y_path)} for each file.
        :return: similarity_matrix: DataFrame containing similarity indices.
        """
        names = list(paths.keys())
        similarity_matrix = np.zeros((len(names), len(names)))

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i <= j:  # Compute only for upper triangle and diagonal
                    similarity_matrix[i, j] = self.calculate_similarity(paths[name1], paths[name2])
                    similarity_matrix[j, i] = similarity_matrix[i, j]

        return pd.DataFrame(similarity_matrix, index=names, columns=names)
