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
        
        :param wheel_angle: Array of steering angles (in degrees).
        :param speed: Array of speed values (in m/s).
        :param sampling_frequency: Sampling frequency of the data (in Hz).
        :return: x_path, y_path: Arrays of x and y positions.
        """
        # Initialize position and orientation
        x, y, theta = 0, 0, 0
        dt = 1 / sampling_frequency
        
        # Convert angles from degrees to radians
        # wheel_angle = np.radians(wheel_angle)
        
        # Initialize arrays to hold the path
        x_path = [x]
        y_path = [y]
        
        # Loop over all data points to calculate the path
        for angle, spd in zip(wheel_angle, speed):
            if spd == 0:
                # If the speed is zero, the vehicle doesn't move.
                x_path.append(x)
                y_path.append(y)
                continue
            
            # Calculate the turning radius (R)
            if angle != 0:
                R = self.wheel_base / np.tan(angle)
            else:
                R = np.inf  # Going straight
            
            # Update the orientation (theta)
            if R != np.inf:
                d_theta = spd * dt / R
            else:
                d_theta = 0
            
            theta += d_theta
            
            # Update x and y positions
            dx = spd * np.cos(theta) * dt
            dy = spd * np.sin(theta) * dt
            
            x += dx
            y += dy
            
            x_path.append(x)
            y_path.append(y)
        
        return np.array(x_path), np.array(y_path)
