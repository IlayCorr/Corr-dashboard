import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import boto3
import pyarrow.parquet as pq
from io import BytesIO

# Cache mechanism to reduce overhead time
@st.cache_data
def load_data(file_path=None, s3_url=None):
    if file_path:
        return pd.read_parquet(file_path)
    elif s3_url:
        return load_data_from_s3(s3_url)
    return None

@st.cache_data
def load_data_from_s3(s3_url):
    # Split the S3 URL into bucket and key
    s3_bucket, s3_key = s3_url.replace("s3://", "").split("/", 1)
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return pd.read_parquet(BytesIO(response['Body'].read()))

@st.cache_data
def preprocess_data(data, method, sampling_frequency, window=None, lowcut=None, highcut=None):
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

def calculate_path(wheel_angle, speed, sampling_frequency, wheel_base=2.5):
    """
    Reconstruct the path of the car using a bicycle model.
    
    :param wheel_angle: Array of steering angles (in degrees).
    :param speed: Array of speed values (in m/s).
    :param sampling_frequency: Sampling frequency of the data (in Hz).
    :param wheel_base: Distance between the front and rear axles (in meters).
    :return: x_path, y_path: Arrays of x and y positions.
    """
    # Initialize position and orientation
    x, y, theta = 0, 0, 0
    dt = 1 / sampling_frequency
    
    # Convert angles from degrees to radians
    wheel_angle = np.radians(wheel_angle)
    
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
            R = wheel_base / np.tan(angle)
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

def main():
    st.title("CorrDash: Dashboard for Data Analysis and Visualization")

    # Sidebar for file upload, S3 URL, and selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.selectbox("Select data source", ["Upload a file", "S3 URL"])

    if data_source == "Upload a file":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["parquet"])
        s3_url = None
    elif data_source == "S3 URL":
        s3_url = st.sidebar.text_input("Enter S3 URL (e.g., s3://bucket_name/path/to/file.parquet)")
        uploaded_file = None

    if uploaded_file or s3_url:
        data = load_data(file_path=uploaded_file, s3_url=s3_url)
        if data is not None:
            st.sidebar.header("Metadata Analysis")
            metadata_field = st.sidebar.selectbox("Select metadata field", data.columns)
            st.sidebar.header("Preprocessing Options")
            preprocessing_option = st.sidebar.selectbox("Choose preprocessing method", ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"])

            # Adjust settings based on preprocessing method
            if preprocessing_option == 'Smoothing':
                window_size = st.sidebar.slider("Smoothing window size", min_value=1, max_value=50, value=5)
                preprocessed_data, preprocessing_label = preprocess_data(data, preprocessing_option, None, window=window_size)
            elif preprocessing_option == 'Band-Pass Filter':
                sampling_frequency = st.sidebar.number_input("Sampling Frequency for Band-Pass (Hz)", value=100, min_value=1)
                lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", value=0.5, min_value=0.1, max_value=100.0, step=0.1)
                highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", value=30.0, min_value=0.1, max_value=100.0, step=0.1)
                preprocessed_data, preprocessing_label = preprocess_data(data, preprocessing_option, sampling_frequency, lowcut=lowcut, highcut=highcut)
            else:
                preprocessed_data, preprocessing_label = preprocess_data(data, preprocessing_option, None)

            st.header(f"Metadata Analysis: {metadata_field}")
            st.write(data[metadata_field].describe())
            st.write("Statistical Plots")
            fig, ax = plt.subplots()
            sns.histplot(data[metadata_field], kde=True, ax=ax)
            st.pyplot(fig)

            # Visualization of Single Drive Files
            st.sidebar.header("Single Drive Visualization")
            selected_signals = st.sidebar.multiselect("Select signals to display", data.columns)
            subplot_option = st.sidebar.checkbox("Use subplots for each signal", value=True)
            sampling_frequency_single = st.sidebar.number_input("Sampling Frequency for Single Drive (Hz)", value=100, min_value=1)
            
            if selected_signals:
                st.header("Single Drive File Visualization")
                time_vector = np.arange(len(preprocessed_data)) / sampling_frequency_single  # Calculate time vector
                fig, axes = plt.subplots(len(selected_signals), 1, sharex=True if subplot_option else False)
                if len(selected_signals) == 1:
                    axes = [axes]

                for i, signal in enumerate(selected_signals):
                    axes[i].plot(time_vector, preprocessed_data[signal].values, label=signal)
                    y_axis_label = signal if preprocessing_label == 'None' else f'{signal} ({preprocessing_label})'
                    axes[i].set_xlabel(f'Time (Frequency: {sampling_frequency_single} Hz)')
                    axes[i].set_ylabel(y_axis_label)
                    axes[i].legend()

                st.pyplot(fig)
            
            # Data Reconstruction: Path Plotting
            st.sidebar.header("Data Reconstruction")
            if 'wheel_angle' in data.columns and 'speed' in data.columns:
                st.sidebar.subheader("Conversion Ratio")
                conversion_ratio = st.sidebar.slider("Conversion Ratio", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

                if st.sidebar.button("Reconstruct Path"):
                    st.header("Reconstructed Path of the Car")
                    x_path, y_path = calculate_path(data['wheel_angle'] * conversion_ratio, data['speed'], sampling_frequency_single)
                    
                    fig, ax = plt.subplots()
                    ax.plot(x_path, y_path, label=f'Path (Conversion Ratio: {conversion_ratio})')
                    ax.set_xlabel('X Position (m)')
                    ax.set_ylabel('Y Position (m)')
                    ax.legend()
                    ax.set_aspect('equal', 'box')
                    st.pyplot(fig)
            else:
                st.sidebar.warning("Data must contain 'wheel_angle' and 'speed' columns for path reconstruction.")

if __name__ == "__main__":
    main()
