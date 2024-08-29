# main_app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import classes from utils.py
from utils import DataLoader, DataProcessor, PathReconstructor

def main():
    st.title("CorrDash: Dashboard for Data Analysis and Visualization")

    # Initialize the DataLoader, DataProcessor, and PathReconstructor classes
    data_loader = DataLoader()
    data_processor = DataProcessor()
    path_reconstructor = PathReconstructor(wheel_base=2.5)

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
        data = data_loader.load_data(file_path=uploaded_file, s3_url=s3_url)
        if data is not None:
            st.sidebar.header("Metadata Analysis")
            metadata_field = st.sidebar.selectbox("Select metadata field", data.columns)
            st.sidebar.header("Preprocessing Options")
            preprocessing_option = st.sidebar.selectbox("Choose preprocessing method", ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"])

            # Adjust settings based on preprocessing method
            if preprocessing_option == 'Smoothing':
                window_size = st.sidebar.slider("Smoothing window size", min_value=1, max_value=50, value=5)
                preprocessed_data, preprocessing_label = data_processor.preprocess_data(data, preprocessing_option, window=window_size)
            elif preprocessing_option == 'Band-Pass Filter':
                sampling_frequency = st.sidebar.number_input("Sampling Frequency for Band-Pass (Hz)", value=100, min_value=1)
                lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", value=0.5, min_value=0.1, max_value=100.0, step=0.1)
                highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", value=30.0, min_value=0.1, max_value=100.0, step=0.1)
                preprocessed_data, preprocessing_label = data_processor.preprocess_data(data, preprocessing_option, sampling_frequency, lowcut=lowcut, highcut=highcut)
            else:
                preprocessed_data, preprocessing_label = data_processor.preprocess_data(data, preprocessing_option)

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
                    x_path, y_path = path_reconstructor.calculate_path(data['wheel_angle'] * conversion_ratio, data['speed'], sampling_frequency_single)
                    
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
