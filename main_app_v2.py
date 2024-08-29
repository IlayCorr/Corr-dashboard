import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import DataLoader, DataProcessor, PathReconstructor

def main():
    # Add your logo at the top of the app
    st.image("logo.png", width=150)  # Adjust the width as needed
    st.title("CorrDash: Dashboard for Data Analysis and Visualization")

    data_loader = DataLoader()
    data_processor = DataProcessor()
    path_reconstructor = PathReconstructor(wheel_base=2.5)

    st.sidebar.header("Data Source")
    data_source = st.sidebar.selectbox("Select data source", ["Upload files", "S3 URL"])

    if data_source == "Upload files":
        uploaded_files = st.sidebar.file_uploader("Choose files", type=["parquet"], accept_multiple_files=True)
        s3_url = None
    elif data_source == "S3 URL":
        s3_url = st.sidebar.text_input("Enter S3 URL (e.g., s3://bucket_name/path/to/file.parquet)")
        uploaded_files = []

    if uploaded_files or s3_url:
        data_dict = {}
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                data = data_loader.load_data(file_path=uploaded_file)
                data_dict[uploaded_file.name] = data

        elif s3_url:
            data = data_loader.load_data(s3_url=s3_url)
            data_dict[s3_url] = data

        if data_dict:
            st.sidebar.header("Data Fields Statistical Analysis")
            metadata_field = st.sidebar.selectbox("Select data field", list(data_dict.values())[0].columns)
            show_on_same_figure_metadata = st.sidebar.checkbox("Show all data field graphs on the same figure")
            num_bins = st.sidebar.slider("Number of bins", min_value=10, max_value=100, value=50)

            st.sidebar.header("Preprocessing Options")
            preprocessing_option = st.sidebar.selectbox("Choose preprocessing method", ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"])

            if preprocessing_option == 'Smoothing':
                window_size = st.sidebar.slider("Smoothing window size", min_value=1, max_value=50, value=5)
            elif preprocessing_option == 'Band-Pass Filter':
                sampling_frequency = st.sidebar.number_input("Sampling Frequency for Band-Pass (Hz)", value=100, min_value=1)
                lowcut = st.sidebar.number_input("Low Cutoff Frequency (Hz)", value=0.5, min_value=0.1, max_value=100.0, step=0.1)
                highcut = st.sidebar.number_input("High Cutoff Frequency (Hz)", value=30.0, min_value=0.1, max_value=100.0, step=0.1)

            st.header(f"Data Fields Statistical Analysis: {metadata_field}")

            statistical_summaries = []
            combined_fig = go.Figure() if show_on_same_figure_metadata else None
            color_sequence = px.colors.qualitative.Set3  # Use a distinct color sequence

            for idx, (name, data) in enumerate(data_dict.items()):
                if metadata_field in data.columns:
                    if preprocessing_option == 'Smoothing':
                        preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option, window=window_size)
                    elif preprocessing_option == 'Band-Pass Filter':
                        preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option, sampling_frequency, lowcut=lowcut, highcut=highcut)
                    else:
                        preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option)

                    # Collect statistical summary for side-by-side display
                    summary = preprocessed_data[metadata_field].describe()
                    summary.name = name
                    statistical_summaries.append(summary)

                    # Plot using Plotly
                    if show_on_same_figure_metadata:
                        combined_fig.add_trace(go.Histogram(x=preprocessed_data[metadata_field], name=f"{name}",
                                                            marker_color=color_sequence[idx % len(color_sequence)], nbinsx=num_bins))
                    else:
                        fig = px.histogram(preprocessed_data, x=metadata_field, nbins=num_bins, title=f"{metadata_field} Distribution ({name})")
                        fig.update_layout(bargap=0.1, height=400, width=800)
                        st.plotly_chart(fig)
                else:
                    st.warning(f"The file '{name}' does not contain the field '{metadata_field}'.")

            # Display combined figure if checkbox is ticked
            if show_on_same_figure_metadata:
                combined_fig.update_layout(barmode='overlay', title=f"Combined {metadata_field} Distribution", height=400, width=800)
                combined_fig.update_traces(opacity=0.75)
                st.plotly_chart(combined_fig)

            # Display statistical summaries side by side
            if statistical_summaries:
                st.subheader("Statistical Summaries Side by Side")
                st.write(pd.concat(statistical_summaries, axis=1))

            # Visualization of Single Drive Files
            st.sidebar.header("Single Drive Visualization")
            selected_signals = st.sidebar.multiselect("Select signals to display", list(data_dict.values())[0].columns)
            show_on_same_figure_signals = st.sidebar.checkbox("Show graphs of the same field on the same figure")
            sampling_frequency_single = st.sidebar.number_input("Sampling Frequency for Single Drive (Hz)", value=100, min_value=1)

            if selected_signals:
                st.header("Single Drive File Visualization")
                time_vectors = {name: np.arange(len(data)) / sampling_frequency_single for name, data in data_dict.items()}
                
                signal_figures = {}
                for signal in selected_signals:
                    if show_on_same_figure_signals:
                        if signal not in signal_figures:
                            signal_figures[signal] = go.Figure()
                    else:
                        signal_figures[signal] = None

                    for idx, (name, data) in enumerate(data_dict.items()):
                        if signal in data.columns:
                            # Use the preprocessed data for visualization
                            if preprocessing_option == 'Smoothing':
                                preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option, window=window_size)
                            elif preprocessing_option == 'Band-Pass Filter':
                                preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option, sampling_frequency, lowcut=lowcut, highcut=highcut)
                            else:
                                preprocessed_data, _ = data_processor.preprocess_data(data, preprocessing_option)

                            if show_on_same_figure_signals:
                                signal_figures[signal].add_trace(go.Scatter(x=time_vectors[name], y=preprocessed_data[signal].values, mode='lines', 
                                                                            name=f'{signal} ({name})', line=dict(color=color_sequence[idx % len(color_sequence)])))
                            else:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=time_vectors[name], y=preprocessed_data[signal].values, mode='lines', name=f'{signal} ({name})'))
                                fig.update_layout(title=f"{signal} Visualization ({name})", xaxis_title="Seconds", yaxis_title=signal, height=400, width=800)
                                st.plotly_chart(fig)
                        else:
                            st.warning(f"The file '{name}' does not contain the signal '{signal}'.")

                # Display combined figures for each signal
                if show_on_same_figure_signals:
                    for signal, fig in signal_figures.items():
                        fig.update_layout(title=f"Combined {signal} Visualization", xaxis_title="Seconds", yaxis_title=signal, height=400, width=800)
                        st.plotly_chart(fig)

            st.sidebar.header("Data Reconstruction")
            show_on_same_figure_reconstruction = st.sidebar.checkbox("Show all reconstructed paths on the same figure")
            if all(['wheel_angle' in data.columns and 'speed' in data.columns for data in data_dict.values()]):
                st.sidebar.subheader("Conversion Ratio")
                conversion_ratio = st.sidebar.slider("Conversion Ratio", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

                if st.sidebar.button("Reconstruct Path"):
                    st.header("Reconstructed Path of the Car")

                    combined_fig = go.Figure() if show_on_same_figure_reconstruction else None
                    for idx, (name, data) in enumerate(data_dict.items()):
                        x_path, y_path = path_reconstructor.calculate_path(data['wheel_angle'] * conversion_ratio, data['speed'], sampling_frequency_single)
                        if show_on_same_figure_reconstruction:
                            combined_fig.add_trace(go.Scatter(x=x_path, y=y_path, mode='lines', name=f'Path ({name})',
                                                              line=dict(color=color_sequence[idx % len(color_sequence)])))
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x_path, y=y_path, mode='lines', name=f'Path ({name})'))
                            fig.update_layout(title=f"Reconstructed Path ({name})", xaxis_title="X Position (m)", yaxis_title="Y Position (m)", height=400, width=800)
                            st.plotly_chart(fig)

                    # Display combined figure if checkbox is ticked
                    if show_on_same_figure_reconstruction:
                        combined_fig.update_layout(title="Combined Reconstructed Paths", xaxis_title="X Position (m)", yaxis_title="Y Position (m)", height=400, width=800)
                        st.plotly_chart(combined_fig)
            else:
                st.sidebar.warning("Data must contain 'wheel_angle' and 'speed' columns for path reconstruction.")

if __name__ == "__main__":
    main()
