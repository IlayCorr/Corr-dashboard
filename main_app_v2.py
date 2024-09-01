import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from utils import DataLoader, DataProcessor, PathReconstructor

def slider_with_input_sidebar(label, min_value, max_value, value, step, key):
    # Create a two-column layout within the sidebar
    col1, col2 = st.sidebar.columns([3, 1])

    # Slider in the first column
    slider_value = col1.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=f"{key}_slider")

    # Number input in the second column
    number_input_value = col2.number_input("", min_value=min_value, max_value=max_value, value=slider_value, step=step, key=f"{key}_input")

    # Sync the slider and number input
    if number_input_value != slider_value:
        slider_value = number_input_value

    return slider_value

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the logo image in the same directory as the script
    logo_path = os.path.join(script_dir, "logo.png")
    
    # Load the logo image if it exists
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.warning("Logo image could not be found.")
    
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
        all_columns = set()  # To hold all unique columns across all files
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                data = data_loader.load_data(file_path=uploaded_file)
                data_dict[uploaded_file.name] = data
                all_columns.update(data.columns)  # Add columns from this file to the set

        elif s3_url:
            data = data_loader.load_data(s3_url=s3_url)
            data_dict[s3_url] = data
            all_columns.update(data.columns)  # Add columns from the S3 file to the set

        if data_dict:
            st.sidebar.header("Data Fields Statistical Analysis")
            selected_fields = st.sidebar.multiselect("Select data fields", list(all_columns))  # Allow multiple field selection
            show_on_same_figure = st.sidebar.checkbox("Show graphs of the same field on the same figure", key="data_fields_checkbox")
            num_bins = slider_with_input_sidebar("Number of bins", min_value=10, max_value=100, value=50, step=1, key="num_bins")

            st.header("Data Fields Statistical Analysis")

            if selected_fields:
                color_sequence = px.colors.qualitative.Set3  # Use a distinct color sequence

                for field in selected_fields:
                    combined_fig = go.Figure() if show_on_same_figure else None
                    statistical_summaries = []

                    for idx, (name, data) in enumerate(data_dict.items()):
                        if field in data.columns:
                            # Collect statistical summary for the field
                            summary = data[field].describe()
                            summary.name = f"{name}"
                            statistical_summaries.append(summary)

                            # Plot using Plotly
                            if show_on_same_figure:
                                combined_fig.add_trace(go.Histogram(x=data[field], name=f"{name} - {field}",
                                                                    marker_color=color_sequence[idx % len(color_sequence)], nbinsx=num_bins))
                            else:
                                # Separate figures for each dataset and field when the checkbox is unticked
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(x=data[field], name=f"{name} - {field}",
                                                           marker_color=color_sequence[idx % len(color_sequence)], nbinsx=num_bins))
                                fig.update_layout(title=f"{field} Distribution ({name})", height=400, width=800)
                                st.plotly_chart(fig)
                        else:
                            st.warning(f"The file '{name}' does not contain the field '{field}'.")

                    # Display combined figure for the current field if checkbox is ticked
                    if show_on_same_figure and combined_fig:
                        combined_fig.update_layout(barmode='overlay', title=f"Combined {field} Distributions", height=400, width=800)
                        combined_fig.update_traces(opacity=0.75)
                        st.plotly_chart(combined_fig)

                    # Display statistical summaries for the current field
                    if statistical_summaries:
                        st.subheader(f"Statistical Summaries for {field}")
                        st.write(pd.concat(statistical_summaries, axis=1))

            # Visualization of Signals
            st.sidebar.header("Signal Visualization")
            selected_signals = st.sidebar.multiselect("Select signals to display", list(all_columns))  # Use all_columns here
            show_on_same_figure_signals = st.sidebar.checkbox("Show graphs of the same field on the same figure", key="signal_visualization_checkbox")
            sampling_frequency_single = st.sidebar.number_input("Sampling Frequency for Signal Visualization (Hz)", value=100, min_value=1)

            # Initialize session state for storing the sequence of preprocessing methods
            if 'preprocessing_steps' not in st.session_state:
                st.session_state.preprocessing_steps = []

            # Add button to insert new preprocessing steps
            if st.sidebar.button("Add preprocessing step"):
                st.session_state.preprocessing_steps.append({'method': 'None'})

            # Display the preprocessing steps
            for i, step in enumerate(st.session_state.preprocessing_steps):
                method = st.sidebar.selectbox(f"Step {i+1}: Select preprocessing method", 
                                              ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"], 
                                              index=["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"].index(step['method']),
                                              key=f"preprocess_{i}")
                st.session_state.preprocessing_steps[i]['method'] = method

                # Collect additional parameters if required
                if method == "Smoothing":
                    window_size = slider_with_input_sidebar("Smoothing window size", min_value=1, max_value=50, value=5, step=1, key=f"smoothing_{i}")
                    st.session_state.preprocessing_steps[i]['params'] = {'window_size': window_size}
                elif method == "Band-Pass Filter":
                    sampling_frequency = st.sidebar.number_input("Sampling Frequency for Band-Pass (Hz)", value=100, min_value=1, key=f"sampling_frequency_{i}")
                    lowcut = slider_with_input_sidebar("Low Cutoff Frequency (Hz)", min_value=0.1, max_value=100.0, value=0.5, step=0.01, key=f"lowcut_{i}")
                    highcut = slider_with_input_sidebar("High Cutoff Frequency (Hz)", min_value=0.1, max_value=100.0, value=30.0, step=0.01, key=f"highcut_{i}")
                    st.session_state.preprocessing_steps[i]['params'] = {
                        'sampling_frequency': sampling_frequency,
                        'lowcut': lowcut,
                        'highcut': highcut
                    }

            if selected_signals:
                st.header("Signal Visualization")
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
                            # Apply preprocessing methods in the specified order
                            preprocessed_data = data.copy()
                            preprocessing_description = []
                            for step in st.session_state.preprocessing_steps:
                                method = step['method']
                                if method == "Smoothing":
                                    params = step['params']
                                    preprocessed_data, description = data_processor.preprocess_data(preprocessed_data, 'Smoothing', window=params['window_size'])
                                elif method == "Band-Pass Filter":
                                    params = step['params']
                                    preprocessed_data, description = data_processor.preprocess_data(preprocessed_data, 'Band-Pass Filter', 
                                                                                         sampling_frequency=params['sampling_frequency'], 
                                                                                         lowcut=params['lowcut'], 
                                                                                         highcut=params['highcut'])
                                elif method != 'None':
                                    preprocessed_data, description = data_processor.preprocess_data(preprocessed_data, method)
                                
                                if method != 'None':
                                    preprocessing_description.append(description)

                            # Construct the title with preprocessing information
                            title = f"{signal} Visualization ({name})"
                            if preprocessing_description:
                                title += f" - Preprocessing: {' -> '.join(preprocessing_description)}"

                            # Plot the preprocessed data
                            if show_on_same_figure_signals:
                                signal_figures[signal].add_trace(go.Scatter(x=time_vectors[name], y=preprocessed_data[signal].values, mode='lines', 
                                                                            name=f'{signal} ({name})', line=dict(color=color_sequence[idx % len(color_sequence)])))
                            else:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=time_vectors[name], y=preprocessed_data[signal].values, mode='lines', name=f'{signal} ({name})'))
                                fig.update_layout(title=title, xaxis_title="Seconds", yaxis_title=signal, height=400, width=800)
                                st.plotly_chart(fig)
                        else:
                            st.warning(f"The file '{name}' does not contain the signal '{signal}'.")

                # Display combined figures for each signal if checkbox is ticked
                if show_on_same_figure_signals:
                    for signal, fig in signal_figures.items():
                        title = f"Combined {signal} Visualization"
                        if preprocessing_description:
                            title += f" - Preprocessing: {' -> '.join(preprocessing_description)}"
                        fig.update_layout(title=title, xaxis_title="Seconds", yaxis_title=signal, height=400, width=800)
                        st.plotly_chart(fig)

            # Add a subsection for cross-field visualization
            st.sidebar.subheader("Cross-Field Visualization")
            field_x = st.sidebar.selectbox("Select X-axis field", ["None"] + list(all_columns), key="field_x")
            field_y = st.sidebar.selectbox("Select Y-axis field", ["None"] + list(all_columns), key="field_y")
            method_x = st.sidebar.selectbox(f"Select preprocessing for {field_x if field_x != 'None' else 'X-axis'}", 
                                            ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"],
                                            key="method_x")

            method_y = st.sidebar.selectbox(f"Select preprocessing for {field_y if field_y != 'None' else 'Y-axis'}", 
                                            ["None", "Derivative", "Z-Score", "Smoothing", "Band-Pass Filter"],
                                            key="method_y")

            transparency = slider_with_input_sidebar("Dot Transparency", min_value=0.01, max_value=1.0, value=0.8, step=0.01, key="transparency")
            
            if field_x != "None" and field_y != "None":
                st.header(f"Cross-Field Visualization: {field_x} vs {field_y}")
                
                for idx, (name, data) in enumerate(data_dict.items()):
                    if field_x in data.columns and field_y in data.columns:
                        # Apply preprocessing on X-axis
                        preprocessed_x = data[field_x].copy()
                        preprocessed_y = data[field_y].copy()
                        preprocessing_description_x = method_x
                        preprocessing_description_y = method_y

                        if method_x != "None":
                            if method_x == "Smoothing":
                                window_size = slider_with_input_sidebar(f"Smoothing window size for {field_x}", min_value=1, max_value=50, value=5, step=1, key=f"smoothing_x_{name}")
                                preprocessed_x, preprocessing_description_x = data_processor.preprocess_data(data[[field_x]], 'Smoothing', window=window_size)
                            elif method_x == "Band-Pass Filter":
                                sampling_frequency = st.sidebar.number_input(f"Sampling Frequency for Band-Pass (Hz) for {field_x}", value=100, min_value=1, key=f"sampling_frequency_x_{name}")
                                lowcut = slider_with_input_sidebar(f"Low Cutoff Frequency (Hz) for {field_x}", min_value=0.1, max_value=100.0, value=0.5, step=0.01, key=f"lowcut_x_{name}")
                                highcut = slider_with_input_sidebar(f"High Cutoff Frequency (Hz) for {field_x}", min_value=0.1, max_value=100.0, value=30.0, step=0.01, key=f"highcut_x_{name}")
                                preprocessed_x, preprocessing_description_x = data_processor.preprocess_data(data[[field_x]], 'Band-Pass Filter', 
                                                                                  sampling_frequency=sampling_frequency, 
                                                                                  lowcut=lowcut, highcut=highcut)
                            else:
                                preprocessed_x, preprocessing_description_x = data_processor.preprocess_data(data[[field_x]], method_x)
                        
                        # Apply preprocessing on Y-axis
                        if method_y != "None":
                            if method_y == "Smoothing":
                                window_size = slider_with_input_sidebar(f"Smoothing window size for {field_y}", min_value=1, max_value=50, value=5, step=1, key=f"smoothing_y_{name}")
                                preprocessed_y, preprocessing_description_y = data_processor.preprocess_data(data[[field_y]], 'Smoothing', window=window_size)
                            elif method_y == "Band-Pass Filter":
                                sampling_frequency = st.sidebar.number_input(f"Sampling Frequency for Band-Pass (Hz) for {field_y}", value=100, min_value=1, key=f"sampling_frequency_y_{name}")
                                lowcut = slider_with_input_sidebar(f"Low Cutoff Frequency (Hz) for {field_y}", min_value=0.1, max_value=100.0, value=0.5, step=0.01, key=f"lowcut_y_{name}")
                                highcut = slider_with_input_sidebar(f"High Cutoff Frequency (Hz) for {field_y}", min_value=0.1, max_value=100.0, value=30.0, step=0.01, key=f"highcut_y_{name}")
                                preprocessed_y, preprocessing_description_y = data_processor.preprocess_data(data[[field_y]], 'Band-Pass Filter', 
                                                                                  sampling_frequency=sampling_frequency, 
                                                                                  lowcut=lowcut, highcut=highcut)
                            else:
                                preprocessed_y, preprocessing_description_y = data_processor.preprocess_data(data[[field_y]], method_y)

                        # Create scatter plot with specified transparency
                        title = f"{field_x} vs {field_y} ({name})"
                        if method_x != 'None' or method_y != 'None':
                            title += f" - Preprocessing: X({preprocessing_description_x}) Y({preprocessing_description_y})"

                        fig = px.scatter(x=preprocessed_x.squeeze(), y=preprocessed_y.squeeze(), title=title)
                        fig.update_traces(marker=dict(opacity=transparency))
                        fig.update_layout(height=400, width=800, xaxis_title=field_x, yaxis_title=field_y)
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"The file '{name}' does not contain the fields '{field_x}' and/or '{field_y}'.")

            st.sidebar.header("Data Reconstruction")
            show_on_same_figure_reconstruction = st.sidebar.checkbox("Show all reconstructed paths on the same figure", key="reconstruction_checkbox")

            # User input for dynamic column names
            st.sidebar.subheader("Path Reconstruction Columns")
            wheel_angle_column = st.sidebar.text_input("Wheel Angle Column Name", value="wheel_angle")
            speed_column = st.sidebar.text_input("Speed Column Name", value="speed")

            # Check if all dataframes contain the necessary columns
            if all([wheel_angle_column in data.columns and speed_column in data.columns for data in data_dict.values()]):
                st.sidebar.subheader("Conversion Ratio")
                conversion_ratio = slider_with_input_sidebar("Conversion Ratio", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="conversion_ratio")

                if st.sidebar.button("Reconstruct Path"):
                    st.header("Reconstructed Path of the Car")

                    combined_fig = go.Figure() if show_on_same_figure_reconstruction else None
                    paths = {}

                    for idx, (name, data) in enumerate(data_dict.items()):
                        x_path, y_path = path_reconstructor.calculate_path(data[wheel_angle_column] * conversion_ratio, data[speed_column], sampling_frequency_single)
                        paths[name] = (x_path, y_path)

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

                    # Calculate and display the similarity matrix
                    st.header("Similarity Matrix")
                    similarity_matrix = path_reconstructor.calculate_similarity_matrix(paths)
                    st.dataframe(similarity_matrix.style.format("{:.2f}"))
            else:
                st.sidebar.warning(f"Data must contain '{wheel_angle_column}' and '{speed_column}' columns for path reconstruction.")

if __name__ == "__main__":
    main()
