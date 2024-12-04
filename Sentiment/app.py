import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Helper function to execute a Jupyter Notebook
def execute_notebook(notebook_path, parameters=None):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Inject parameters if provided
    if parameters:
        nb.cells.insert(0, nbformat.v4.new_code_cell(f"# Injected Parameters\n{parameters}"))

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': './'}})

    # Collect results from the notebook
    output_cells = [cell for cell in nb.cells if cell.cell_type == 'code' and 'outputs' in cell]
    results = []
    for cell in output_cells:
        for output in cell.outputs:
            if 'text' in output:
                results.append(output['text'])
    return results


# Function for automated EDA
def automated_eda(df, target=None):
    st.write("### Dataset Summary")
    st.dataframe(df.describe().transpose())

    st.write("### Column Distributions")
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[column], kde=True, bins=30, color='blue', ax=ax)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    st.write("### Correlation Heatmap")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.write("### Outlier Detection")
    for column in numerical_columns:
        df[f"{column}_zscore"] = zscore(df[column])
        outliers = df[df[f"{column}_zscore"].abs() > 3]
        st.write(f"Outliers in {column}: {len(outliers)}")
        if not outliers.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)

    if target:
        st.write("### Feature Importance")
        X = df.drop(columns=[target])
        y = df[target]
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
                X[col] = LabelEncoder().fit_transform(X[col])
        X = X.select_dtypes(include=['float64', 'int64', 'bool'])
        model = RandomForestRegressor() if pd.api.types.is_numeric_dtype(y) else RandomForestClassifier()
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write(importance)
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.plot(kind='bar', ax=ax)
        st.pyplot(fig)


# Streamlit app
st.title("Business Data and Reviews Analysis")

# Step 1: Select the type of analysis
if "data_type" not in st.session_state:
    st.session_state["data_type"] = None

if st.session_state["data_type"] is None:
    st.write("### Select the type of analysis:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Business Data"):
            st.session_state["data_type"] = "Business Data"
    with col2:
        if st.button("Business Reviews"):
            st.session_state["data_type"] = "Business Reviews"

# Step 2: File uploader
if st.session_state["data_type"]:
    st.write(f"**You selected:** {st.session_state['data_type']}")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Save and process the uploaded file
        input_file_path = f"./{uploaded_file.name}"
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        if st.session_state["data_type"] == "Business Data":
            # Business Data EDA
            target_column = st.text_input("Enter the target column for analysis (optional):")
            automated_eda(df, target=target_column)

        elif st.session_state["data_type"] == "Business Reviews":
            # Business Reviews Analysis
            st.write("## Running Cleaning Process")
            cleaning_results = execute_notebook("cleaning.ipynb", parameters=f"input_file = '{input_file_path}'")
            st.success("✅ Cleaning completed successfully!")
            st.write("\n".join(cleaning_results))

            st.write("### Select Review Analysis Type")
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Most Common Aspects"):
                    st.write("Running sentiment analysis...")
                    sentiment_results = execute_notebook("senntiment_analysis_top.ipynb",
                                                         parameters=f"input_file = '{input_file_path}'")
                    st.success("✅ Sentiment analysis completed successfully!")
                    st.write("\n".join(sentiment_results))
            with col4:
                if st.button("Type 5 Words"):
                    aspects = st.text_input("Enter 5 aspects (comma-separated):")
                    if st.button("Run Analysis"):
                        analysis_results = execute_notebook("Untitled-1.ipynb",
                                                            parameters=f"input_file = '{input_file_path}'\naspects = '{aspects}'")
                        st.success("✅ Custom analysis completed successfully!")
                        st.write("\n".join(analysis_results))
