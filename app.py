import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.datasets import load_breast_cancer

# PAGE CONFIGURATION


st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# FILE PATHS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "output"
)

RESULTS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.csv"
)

GRAPH_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_performance_comparison.png"
)

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "best_model.pkl"
)

SCALER_FILE = os.path.join(
    OUTPUT_FOLDER,
    "scaler.pkl"
)


# CUSTOM DESIGN


st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🩺 Breast Cancer Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Model Comparison and Patient Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()


# PERFORMANCE COMPARISON GRAPH


st.header("📊 Performance Comparison of Machine Learning Models")

if os.path.exists(GRAPH_FILE):

    st.image(
        GRAPH_FILE,
        use_container_width=True
    )

else:

    st.warning(
        "Performance graph not found. Please run breast_cancer_prediction.py first."
    )


st.divider()



# CONFUSION MATRICES

st.header("🔍 Confusion Matrices")

matrices = [
    (
        "Logistic Regression",
        "logistic_regression_confusion_matrix.png"
    ),
    (
        "Decision Tree",
        "decision_tree_confusion_matrix.png"
    ),
    (
        "Random Forest",
        "random_forest_confusion_matrix.png"
    ),
    (
        "Support Vector Machine",
        "support_vector_machine_confusion_matrix.png"
    )
]

col1, col2 = st.columns(2)

for index, (name, filename) in enumerate(matrices):

    image_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    column = col1 if index % 2 == 0 else col2

    with column:

        st.subheader(name)

        if os.path.exists(image_path):

            st.image(
                image_path,
                use_container_width=True
            )

        else:

            st.warning(
                f"{filename} not found."
            )


st.divider()

# MODEL COMPARISON TABLE


st.header("📋 Model Comparison Results")

if os.path.exists(RESULTS_FILE):

    results_df = pd.read_csv(
        RESULTS_FILE
    )

    # Convert scores to percentage for display
    display_df = results_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    for column in metric_columns:

        display_df[column] = (
            display_df[column] * 100
        )

    # Find the best model
    best_index = results_df["Accuracy"].idxmax()

    def highlight_best(row):

        if row.name == best_index:

            return [
                "background-color: #d9ead3; font-weight: bold"
            ] * len(row)

        return [""] * len(row)

    styled_df = (
        display_df.style
        .apply(
            highlight_best,
            axis=1
        )
        .format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%"
        })
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # Display best model
    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]

    best_accuracy = (
        results_df.loc[
            best_index,
            "Accuracy"
        ] * 100
    )

    st.success(
        f"🏆 Best Performing Model: {best_model_name} | "
        f"Accuracy: {best_accuracy:.2f}%"
    )

else:

    st.warning(
        "Model comparison results not found."
    )


st.divider()


# ============================================================
# PATIENT PREDICTION
# ============================================================

st.header("🔮 Patient Prediction")

st.write(
    "Enter the tumor feature values below. The model will analyze the data and provide a prediction."
)


# Check if model and scaler files exist
if (
    os.path.exists(MODEL_FILE)
    and os.path.exists(SCALER_FILE)
):

    # Load model and scaler
    best_model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Load dataset information
    data = load_breast_cancer()
    feature_names = data.feature_names

    # Create patient input form
    with st.form("patient_prediction_form"):

        st.subheader("Patient Tumor Feature Values")

        st.info(
            "For demonstration, the fields contain average values from the dataset. "
            "You can change the values to test the model."
        )

        input_values = []

        col1, col2, col3 = st.columns(3)

        # Create all 30 input fields
        for i, feature in enumerate(feature_names):

            column = [
                col1,
                col2,
                col3
            ][i % 3]

            with column:

                value = st.number_input(
                    feature,
                    value=float(data.data[:, i].mean()),
                    format="%.4f",
                    key=f"feature_{i}"
                )

                input_values.append(value)

        submitted = st.form_submit_button(
            "🔍 Predict Result",
            use_container_width=True
        )


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    if submitted:

        # Convert the patient data into an array
        patient_data = np.array(
            input_values
        ).reshape(1, -1)

        # Scale the patient data
        patient_data_scaled = scaler.transform(
            patient_data
        )

        # Make prediction
        prediction = best_model.predict(
            patient_data_scaled
        )[0]

        st.divider()
        st.subheader("Prediction Result")


        # ====================================================
        # POSITIVE CASE
        # ====================================================

        # 0 = Malignant / Positive
        if prediction == 0:

            st.markdown(
                """
                <div style="
                    padding: 30px;
                    border: 3px solid red;
                    border-radius: 12px;
                    text-align: center;
                    background-color: #fff0f0;
                ">
                    <h1 style="
                        color: red;
                        font-weight: bold;
                    ">
                        🔴 CASE POSITIVE
                    </h1>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # NEGATIVE CASE
        # ====================================================

        # 1 = Benign / Negative
        else:

            st.markdown(
                """
                <div style="
                    padding: 30px;
                    border: 3px solid green;
                    border-radius: 12px;
                    text-align: center;
                    background-color: #f0fff0;
                ">
                    <h1 style="
                        color: green;
                        font-weight: bold;
                    ">
                        🟢 CASE NEGATIVE
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )


else:

    st.error(
        "Model files not found. Please run breast_cancer_prediction.py first."
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("📌 Project Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        "**Dataset:** Wisconsin Breast Cancer Dataset"
    )

    st.write(
        "**Data Split:** 80% Training / 20% Testing"
    )

    st.write(
        "**Preprocessing:** StandardScaler"
    )


with info_col2:

    st.write(
        "**Models:** Logistic Regression, Decision Tree"
    )

    st.write(
        "**Models:** Random Forest, Support Vector Machine"
    )

    st.write(
        "**Evaluation:** Accuracy, Precision, Recall and F1-Score"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <p style="text-align: center; color: gray;">
        Breast Cancer Prediction Project | Machine Learning Comparative Analysis
    </p>
    """,
    unsafe_allow_html=True
)