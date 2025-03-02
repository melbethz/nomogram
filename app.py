import streamlit as st
import math
import simpleNomo
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. App Title & Description
# -----------------------------------------------------------------------------
st.title("Nomogram for Postoperative Bleeding Risk")

st.write("""
This interactive app calculates the risk of postoperative bleeding based on a logistic regression model.
Enter your predictor values below and see both the calculated risk and the nomogram.
""")

# -----------------------------------------------------------------------------
# 2. User Inputs
# -----------------------------------------------------------------------------
# Use unique keys for each widget to avoid duplicate element ID errors.
has_bled = st.number_input(
    "HAS-BLED Score (0 to 9)",
    min_value=0,
    max_value=9,
    value=3,
    key="has_bled"
)

alcohol = st.selectbox(
    "High-Risk Alcohol Consumption?",
    ["No", "Yes"],
    key="alcohol_key"
)

platelet = st.selectbox(
    "Platelet Aggregation Inhibitor Therapy?",
    ["No", "Yes"],
    key="platelet_key"
)

oral_ant = st.selectbox(
    "Oral Anticoagulation Therapy?",
    ["No", "Yes"],
    key="oral_ant_key"
)

bridge = st.selectbox(
    "Perioperative Bridging Therapy?",
    ["No", "Yes"],
    key="bridge_key"
)

# -----------------------------------------------------------------------------
# 3. Risk Calculation
# -----------------------------------------------------------------------------
# Coefficients from your logistic regression model:
INTERCEPT = -3.7634
B_HAS     =  0.0284
B_ALC     =  0.9575
B_PAI     =  1.0074
B_OAC     =  0.5272
B_BRDG    =  1.0557

# Convert selectbox outputs to numeric (0 or 1)
x_alc  = 1 if alcohol == "Yes" else 0
x_pai  = 1 if platelet == "Yes" else 0
x_oac  = 1 if oral_ant == "Yes" else 0
x_brdg = 1 if bridge == "Yes" else 0

# Compute the log-odds
log_odds = (INTERCEPT +
            B_HAS  * has_bled +
            B_ALC  * x_alc +
            B_PAI  * x_pai +
            B_OAC  * x_oac +
            B_BRDG * x_brdg)

# Convert log-odds to a probability
predicted_risk = 1.0 / (1.0 + math.exp(-log_odds))
predicted_percent = predicted_risk * 100

st.markdown(f"### Predicted Bleeding Risk: **{predicted_percent:.2f}%**")

# -----------------------------------------------------------------------------
# 4. Display the Nomogram
# -----------------------------------------------------------------------------
# Ensure that "model.xlsx" is placed in your repository directory.
excel_path = "model.xlsx"

# Generate the nomogram figure using simpleNomo
nomo_fig = simpleNomo.nomogram(
    path=excel_path,
    result_title="Postoperative Bleeding Risk",
    fig_width=10,
    single_height=0.45,
    dpi=300,
    ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"},
    tick_para={"direction": "in", "length": 3, "width": 1.5},
    xtick_para={"fontsize": 10, "fontfamily": "Arial", "fontweight": "bold"},
    ylabel_para={
        "fontsize": 12,
        "fontname": "Arial",
        "labelpad": 100,
        "loc": "center",
        "color": "black",
        "rotation": "horizontal"
    },
    total_point=100
)

# Use Streamlit to display the matplotlib figure
st.pyplot(nomo_fig)
