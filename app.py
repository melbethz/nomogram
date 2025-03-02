import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt

########################################
# Logistic-Regression Coefficients
########################################
INTERCEPT = -3.7634
C_HAS     =  0.0284
C_ALCOHOL =  0.9575
C_PAI     =  1.0074
C_OAT     =  0.5272
C_BRIDGE  =  1.0557

def compute_risk(has_bled, alcohol, pai, oat, bridging):
    """
    Compute the predicted risk using the logistic model:
      probability = 1 / (1 + exp( - (intercept + sum(coeff_i * x_i) ) ))
    """
    y = (INTERCEPT +
         C_HAS     * has_bled +
         C_ALCOHOL * alcohol +
         C_PAI     * pai +
         C_OAT     * oat +
         C_BRIDGE  * bridging)
    return 1.0 / (1.0 + math.exp(-y))

########################################
# Streamlit App Layout
########################################
st.title("Postoperative Bleeding Nomogram & Risk Calculator")

st.markdown("""
**Instructions:**
- Enter the predictor values below.
- Click **Generate** to view:
  - The nomogram (generated from the Excel file).
  - The computed risk from the logistic regression model.

_Note:_ For the nomogram, binary features are declared as "category" in the Excel
so that each line shows exactly two ticks: 0 and 1.
""")

# 1) Get user inputs with unique keys
has_bled = st.number_input(
    "HAS-BLED Score (0 to 9)",
    min_value=0, max_value=9, value=3,
    key="has_bled"
)

alcohol_choice = st.radio(
    "High-Risk Alcohol Consumption?",
    ("No", "Yes"), key="alc_radio"
)
alcohol = 1 if alcohol_choice == "Yes" else 0

pai_choice = st.radio(
    "Platelet Aggregation Inhibitor Therapy?",
    ("No", "Yes"), key="pai_radio"
)
pai = 1 if pai_choice == "Yes" else 0

oat_choice = st.radio(
    "Oral Anticoagulation Therapy?",
    ("No", "Yes"), key="oat_radio"
)
oat = 1 if oat_choice == "Yes" else 0

bridge_choice = st.radio(
    "Perioperative Bridging Therapy?",
    ("No", "Yes"), key="bridge_radio"
)
bridge = 1 if bridge_choice == "Yes" else 0

# 2) Generate results on button click
if st.button("Generate"):
    # (A) Calculate the logistic regression predicted risk
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.write(f"**Predicted Bleeding Risk:** {risk * 100:.2f}%")
    
    # (B) Generate the nomogram using simpleNomo
    excel_path = "model_2.xlsx"  # Make sure model.xlsx is in the same directory
    fig = simpleNomo.nomogram(
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
    
    # Display the nomogram using Streamlit
    st.pyplot(fig)
