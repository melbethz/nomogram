import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt

########################################
# Logistic-Regression Coefficients
# (Replace with your own if needed)
########################################
INTERCEPT = -3.7634
C_HAS     =  0.0284
C_ALCOHOL =  0.9575
C_PAI     =  1.0074
C_OAT     =  0.5272
C_BRIDGE  =  1.0557

def compute_risk(has_bled, alcohol, pai, oat, bridging):
    """
    Logistic model:
        Probability = 1 / (1 + exp( - [intercept + sum(coeff * X)] ))
    """
    y = (INTERCEPT
         + C_HAS     * has_bled
         + C_ALCOHOL * alcohol
         + C_PAI     * pai
         + C_OAT     * oat
         + C_BRIDGE  * bridging)
    prob = 1.0 / (1.0 + math.exp(-y))
    return prob

########################################
# Streamlit App
########################################
st.title("Postoperative Bleeding Nomogram & Risk Calculator")

st.markdown("""
**Instructions:**
- Enter each predictor below.
- Click **Generate** to see:
  1. The standard nomogram (drawn from your Excel).
  2. The computed risk from your logistic model.

_Note:_ The *simpleNomo* chart is static. It won’t dynamically mark your input,
but you can interpret the 0/1 categories or the numeric scale for HAS-BLED.
""")

# 1) Gather user inputs (use unique `key=` to avoid ID conflicts):
has_bled = st.number_input(
    "HAS-BLED Score (0 to 9)",
    min_value=0, max_value=9, value=3, key="has_bled"
)

alcohol_choice = st.radio(
    "High-Risk Alcohol Consumption?", ("No", "Yes"), key="alc_radio"
)
alcohol = 1 if alcohol_choice == "Yes" else 0

pai_choice = st.radio(
    "Platelet Aggregation Inhibitor Therapy?", ("No", "Yes"), key="pai_radio"
)
pai = 1 if pai_choice == "Yes" else 0

oat_choice = st.radio(
    "Oral Anticoagulation Therapy?", ("No", "Yes"), key="oat_radio"
)
oat = 1 if oat_choice == "Yes" else 0

bridge_choice = st.radio(
    "Perioperative Bridging Therapy?", ("No", "Yes"), key="bridge_radio"
)
bridge = 1 if bridge_choice == "Yes" else 0

# 2) When user clicks "Generate", show risk & nomogram
if st.button("Generate"):
    # (A) Calculate logistic predicted risk
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.write(f"**Predicted Risk:** {risk*100:.2f}%")

    # (B) Generate the nomogram from your Excel file
    excel_path = "model_2.xlsx"  # must match your file name
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

    # Show the nomogram in Streamlit
    st.pyplot(fig)
