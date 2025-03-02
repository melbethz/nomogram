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
    Computes predicted probability using your logistic model:
    Probability = 1 / (1 + exp( - (intercept + sum_of_coeffs * X) ))
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
- Enter the predictor values below.
- Click **Generate** to see the standard nomogram image and the predicted risk
  based on the logistic‐regression model.
  
_Note:_ The *simpleNomo* nomogram is static (it does not dynamically place pointers).
You can visually interpret the chart or rely on the computed risk below.
""")

# 1) User inputs (unique `key` for each widget)
has_bled = st.number_input("HAS-BLED Score (0 to 9)",
                           min_value=0, max_value=9, value=3, key="has_bled")
alcohol  = st.number_input("High-Risk Alcohol (0=No, 1=Yes)",
                           min_value=0, max_value=1, value=0, key="alc")
pai      = st.number_input("Platelet Aggregation Inhibitor (0=No, 1=Yes)",
                           min_value=0, max_value=1, value=0, key="pai")
oat      = st.number_input("Oral Anticoagulation (0=No, 1=Yes)",
                           min_value=0, max_value=1, value=0, key="oat")
bridge   = st.number_input("Perioperative Bridging (0=No, 1=Yes)",
                           min_value=0, max_value=1, value=0, key="bridge")

# 2) Button to generate
if st.button("Generate"):
    # (A) Calculate the predicted risk from logistic model
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.write(f"**Predicted Risk:** {risk*100:.2f}%")

    # (B) Generate the nomogram figure from simpleNomo
    excel_path = "model_2.xlsx"  # Update path if needed
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
