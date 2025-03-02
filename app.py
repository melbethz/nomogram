import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt

########################################
# 1) Logistic Regression Coefficients
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
    linear_term = (
        INTERCEPT
        + C_HAS     * has_bled
        + C_ALCOHOL * alcohol
        + C_PAI     * pai
        + C_OAT     * oat
        + C_BRIDGE  * bridging
    )
    prob = 1.0 / (1.0 + math.exp(-linear_term))
    return prob

########################################
# 2) Streamlit UI
########################################
st.title("Nomogram & Postoperative Bleeding Risk Calculator")

st.markdown("""
**Instructions**  
1. Enter the predictor values below.  
2. Click **Generate** to see:  
   - The predicted bleeding risk from your logistic model.  
   - The static nomogram from `simpleNomo`.  

_Note:_ The nomogram itself does **not** dynamically move pointers, but
the axes for binary variables will now only show **0** and **1**.
""")

# --- A) User inputs ---
has_bled = st.number_input(
    "HAS-BLED Score (0 to 9)",
    min_value=0, max_value=9, value=3, key="has_bled"
)
alcohol = st.number_input(
    "High-Risk Alcohol (0=No, 1=Yes)",
    min_value=0, max_value=1, value=0, key="alc"
)
pai = st.number_input(
    "Platelet Aggregation Inhibitor (0=No, 1=Yes)",
    min_value=0, max_value=1, value=0, key="pai"
)
oat = st.number_input(
    "Oral Anticoagulation (0=No, 1=Yes)",
    min_value=0, max_value=1, value=0, key="oat"
)
bridge = st.number_input(
    "Perioperative Bridging (0=No, 1=Yes)",
    min_value=0, max_value=1, value=0, key="bridge"
)

# --- B) Button to run ---
if st.button("Generate"):
    # 1) Compute predicted risk
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.write(f"**Predicted Risk:** {risk*100:.2f}%")

    # 2) Generate nomogram from your Excel file.
    #    The key is to adjust fig_width + single_height + total_point
    excel_path = "model_2.xlsx"  # your Excel
    fig = simpleNomo.nomogram(
        path=excel_path,
        result_title="Postoperative Bleeding Risk",
        
        # Make the figure a bit wider:
        fig_width=12,
        
        # Increase the vertical space per row:
        single_height=1.0,  # try 1.0, 1.2, or bigger as needed
        
        dpi=300,
        
        # Control the lines on each axis:
        ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"},
        
        # Control the tick marks themselves:
        tick_para={"direction": "in", "length": 3, "width": 1.5},
        
        # Font size/style of X tick labels
        xtick_para={"fontsize": 10, "fontfamily": "Arial", "fontweight": "bold"},
        
        # Y‐axis label style (which is often turned sideways).
        ylabel_para={
            "fontsize": 12,
            "fontname": "Arial",
            "labelpad": 100,
            "loc": "center",
            "color": "black",
            "rotation": "horizontal"
        },
        
        # Raise this if your total points can go above 100:
        total_point=400
    )

    # 3) Show the figure in Streamlit
    st.pyplot(fig)
