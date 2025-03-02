import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
    Computes predicted probability using the logistic model:
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
- Click **Generate** to see the nomogram and the predicted risk.
""")

# 1) User Inputs
has_bled = st.number_input("HAS-BLED Score (0 to 9)", min_value=0, max_value=9, value=3)
alcohol  = st.number_input("High-Risk Alcohol Consumption (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
pai      = st.number_input("Platelet Aggregation Inhibitor Therapy (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
oat      = st.number_input("Oral Anticoagulation Therapy (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
bridge   = st.number_input("Perioperative Bridging Therapy (0=No, 1=Yes)", min_value=0, max_value=1, value=0)

# 2) Button to generate
if st.button("Generate"):
    # (A) Calculate the predicted risk from logistic model
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.write(f"**Predicted Risk:** {risk*100:.2f}%")

    # (B) Generate the corrected nomogram figure
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Define axes and labels
    points_scale = np.arange(0, 110, 10)  # Scale for HAS-BLED (0-100)
    binary_scale = [0, 1]  # Binary variables (0, 1 only)
    total_score_scale = np.arange(0, 450, 50)  # Overall points (0-400)
    
    y_labels = [
        "HAS-BLED Score", "High-risk Alcohol Consumption", "Anticoagulation Therapy"
    ]
    y_positions = np.array([2, 1, 0])  # Top to bottom positioning
    
    # HAS-BLED score line (numerical scale)
    ax.hlines(y=2, xmin=0, xmax=100, color="black", linewidth=1.5)
    ax.set_xticks(np.arange(0, 10, 1), minor=True)  # HAS-BLED subscale
    ax.set_xticklabels(np.arange(0, 10, 1), fontsize=10, fontweight="bold", minor=True)
    
    # Binary variable dashed lines
    for y_pos in y_positions[1:]:  # Skip HAS-BLED row
        ax.hlines(y=y_pos, xmin=0, xmax=100, color="black", linestyle="dashdot", linewidth=1.2)
    
    # Correct 0-1 labels positioning
    for y_pos in y_positions[1:]:  # Skip HAS-BLED row
        ax.text(-5, y_pos, "0", fontsize=12, fontweight="bold", ha="right")
        ax.text(105, y_pos, "1", fontsize=12, fontweight="bold", ha="left")
    
    # Set Y-axis labels correctly
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")
    
    # X-axis settings
    ax.set_xticks(points_scale)
    ax.set_xticklabels(points_scale, fontsize=10, fontweight="bold")
    ax.set_xlim(-10, 110)
    ax.set_ylim(-1, 3)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Display in Streamlit
    st.pyplot(fig)
