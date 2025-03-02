import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

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
    
    # Generate the nomogram with the original settings
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
    
    # Post-processing to modify the figure
    axes = fig.get_axes()
    
    # Define the variable names as they appear in your nomogram (in order)
    var_names = [
        "Point",
        "HAS-BLED Score", 
        "High-risk Alcohol Consumption", 
        "Oral Anticoagulation Therapy", 
        "Platelet Aggregation Inhibitor Therapy", 
        "Perioperative Bridging Therapy",
        "Overall point",
        "Postoperative Bleeding Risk"
    ]
    
    # Define binary variables
    binary_vars = [
        "High-risk Alcohol Consumption", 
        "Oral Anticoagulation Therapy", 
        "Platelet Aggregation Inhibitor Therapy", 
        "Perioperative Bridging Therapy"
    ]
    
    # Create a custom formatter to handle tick labels
    def format_ticks(x, pos):
        return f"{int(x)}"
    
    # Process each axis based on variable name
    for i, ax in enumerate(axes):
        if i < len(var_names):
            # Get current label and check if it's a binary variable
            current_label = ax.get_ylabel()
            var_name = var_names[i] if i < len(var_names) else current_label
            
            # If this axis represents a binary variable
            if any(binary_name in var_name for binary_name in binary_vars):
                # Get current axis limits
                xmin, xmax = ax.get_xlim()
                
                # Set tick formatter for whole numbers only
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
                
                # Set ticks to only show at start and end (0 and 1)
                ax.set_xticks([xmin, xmax])
                ax.set_xticklabels(['0', '1'])
                
                # Use dash-dot style for binary variables
                ax.grid(False)  # Remove any grid
                
                # Replace the axis line with a dash-dot style
                line = ax.get_lines()
                if line:
                    for ln in line:
                        ln.set_linestyle('-.')
                else:
                    # If no line exists, create one
                    ax.plot([xmin, xmax], [0, 0], 'k-.', linewidth=1.3)
            
            # For HAS-BLED axis, ensure it shows ticks 0-9
            elif "HAS-BLED" in var_name:
                # Get current limits
                xmin, xmax = ax.get_xlim()
                
                # Set ticks 0-9
                ax.set_xticks(np.linspace(xmin, xmax, 10))
                ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                
            # For point axes, ensure they show appropriate ticks
            elif "Point" in var_name or "point" in var_name:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    
    # Add title
    fig.suptitle("Figure 1: Visualization of Nomogram", fontsize=12, style='italic')
    
    # Show the nomogram in Streamlit
    st.pyplot(fig)
