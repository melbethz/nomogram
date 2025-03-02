import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Option to add customization to how the variables are displayed
    # Check the simpleNomo documentation for specifics
    # This is a direct approach modification
    try:
        # First attempt: Try to modify the Excel file directly (if supported)
        import pandas as pd
        # Read the Excel file
        model_df = pd.read_excel(excel_path)
        
        # If the Excel structure allows, modify binary variable parameters
        # This is a placeholder - actual structure depends on simpleNomo's Excel format
        binary_vars = ["High-risk Alcohol Consumption", "Oral Anticoagulation Therapy", 
                      "Platelet Aggregation Inhibitor Therapy", "Perioperative Bridging Therapy"]
        
        # Save a modified version for this run
        modified_excel = "temp_model.xlsx"
        model_df.to_excel(modified_excel, index=False)
        excel_path = modified_excel
    except:
        # If modifying Excel fails, continue with original file
        pass
    
    # Generate the nomogram
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
    
    # Post-processing the figure to fix binary variables
    axes = fig.get_axes()
    
    # Try a simpler approach by identifying axes based on their position in the figure
    # Most nomograms arrange variables in order
    # Assuming HAS-BLED is first, then binary variables
    
    # Find axes that should be binary (usually after the first variable)
    # This assumes a specific order - adjust indices if needed
    binary_axes_indices = list(range(1, 5))  # Adjust based on your actual nomogram structure
    
    for i, idx in enumerate(binary_axes_indices):
        if idx < len(axes):
            ax = axes[idx]
            
            # Get the current x range
            xmin, xmax = ax.get_xlim()
            
            # Set only 0 and 1 ticks
            ax.set_xticks([xmin, xmax])
            ax.set_xticklabels(['0', '1'])
            
            # Make sure the line is drawn completely
            ax.plot([xmin, xmax], [0, 0], 'k-', linewidth=1.3)
            
            # Ensure y-axis labels don't get rotated
            for label in ax.get_yticklabels():
                label.set_rotation(0)
    
    # Add a threshold line for risk if needed
    risk_ax = axes[-1] if len(axes) > 5 else None
    if risk_ax:
        # Add a horizontal line at 0.5 threshold
        y_pos = 0.5
        risk_ax.axhline(y=y_pos, color='green', linestyle='--', alpha=0.7)
        risk_ax.text(0.95, y_pos, f"threshold={y_pos}", 
                   verticalalignment='bottom', horizontalalignment='right',
                   transform=risk_ax.transData, fontsize=9,
                   bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Show the nomogram in Streamlit
    st.pyplot(fig)
