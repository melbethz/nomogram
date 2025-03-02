import math
import streamlit as st
import simpleNomo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Coefficients ---
INTERCEPT = -3.7634
C_HAS = 0.0284
C_ALCOHOL = 0.9575
C_PAI = 1.0074
C_OAT = 0.5272
C_BRIDGE = 1.0557

def compute_risk(has_bled, alcohol, pai, oat, bridging):
    y = (INTERCEPT + C_HAS * has_bled + C_ALCOHOL * alcohol +
         C_PAI * pai + C_OAT * oat + C_BRIDGE * bridging)
    return 1 / (1 + math.exp(-y))

# --- Streamlit App ---
st.title("Postoperative Bleeding Nomogram & Risk Calculator")

st.markdown("""
**Instructions:**
- Enter predictor values
- Click **Generate** for nomogram and calculated risk
""")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    has_bled = st.number_input("HAS-BLED Score (0-9)", 0, 9, 3)
    alcohol = st.number_input("High-risk Alcohol (0=No, 1=Yes)", 0, 1, 0)
with col2:
    pai = st.number_input("Platelet Aggregation Inhibitor (0=No, 1=Yes)", 0, 1, 0)
    oat = st.number_input("Oral Anticoagulation (0=No, 1=Yes)", 0, 1, 0)
    bridge = st.number_input("Perioperative Bridging (0=No, 1=Yes)", 0, 1, 0)

if st.button("Generate"):
    risk = compute_risk(has_bled, alcohol, pai, oat, bridge)
    st.markdown(f"**Predicted Risk:** {risk*100:.1f}%")

    # Generate nomogram
    fig = simpleNomo.nomogram(
        path="model_2.xlsx",
        result_title="Positive Risk",
        fig_width=9,
        single_height=0.6,
        dpi=100,
        ax_para={"c": "black", "linewidth": 1.5},
        tick_para={"length": 4, "width": 1.5},
        xtick_para={
            "fontsize": 12,
            "fontfamily": "Arial",
            "fontweight": "bold"
        },
        ylabel_para={
            "fontsize": 12,
            "fontname": "Arial",
            "labelpad": 25,
            "rotation": 0,
            "va": "center"
        },
        total_point=100
    )

    # Format axes to integers
    for ax in fig.axes:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.tick_params(axis='x', which='major', pad=10)

    # Add section titles
    fig.text(0.08, 0.93, "Point", fontsize=14, weight='bold', ha='center')
    fig.text(0.92, 0.93, "Positive Risk", fontsize=14, weight='bold', ha='center')

    st.pyplot(fig)
