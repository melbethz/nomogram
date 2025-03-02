import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib import gridspec
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import BboxConnector

########################################################
#           1) HELPER FUNCTIONS FOR NOMOGRAM
########################################################

def generate_df_rank(path, total_point=100):
    """Reads Excel (path), returns dataframe + intercept + threshold for the logistic model."""
    df = pd.read_excel(path)
    df.index = df["feature"].values
    
    intercept = df.loc["intercept","coef"]
    threshold = df.loc["threshold","coef"]
    
    df = df.drop(index=["intercept", "threshold"])
    df = df.reset_index(drop=True)
    
    df["sequence"] = list(range(0, df.shape[0]))
    df["range*coef"] = df["coef"] * (df["max"] - df["min"])
    df["abs_range*coef"] = df["range*coef"].abs()
    df = df.sort_values(by="abs_range*coef", ascending=False)

    # The following lines compute how many points each predictor uses.
    # The top predictor (first row) is assigned 'total_point', and subsequent
    # predictors are scaled by the ratio of their absolute effect relative to that top predictor.
    point_factors = df["abs_range*coef"] / df["abs_range*coef"].shift(1)
    point_factors.iloc[0] = 1.0  # First is 1 by definition
    df["point"] = total_point * point_factors.cumprod()
    
    df["negative_coef"] = np.minimum(df["coef"], 0)
    df["positive_coef"] = np.maximum(df["coef"], 0)
    
    # Re-sort back to the original predictor sequence
    df = df.sort_values(by="sequence", ascending=True)
    return df, intercept, threshold


def compute_x(df, lm_intercept, total_point, maxi_point, mini_point=0):
    """Computes probability curve coordinates for the 'Positive Risk' axis at bottom."""
    # Minimum possible linear predictor
    mini_score = (
        sum(df["negative_coef"] * df["max"]) 
        + sum(df["positive_coef"] * df["min"]) 
        + lm_intercept
    )
    # Maximum possible linear predictor
    maxi_score = (
        sum(df["negative_coef"] * df["min"]) 
        + sum(df["positive_coef"] * df["max"]) 
        + lm_intercept
    )
    
    # 'coef' is slope from [mini_score..maxi_score] to [mini_point..maxi_point]
    coef = (maxi_score - mini_score) / maxi_point

    # Probability range from logistic transformation
    score = np.linspace(mini_score, maxi_score, 500)
    prob = 1.0 / (1.0 + np.exp(-score))

    x_point = np.linspace(0, 1, 500)
    label = np.linspace(mini_point, maxi_point, 500)
    
    # Focus on the portion where prob is between 0.01 and 0.99
    flag = (prob <= 0.99) & (prob >= 0.01)

    mini_overallpoint = label[flag][0] // (total_point / 2) * (total_point / 2)
    maxi_overallpoint = (label[flag][-1] // (total_point / 2) + 1) * (total_point / 2)
    
    score = np.linspace(coef * mini_overallpoint + mini_score,
                        coef * maxi_overallpoint + mini_score, 500)
    prob = 1.0 / (1.0 + np.exp(-score))
    return mini_overallpoint, maxi_overallpoint, x_point + 0.02, prob


def set_axis(ax, title, min_point, max_point, xticks, xticklabels, position,
             total_point, type_,
             ax_para={"c": "black", "linewidth": 1, "linestyle": "-"},
             xtick_para={"fontsize": 8, "fontfamily": "Times New Roman", "fontweight": "bold"},
             ylabel_para={"fontsize": 10, "fontname": "Songti Sc", "labelpad":140,
                          "loc": "center", "color": "black", "rotation":"horizontal"}):
    """Draws one horizontal axis for a predictor or the Points scale."""
    ax.set_xlim(0, 1.1)
    
    # Draw the horizontal axis line from min_point to max_point (scaled by total_point)
    ax.axhline(
        y=0.6,
        xmin=(min_point / total_point + 0.02) / 1.1,
        xmax=(max_point / total_point + 0.02) / 1.1,
        **ax_para
    )

    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.NullLocator())

    # Hide spines
    for spine in ["right", "left", "top", "bottom"]:
        ax.spines[spine].set_visible(False)
    
    # Draw tick marks
    for i in range(len(xticks)):
        ax.axvline(x=xticks[i] + 0.02, ymin=0.6 - 0.2, ymax=0.6, **ax_para)
        
        if position[i] == "up":
            ax.annotate(
                xticklabels[i],
                xy=(xticks[i] + 0.02, 0.75),
                horizontalalignment="center",
                **xtick_para
            )
        else:
            ax.annotate(
                xticklabels[i],
                xy=(xticks[i] + 0.02, 0),
                horizontalalignment="center",
                **xtick_para
            )
        
        # Minor ticks for continuous variables
        if i < len(xticks) - 1 and type_ == "continuous":
            gap = abs(xticks[i+1] - xticks[i])
            if gap > 0.07:
                # Insert 5 minor ticks (6 intervals) for large gaps
                for j in np.linspace(xticks[i], xticks[i+1], 6):
                    ax.axvline(x=j + 0.02, ymin=0.6 - 0.1, ymax=0.6, **ax_para)
            elif gap > 0.025:
                # Insert 2 minor ticks (3 intervals) for moderate gaps
                for j in np.linspace(xticks[i], xticks[i+1], 3):
                    ax.axvline(x=j + 0.02, ymin=0.6 - 0.1, ymax=0.6, **ax_para)

    ax.set_ylabel(title, **ylabel_para)


def plot_prob(ax, title, x_point, prob, threshold=None,
              ax_para={"c":"black", "linewidth":1, "linestyle": "-"},
              threshold_para={"c":"g", "linewidth":1, "linestyle": "-."},
              tick_para={"direction": 'in',"length": 3, "width": 1.5,},
              text_para={"fontsize": 10,"fontfamily": "Songti Sc", "fontweight": "bold"},
              xtick_para={"fontsize": 8,"fontfamily": "Times New Roman", "fontweight": "bold"},
              ylabel_para={"fontsize": 10, "fontname": "Songti Sc", "labelpad":100,
                           "loc": "bottom", "color": "black", "rotation":"horizontal"},
              total_point=100):
    """Draws the bottom logistic curve (mapping Overall points -> Probability)."""
    for spine in ["right","top","bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([-0.5, 0.5])
    
    # Plot the logistic curve. We shift it up/down so 0->top,1->bottom, etc.
    ax.plot(x_point, -prob + 0.5, **ax_para)
    
    # If a threshold is defined, draw a horizontal dash line
    if threshold is not None:
        ax.axhline(-threshold + 0.5, xmin=0, xmax=1, **threshold_para)
        ax.text(x=1.05, y=-threshold + 0.5, s=f"threshold={threshold}", **text_para,
                bbox=dict(facecolor='black', alpha=0.2))

    ax.tick_params(**tick_para)
    
    # X ticks from 0..1, but no numeric labels
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels(['','','','','',''])
    
    # Y ticks from 0->1 but inverted in the code
    ax.set_yticks([0.5,0.3,0.1,-0.1,-0.3,-0.5])
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], **xtick_para)
    
    ax.set_ylabel(ylabel=title, **ylabel_para)


def grid_connect(ax1, ax2, xticks, prop_lines={"lw": 0.5, "color": "b", "linestyle": "-."}):
    """Draws dashed vertical lines between the top axis and the current axis for alignment."""
    for x in xticks:
        bbox = Bbox.from_extents(x+0.02, 0.02, x+0.02, 1.02)
        bbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
        bbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())
        c = BboxConnector(bbox1, bbox2, loc1=3, loc2=3, clip_on=False, **prop_lines)
        ax2.add_patch(c)


def generate_xtick(range_, type_, mini, maxi, point, total_point):
    """Helper to decide how many major ticks to draw for a variable axis."""
    ran = point / total_point
    if type_ in ["nominal","ordinal"]:
        # If categorical/ordinal with small integer range
        range_ = int(range_)
        if (ran <= 0.1) and (range_ >= 3):
            xticks = np.linspace(0, point/total_point, 3)
            xticklabels = np.linspace(mini, maxi, 3)
            xticklabels = [int(i) for i in xticklabels]
        else:
            xticks = np.linspace(0, point/total_point, range_+1)
            xticklabels = list(range(int(mini), int(maxi)+1))
    elif (0.1 <= ran < 0.25) or (range_ < 2):
        xticks = np.linspace(0, point/total_point, 5)
        xticklabels = np.linspace(mini, maxi, 5)
    elif ran < 0.1:
        xticks = np.linspace(0, point/total_point, 2)
        xticklabels = np.linspace(mini, maxi, 2)
    elif range_ <= 11:
        # For small integer range
        range_ = int(range_)
        xticks = np.linspace(0, point/total_point, range_+1)
        xticklabels = list(range(int(mini), int(maxi)+1))
    elif range_ <= 150:
        if (range_ % 10 == 0) and (ran/(range_/5+1) < 0.07):
            num = int(range_/10)
            xticks = np.linspace(0, point/total_point, num+1)
            xticklabels = np.linspace(mini, maxi, num+1)
        elif range_ % 5 == 0:
            num = int(range_/5)
            xticks = np.linspace(0, point/total_point, num+1)
            xticklabels = np.linspace(mini, maxi, num+1)
        else:
            xticks = np.linspace(0, point/total_point, 6)
            xticklabels = np.linspace(mini, maxi, 6)
    elif range_ <= 200:
        if range_ % 10 == 0:
            num = int(range_/10)
            xticks = np.linspace(0, point/total_point, num)
            xticklabels = np.linspace(mini, maxi, num)
        else:
            num = int((maxi-mini)//10+1)
            xticks = np.linspace(0, point/total_point, num)
            xticklabels = np.linspace(mini, maxi, num)
    else:
        num = int((maxi-mini)//50+1)
        xticks = np.linspace(0, point/total_point, num)
        xticklabels = [int(i) for i in np.linspace(mini, maxi, num)]
    
    position = ["down" for _ in xticks]
    return list(xticks), list(xticklabels), position


def nomogram(path, result_title="Positive Risk", fig_width=10, single_height=0.45, dpi=100,
             ax_para={"c":"black", "linewidth":1.3, "linestyle": "-"},
             tick_para={"direction": 'in',"length": 3, "width": 1.5,},
             xtick_para={"fontsize": 10, "fontfamily": "Songti Sc", "fontweight": "bold"},
             ylabel_para={"fontsize": 12, "fontname": "Songti Sc", "labelpad":100,
                          "loc": "center", "color": "black", "rotation":"horizontal"},
             total_point=100):
    """
    Draws a nomogram from an Excel file defining logistic model coefficients and variable ranges.
    - path: path to Excel (e.g. "model_2.xlsx")
    - total_point: default max points for top axis
    - fig_width, single_height: figure dimension controls
    """
    df, lm_intercept, threshold = generate_df_rank(path=path, total_point=total_point)
    new = df["feature"].str.split(pat="_", expand=True)

    if new.shape[1] == 1:
        df["main_feature"] = new[0]
    else:
        df["main_feature"] = new[0]
        df["sub_feature"]  = new[1]
        
    group     = df.groupby(["main_feature"], sort=False)
    num_axes  = len(group)
    maxi_point = df["point"].sum()

    mini_overallpoint, maxi_overallpoint, x_point, prob = compute_x(
        df, lm_intercept, total_point, maxi_point
    )

    # We create a figure with 'num_axes + 5' vertical rows
    # to hold the top Points axis, each predictor axis, an "Overall point" axis, and the bottom curve.
    fig = plt.figure(figsize=(fig_width, single_height*(num_axes + 2 + 7)), dpi=dpi)
    gs = gridspec.GridSpec(num_axes + 5, 1)
    
    # 1) Top axis for "Points"
    ax0 = fig.add_subplot(gs[0, :])
    # We'll set tick marks from 0..100 in increments of 10 by default
    top_ticks = np.linspace(0, 1, 11)  # 0..1 in 10 intervals
    top_labels = list(range(0, total_point+1, total_point//10))  # e.g. 0..100 by 10
    top_positions = ["down" for _ in top_ticks]

    set_axis(
        ax=ax0,
        title="Point",
        min_point=0,
        max_point=total_point,
        position=top_positions,
        total_point=total_point,
        xticks=top_ticks,
        xticklabels=top_labels,
        type_='continuous',
        ax_para=ax_para,
        xtick_para=xtick_para,
        ylabel_para=ylabel_para
    )
    
    # 2) Each predictor axis
    features_in_order = df["main_feature"].unique()  # maintains original order
    for i, feat in enumerate(features_in_order, start=1):
        d = group.get_group(feat)
        ax = fig.add_subplot(gs[i, :])

        if d.shape[0] > 1:
            # If multiple rows => nominal/ordinal with sub_features
            min_point = d["point"].min()
            max_point = d["point"].max()
            xticks = [p/total_point for p in d["point"].values]
            xticklabels = [str(x) for x in d["sub_feature"]]
            position = ["down"] * len(xticks)
            # You could get fancy with "up" or "down" if needed.

        else:
            # Single row => continuous or discrete
            maxi = float(d["max"])
            mini = float(d["min"])
            point_val = d["point"].values[0]
            range_ = maxi - mini
            var_type = d["type"].values[0]  # e.g. "continuous", "discrete"
            xticks, xticklabels, position = generate_xtick(
                range_, var_type, mini, maxi, point_val, total_point
            )
            if d["coef"].values[0] < 0:
                xticklabels.reverse()
            # Convert large range to int
            if range_ > 10:
                xticklabels = [int(x) for x in xticklabels]
            else:
                xticklabels = [round(x,2) for x in xticklabels]
            min_point = 0
            max_point = point_val

        # Possibly override line style if it's a discrete or nominal variable
        var_type = list(set(d["type"].values))[0]
        if var_type == "nominal":
            ax_para["linestyle"] = "-."
        elif var_type == "discrete":
            ax_para["linestyle"] = "-."
        else:
            ax_para["linestyle"] = "-"

        set_axis(
            ax=ax,
            title=feat,
            min_point=min_point,
            max_point=max_point,
            xticks=xticks,
            xticklabels=xticklabels,
            position=position,
            total_point=total_point,
            type_=var_type,
            ax_para=ax_para,
            xtick_para=xtick_para,
            ylabel_para=ylabel_para
        )

        # Draw vertical dashed lines connecting top axis (ax0) and this axis
        tick_num = int(np.ceil(max(xticks) / 0.1))
        grid_connect(ax0, ax, xticks=np.linspace(0, 0.1 * tick_num, tick_num+1))
    
    # 3) "Overall point" axis
    ax_overall = fig.add_subplot(gs[num_axes+1, :])
    # Generate tick marks from min..max overall points
    overall_range = maxi_overallpoint - mini_overallpoint
    xticks_over, xticklabels_over, pos_over = generate_xtick(
        overall_range,
        "continuous",
        mini_overallpoint,
        maxi_overallpoint,
        maxi_overallpoint,
        maxi_overallpoint
    )
    ax_para["linestyle"] = "-"
    set_axis(
        ax=ax_overall,
        title="Overall point",
        min_point=0,
        max_point=maxi_overallpoint,
        position=pos_over,
        type_="continuous",
        xticks=xticks_over,
        xticklabels=xticklabels_over,
        total_point=maxi_overallpoint,
        ax_para=ax_para,
        xtick_para=xtick_para,
        ylabel_para=ylabel_para
    )
    
    # 4) Bottom axis with the Probability curve
    ax_prob = fig.add_subplot(gs[num_axes+2:, :])
    ylabel_para["loc"] = "center"  # Probability label in center
    plot_prob(
        ax=ax_prob, title=result_title, x_point=x_point, prob=prob, 
        threshold=threshold, total_point=total_point,
        tick_para=tick_para, ylabel_para=ylabel_para
    )
    ax_prob.grid(color='b', ls='-.', lw=0.25, axis="both")
    
    fig.tight_layout()
    return fig


########################################################
#           2) LOGISTIC MODEL & STREAMLIT APP
########################################################

# === Logistic regression coefficients from your model ===
INTERCEPT = -3.7634
C_HAS     =  0.0284
C_ALCOHOL =  0.9575
C_PAI     =  1.0074
C_OAT     =  0.5272
C_BRIDGE  =  1.0557

def compute_risk(has_bled, alcohol, pai, oat, bridging):
    """Computes predicted probability from your logistic model."""
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
# STREAMLIT UI
########################################
st.title("Nomogram & Postoperative Bleeding Risk Calculator")

st.markdown("""
**Instructions**  
1. Enter the predictor values below.  
2. Click **Generate** to compute the risk **and** see the nomogram.  

**Note**: This app relies on an Excel file (named `"model_2.xlsx"`) that must be formatted for the `nomogram()` function.  
- E.g. each row has: `feature, coef, min, max, type, ...`  
- Must have rows for `"intercept"` and `"threshold"`.  
- Binary variables (0 or 1) typically set `type="discrete"` or `"nominal"`.  
- Continuous variables (e.g. HAS-BLED) set `type="continuous"` with `min=0, max=9`, etc.
""")

# --- A) User inputs ---
has_bled_val = st.number_input("HAS-BLED Score (0–9)", 0, 9, value=3)
alcohol_val  = st.number_input("High-Risk Alcohol (0=No, 1=Yes)", 0, 1, value=0)
pai_val      = st.number_input("Platelet Aggregation Inhibitor (0=No, 1=Yes)", 0, 1, value=0)
oat_val      = st.number_input("Oral Anticoagulation (0=No, 1=Yes)", 0, 1, value=0)
bridge_val   = st.number_input("Perioperative Bridging (0=No, 1=Yes)", 0, 1, value=0)

# --- B) Generate button ---
if st.button("Generate"):
    # 1) Calculate predicted risk
    risk = compute_risk(has_bled_val, alcohol_val, pai_val, oat_val, bridge_val)
    st.write(f"**Predicted Bleeding Risk**: {risk*100:.2f}%")

    # 2) Create the Nomogram figure from your local Excel
    excel_path = "model_2.xlsx"  # adjust as needed
    fig = nomogram(
        path=excel_path,
        result_title="Postoperative Bleeding Risk",
        fig_width=12,       # Wider figure so scales are not cramped
        single_height=1.0,  # More vertical space per axis
        dpi=300,            # Higher resolution
        # Provide your chosen style parameters:
        ax_para={"c":"black","linewidth":1.3,"linestyle":"-."},  # dash-dot for illustration
        tick_para={"direction": "in","length": 3, "width": 1.5},
        xtick_para={"fontsize": 10, "fontfamily": "Arial", "fontweight": "bold"},
        ylabel_para={
            "fontsize": 12, "fontname": "Arial",
            "labelpad": 100, "loc": "center",
            "color": "black", "rotation":"horizontal"
        },
        total_point=400     # If your top axis needs to go up to 400
    )

    # 3) Show the figure in Streamlit
    st.pyplot(fig)
