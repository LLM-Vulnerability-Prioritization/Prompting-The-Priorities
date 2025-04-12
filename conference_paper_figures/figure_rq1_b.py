import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
from termcolor import cprint
import ssvc
import random

# Function to get SSVC decision
def get_ssvc_decision(e: str, a: str, ti: str, mw: str) -> str:
    """Get SSVC decision if all values are known."""
    if any(val == 'unknown' for val in (e, a, ti, mw)):
        return 'unknown'
    
    if any(val == 'error' for val in (e, a, ti, mw)):
        return 'error'
    
    try:
        decision = ssvc.Decision(
            exploitation=e.lower(),
            automatable=a.lower(),
            technical_impact=ti.lower(),
            mission_wellbeing=mw.lower(),
        )
        return str(decision.evaluate().action.value)
    except Exception as e:
        cprint(f"Error in SSVC decision: {e}", "red")
        return 'unknown'

# Reset all rcParams to defaults
plt.rcParams.update(plt.rcParamsDefault)

# Explicitly disable LaTeX
plt.rcParams['text.usetex'] = False

# Set a professional style
plt.style.use('seaborn-v0_8-whitegrid')

# Load the data
prompt_queries_llm_responses_df = pd.read_csv('prompt_queries_llm_responses.csv', delimiter='\t')

# Filter to include only high risk scenarios
high_risk_df = prompt_queries_llm_responses_df[prompt_queries_llm_responses_df['scenario_system_role_risk'] == 'high']

# Option 3: Alternative syntax that some find more readable
random_llm_value = random.choice(high_risk_df['llm'].unique())
mask = high_risk_df['llm'] == random_llm_value
columns_to_select = ['cve_id', 'vulnrichment_ssvc_exploitation', 
                     'vulnrichment_ssvc_automatable', 'vulnrichment_ssvc_technical_impact', 
                     'scenario_system_role_risk']
vulnrichment_high_risk_df = high_risk_df.loc[mask, columns_to_select]

# Calculate Vulnrichment ground truth SSVC decisions
vulnrichment_decisions = []
for _, row in vulnrichment_high_risk_df.iterrows():
    decision = get_ssvc_decision(
        row['vulnrichment_ssvc_exploitation'],
        row['vulnrichment_ssvc_automatable'],
        row['vulnrichment_ssvc_technical_impact'],
        row['scenario_system_role_risk']
    )
    vulnrichment_decisions.append(decision)

# Create a Series of Vulnrichment decisions
vulnrichment_decision_series = pd.Series(vulnrichment_decisions)

# Get the count of each Vulnrichment decision
vulnrichment_counts = vulnrichment_decision_series.value_counts()

# Create a cross-tabulation of LLM vs SSVC decisions for high risk scenarios
llm_cross_tab = pd.crosstab(high_risk_df['llm'], high_risk_df['llm-ssvc-decision'])

# Create a DataFrame for Vulnrichment ground truth decisions
vulnrichment_df = pd.DataFrame(index=['Vulnrichment'], columns=llm_cross_tab.columns)
vulnrichment_df = vulnrichment_df.fillna(0)  # Fill with zeros initially

# Fill in the Vulnrichment counts
for decision in vulnrichment_counts.index:
    if decision in vulnrichment_df.columns:
        vulnrichment_df.at['Vulnrichment', decision] = vulnrichment_counts[decision]

# Combine LLM decisions with Vulnrichment
combined_cross_tab = pd.concat([llm_cross_tab, vulnrichment_df])

# Define the custom order for SSVC decisions (from bottom to top in stacked bar)
decision_order = ['error', 'unknown', 'Track', 'Track*', 'Attend', 'Act']

# Ensure all categories exist in the DataFrame, add with zeros if missing
for decision in decision_order:
    if decision not in combined_cross_tab.columns:
        combined_cross_tab[decision] = 0

# Reorder the columns according to the specified order
combined_cross_tab = combined_cross_tab[decision_order]

# Calculate row totals for percentage calculations
row_totals = combined_cross_tab.sum(axis=1)

# Define custom colors for each decision
colors = {
    'error': 'black',
    'unknown': 'grey',
    'Track': 'green',
    'Track*': 'yellow',
    'Attend': 'orange',
    'Act': 'red'
}

# Define patterns for each decision (for black and white printing)
patterns = {
    'error': '/',
    'unknown': '\\',
    'Track': 'o',
    'Track*': '+',
    'Attend': 'x',
    'Act': '*'
}

# Create the figure with appropriate dimensions for a single column
fig, ax = plt.subplots(figsize=(3.5, 7.0), dpi=300)

# Create the stacked bar chart with custom colors
bottom = np.zeros(len(combined_cross_tab))

# Plot each decision category as a separate bar segment with pattern
for decision in decision_order:
    if decision in combined_cross_tab.columns:
        bars = ax.bar(combined_cross_tab.index, combined_cross_tab[decision], 
                      bottom=bottom, label=decision, color=colors[decision],
                      edgecolor='black', linewidth=0.5)
        
        # Add pattern to bars
        for bar in bars:
            bar.set_hatch(patterns[decision])
            
        # Update the bottom for the next series
        bottom += np.array(combined_cross_tab[decision])

# Add labels and title with appropriate font sizes
# ax.set_title('SSVC Decisions: LLMs vs Vulnrichment\n(Mission & Wellbeing High Risk Scenarios)', fontsize=10)
ax.set_xlabel('', fontsize=9)  # No x-label needed
ax.set_ylabel('Count', fontsize=9)

# Rotate x-tick labels and make them smaller
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# Create custom legend elements with colors and patterns
legend_elements = []
for decision in decision_order:
    if decision in combined_cross_tab.columns and combined_cross_tab[decision].sum() > 0:
        element = Patch(facecolor=colors[decision], edgecolor='black',
                        label=decision, hatch=patterns[decision])
        legend_elements.append(element)

# Add the legend below the plot with more space and more columns
ax.legend(handles=legend_elements, title='SSVC Decision', fontsize=7, 
          title_fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), 
          ncol=3)

# Add data labels to each segment with white background boxes and percentages
# Only show labels for segments that represent > 1% of the total
# Position labels slightly to the right of center
for i, decision in enumerate(decision_order):
    if decision in combined_cross_tab.columns:
        for j, value in enumerate(combined_cross_tab[decision]):
            if value > 0:  # Only add labels for non-zero values
                # Calculate the vertical position (middle of the segment)
                height = combined_cross_tab[decision][j]
                # Calculate the bottom position for this segment
                bottom_pos = 0
                for k in range(i):
                    if decision_order[k] in combined_cross_tab.columns:
                        bottom_pos += combined_cross_tab[decision_order[k]][j]
                
                # Calculate percentage
                percentage = (value / row_totals[j]) * 100
                
                # Only show labels for segments that represent > 1% of the total
                if percentage > 1.0:
                    # Position label in the middle of each segment, slightly to the right
                    y_pos = bottom_pos + height/2
                    x_pos = j + 0.15  # Slightly to the right of center
                    
                    # Add white background box behind text with count and percentage
                    ax.text(x_pos, y_pos, f"{int(value)}\n({percentage:.1f}%)", 
                           ha='left', va='center',  # Left-aligned for right-of-center positioning
                           fontsize=6, color='black',
                           fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', 
                                     boxstyle='round,pad=0.3', linewidth=0.5))

# Adjust the bottom margin to prevent overlap between x-labels and legend
plt.subplots_adjust(bottom=0.45)

# Ensure the layout is tight but with enough space for labels and legend
plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# Save the figure in multiple formats suitable for LaTeX
plt.savefig('ssvc_decisions_comparison_high_risk.pdf', bbox_inches='tight', dpi=600)
plt.savefig('ssvc_decisions_comparison_high_risk.png', bbox_inches='tight', dpi=600)

print("Professional vertical bar chart comparing LLM decisions to Vulnrichment ground truth saved.")
