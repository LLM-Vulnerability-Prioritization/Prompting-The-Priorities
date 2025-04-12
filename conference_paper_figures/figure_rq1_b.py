import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
from termcolor import cprint
import ssvc
import random
import re

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

# Function to process LLM names
def process_llm_name(llm_name):
    if '/' in llm_name:
        # Extract content between / and the next -
        parts = llm_name.split('/')
        if len(parts) > 1:
            after_slash = parts[-1]
            if '-' in after_slash:
                result = after_slash.split('-', 1)[0]
            else:
                result = after_slash
            # Capitalize the first letter
            return result[0].upper() + result[1:] if result else result
    
    # If no /, just return the original name
    return llm_name

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

# Process LLM names in the DataFrame
high_risk_df['llm'] = high_risk_df['llm'].apply(process_llm_name)

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

# Convert counts to proportions
combined_cross_tab_pct = combined_cross_tab.div(row_totals, axis=0) * 100

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

# Capitalize the first letter of decision names for the legend
capitalized_decision_names = {decision: decision[0].upper() + decision[1:] if decision not in ['error', 'unknown'] else decision for decision in decision_order}

# Create the figure with 2:1 width-to-height ratio
fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)

# Create the stacked bar chart with custom colors
bottom = np.zeros(len(combined_cross_tab_pct))

# Plot each decision category as a separate bar segment with pattern
for decision in decision_order:
    if decision in combined_cross_tab_pct.columns:
        bars = ax.bar(combined_cross_tab_pct.index, combined_cross_tab_pct[decision], 
                      bottom=bottom, label=capitalized_decision_names[decision], color=colors[decision],
                      edgecolor='black', linewidth=0.5)
        
        # Add pattern to bars
        for bar in bars:
            bar.set_hatch(patterns[decision])
            
        # Update the bottom for the next series
        bottom += np.array(combined_cross_tab_pct[decision])

# Add labels and title with appropriate font sizes
ax.set_xlabel('', fontsize=9)  # No x-label needed
ax.set_ylabel('Proportion (%)', fontsize=9)

# Rotate x-tick labels and make them smaller
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# Create custom legend elements with colors and patterns
# Only include decisions that have at least 1% in any category
legend_elements = [Patch(facecolor=colors[d], edgecolor='black', 
                         label=d.capitalize(), hatch=patterns[d]) 
                  for d in decision_order 
                  if d in combined_cross_tab_pct.columns and (combined_cross_tab_pct[d] > 1.0).any()]

# Add the legend below the plot with more space and more columns
ax.legend(handles=legend_elements, title='SSVC Decision', fontsize=7, 
          title_fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), 
          ncol=3)

# Add data labels to each segment with alternating left/right positions within each stacked bar
for j in range(len(combined_cross_tab_pct.index)):  # Loop through columns (stacked bars)
    # Track which segments in this column get labels
    labeled_segments = []
    
    # First pass: identify which segments will get labels (>1%)
    for i, decision in enumerate(decision_order):
        if decision in combined_cross_tab_pct.columns:
            value = combined_cross_tab_pct[decision][j]
            if value > 1.0:  # Only segments > 1% get labels
                labeled_segments.append(i)
    
    # Second pass: add the labels with alternating positions
    for idx, segment_idx in enumerate(labeled_segments):
        decision = decision_order[segment_idx]
        value = combined_cross_tab_pct[decision][j]
        
        # Calculate the vertical position (middle of the segment)
        height = value
        # Calculate the bottom position for this segment
        bottom_pos = 0
        for k in range(segment_idx):
            if decision_order[k] in combined_cross_tab_pct.columns:
                bottom_pos += combined_cross_tab_pct[decision_order[k]][j]
        
        # Determine position based on segment index in the labeled segments list
        if idx % 2 == 0:  # Even indices go right
            x_pos = j + 0.15  # Slightly to the right
            ha = 'left'
        else:  # Odd indices go left
            x_pos = j - 0.15  # Slightly to the left
            ha = 'right'
        
        # Position label in the middle of each segment
        y_pos = bottom_pos + height/2
        
        # Get the original count value
        count_value = combined_cross_tab[decision][j]
        
        # Add white background box behind text with count and percentage
        ax.text(x_pos, y_pos, f"{int(count_value)}\n({value:.1f}%)", 
               ha=ha, va='center',  # Alignment based on position
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

print("Professional chart with all requested modifications saved.")
