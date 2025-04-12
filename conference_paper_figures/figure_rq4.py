import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
import ssvc

# Configure matplotlib for high-quality output
mpl.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded properly
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'  # Serif fonts are common in academic publications
mpl.rcParams['font.serif'] = ['Times New Roman']  # Common journal font
mpl.rcParams['axes.linewidth'] = 0.8  # Slightly thinner lines for a cleaner look
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8

# Set a professional style
plt.style.use('seaborn-v0_8-whitegrid')

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
        print(f"Error in SSVC decision: {e}")
        return 'unknown'

# Load the data
prompt_queries_llm_responses_df = pd.read_csv('prompt_queries_llm_responses.csv', delimiter='\t')

# Define the severity order for SSVC decisions
severity_order = ['error', 'unknown', 'Track', 'Track*', 'Attend', 'Act']
severity_rank = {decision: i for i, decision in enumerate(severity_order)}

# Calculate ground truth for each row
ground_truths = []
for _, row in prompt_queries_llm_responses_df.iterrows():
    # Extract the necessary values for the SSVC decision
    exploitation = row.get('vulnrichment_ssvc_exploitation', 'unknown')
    automatable = row.get('vulnrichment_ssvc_automatable', 'unknown')
    technical_impact = row.get('vulnrichment_ssvc_technical_impact', 'unknown')
    mission_wellbeing = row.get('scenario_system_role_risk', 'unknown')
    
    # Calculate the ground truth decision
    ground_truth = get_ssvc_decision(
        exploitation,
        automatable,
        technical_impact,
        mission_wellbeing
    )
    
    ground_truths.append(ground_truth)

# Add ground truth to the dataframe
prompt_queries_llm_responses_df['ground_truth'] = ground_truths

# Function to categorize the LLM decision compared to ground truth
def categorize_decision(llm_decision, ground_truth):
    # Handle error and unknown cases
    if llm_decision in ['error', 'unknown']:
        return 'Unknown/Error'
    if ground_truth in ['error', 'unknown']:
        return 'Unknown/Error'  # If ground truth is unknown, we can't categorize
    
    # Get severity ranks
    llm_rank = severity_rank.get(llm_decision)
    gt_rank = severity_rank.get(ground_truth)
    
    # If either decision is not in our severity order, return unknown
    if llm_rank is None or gt_rank is None:
        return 'Unknown/Error'
    
    # Categorize based on severity comparison
    if llm_rank == gt_rank:
        return 'Correct'
    elif llm_rank < gt_rank:
        return 'False Negative'
    else:  # llm_rank > gt_rank
        return 'False Positive'

# Create a new dataframe to store the results
results = []

# Process each row in the combined dataframe
for _, row in prompt_queries_llm_responses_df.iterrows():
    # Get the LLM decision
    llm = row['llm']
    llm_decision = row['llm-ssvc-decision']
    
    # Get the ground truth decision
    ground_truth = row['ground_truth']
    
    # Categorize the decision
    category = categorize_decision(llm_decision, ground_truth)
    
    # Store the result
    results.append({
        'llm': llm,
        'category': category
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Count the occurrences of each category for each LLM
category_counts = results_df.groupby(['llm', 'category']).size().reset_index(name='count')

# Create a figure with appropriate dimensions
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Define colors for each category as specified
colors = {
    'Correct': '#2ecc71',       # Green
    'False Positive': '#f1c40f', # Yellow
    'False Negative': '#e74c3c', # Red
    'Unknown/Error': '#95a5a6'  # Grey
}

# Define patterns for each category
patterns = {
    'Correct': '',              # No pattern for correct
    'False Positive': '///',    # Diagonal lines for false positive
    'False Negative': '\\\\\\', # Diagonal lines (other direction) for false negative
    'Unknown/Error': 'xxx'      # Cross-hatch for unknown/error
}

# Define the order of categories for grouping
category_order = ['Correct', 'False Positive', 'False Negative', 'Unknown/Error']

# Get unique LLMs
llms = sorted(category_counts['llm'].unique())

# Set up the x positions for the bars
x = np.arange(len(llms))
width = 0.2  # Width of each bar

# Plot grouped bars for each category
for i, category in enumerate(category_order):
    # Filter data for this category
    cat_data = category_counts[category_counts['category'] == category]
    
    # Create a dictionary mapping LLMs to counts for this category
    llm_to_count = {row['llm']: row['count'] for _, row in cat_data.iterrows()}
    
    # Get counts for each LLM, using 0 if the LLM doesn't have this category
    counts = [llm_to_count.get(llm, 0) for llm in llms]
    
    # Position for this category's bars
    pos = x + (i - 1.5) * width
    
    # Plot bars for this category
    bars = ax.bar(pos, counts, width, label=category, color=colors[category],
                 edgecolor='black', linewidth=0.5)
    
    # Add pattern to bars
    for bar in bars:
        bar.set_hatch(patterns[category])
    
    # Add data labels
    for j, v in enumerate(counts):
        if v > 0:
            ax.text(pos[j], v + 5, str(int(v)), ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='black')

# Set x-ticks and labels with processed names using a list comprehension
ax.set_xticks(x)
ax.set_xticklabels([(llm.split('/')[-1].split('-', 1)[0][0].upper() + llm.split('/')[-1].split('-', 1)[0][1:]) 
                    if '/' in llm and '-' in llm.split('/')[-1] 
                    else (llm.split('/')[-1][0].upper() + llm.split('/')[-1][1:]) if '/' in llm 
                    else (llm.split('-', 1)[0][0].upper() + llm.split('-', 1)[0][1:]) if '-' in llm 
                    else (llm[0].upper() + llm[1:]) for llm in llms], 
                   rotation=45, ha='right')
ax.set_xlabel('LLM Model', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

# Add a legend
ax.legend(title='SSVC Decision Outcome', loc='upper right', fontsize=10)

# Add gridlines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the figure in multiple formats
plt.savefig('llm_decision_accuracy_grouped.pdf', bbox_inches='tight', dpi=600)
plt.savefig('llm_decision_accuracy_grouped.png', bbox_inches='tight', dpi=600)
plt.savefig('llm_decision_accuracy_grouped.tiff', bbox_inches='tight', dpi=600)

print("LLM decision accuracy grouped bar chart saved in multiple formats.")
