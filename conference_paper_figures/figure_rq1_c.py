import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import cprint
import ssvc
import random
from matplotlib.patches import Patch

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

# Function to convert SSVC decisions to ordinal values
def convert_to_ordinal(decisions):
    """
    Convert SSVC decisions to ordinal values.
    
    Ordinal scale:
    - Track: 1 (least severe)
    - Track*: 2
    - Attend: 3
    - Act: 4 (most severe)
    - unknown: 0 (treated as missing/uncertain)
    - error: -1 (treated as invalid)
    
    Returns a numpy array of ordinal values.
    """
    ordinal_map = {
        'Track': 1,
        'Track*': 2,
        'Attend': 3,
        'Act': 4,
        'unknown': 0,
        'error': -1
    }
    
    return np.array([ordinal_map.get(decision, 0) for decision in decisions])

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

# Get a random LLM to extract Vulnrichment data (since it's the same for all LLMs)
random_llm_value = random.choice(high_risk_df['llm'].unique())
mask = high_risk_df['llm'] == random_llm_value
columns_to_select = ['cve_id', 'vulnrichment_ssvc_exploitation', 
                     'vulnrichment_ssvc_automatable', 'vulnrichment_ssvc_technical_impact', 
                     'scenario_system_role_risk']
vulnrichment_high_risk_df = high_risk_df.loc[mask, columns_to_select].reset_index(drop=True)

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

# Create a dictionary to store ground truth decisions by CVE
gt_decisions_by_cve = {}
for i, row in vulnrichment_high_risk_df.iterrows():
    cve = row['cve_id']
    # Make sure i is within the range of vulnrichment_decisions
    if i < len(vulnrichment_decisions):
        gt_decisions_by_cve[cve] = vulnrichment_decisions[i]
    else:
        cprint(f"Warning: Index {i} out of range for vulnrichment_decisions (length: {len(vulnrichment_decisions)})", "yellow")

# Dictionary to store results
results = {
    'LLM': [],
    'Weighted_Cohens_Kappa': [],
    'Unweighted_Cohens_Kappa': [],
    'Sample_Size': [],
    'Valid_Sample_Size': []  # Count excluding error/unknown
}

# Calculate Cohen's Kappa for each LLM vs ground truth
for llm_name in high_risk_df['llm'].unique():
    llm_subset = high_risk_df[high_risk_df['llm'] == llm_name]
    
    # Create paired lists of decisions (LLM and ground truth)
    llm_decisions = []
    gt_decisions = []
    
    for _, row in llm_subset.iterrows():
        cve = row['cve_id']
        if cve in gt_decisions_by_cve:
            llm_decisions.append(row['llm-ssvc-decision'])
            gt_decisions.append(gt_decisions_by_cve[cve])
    
    # Calculate Cohen's Kappa only if we have paired decisions
    if len(llm_decisions) > 0 and len(gt_decisions) > 0:
        try:
            # Convert to ordinal values
            llm_ordinal = convert_to_ordinal(llm_decisions)
            gt_ordinal = convert_to_ordinal(gt_decisions)
            
            # Count valid samples (excluding error/unknown)
            valid_samples = sum(1 for llm, gt in zip(llm_ordinal, gt_ordinal) 
                               if llm > 0 and gt > 0)
            
            # Filter out error/unknown for weighted kappa calculation
            valid_indices = [i for i, (llm, gt) in enumerate(zip(llm_ordinal, gt_ordinal)) 
                            if llm > 0 and gt > 0]
            
            valid_llm_ordinal = [llm_ordinal[i] for i in valid_indices]
            valid_gt_ordinal = [gt_ordinal[i] for i in valid_indices]
            
            # Calculate unweighted Cohen's Kappa (original categorical)
            unweighted_kappa = cohen_kappa_score(gt_decisions, llm_decisions)
            
            # Calculate weighted Cohen's Kappa for ordinal data
            # Only if we have valid samples
            if valid_samples > 0:
                weighted_kappa = cohen_kappa_score(
                    valid_gt_ordinal, 
                    valid_llm_ordinal,
                    weights='linear'  # Linear weights for ordinal data
                )
            else:
                weighted_kappa = np.nan
            
            # Store results
            results['LLM'].append(llm_name)
            results['Weighted_Cohens_Kappa'].append(weighted_kappa)
            results['Unweighted_Cohens_Kappa'].append(unweighted_kappa)
            results['Sample_Size'].append(len(llm_decisions))
            results['Valid_Sample_Size'].append(valid_samples)
            
            # Print results
            cprint(f"Results for {llm_name} vs Ground Truth:", "cyan")
            cprint(f"  Weighted Cohen's Kappa: {weighted_kappa:.4f}", "green")
            cprint(f"  Unweighted Cohen's Kappa: {unweighted_kappa:.4f}", "green")
            cprint(f"  Total sample size: {len(llm_decisions)}", "blue")
            cprint(f"  Valid sample size: {valid_samples}", "blue")
            
        except Exception as e:
            cprint(f"Error calculating Cohen's Kappa for {llm_name}: {e}", "red")
    else:
        cprint(f"No paired decisions found for {llm_name}", "yellow")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by Weighted Kappa score (descending)
if not results_df.empty:
    results_df = results_df.sort_values('Weighted_Cohens_Kappa', ascending=False)
    
    # Save results to CSV
    results_df.to_csv('cohens_kappa_results.csv', index=False)
    
    # Create figure for grouped bar chart
    plt.figure(figsize=(3.5, 2.5), dpi=300)
    
    # Define colors and patterns for the bars
    colors = ['#1f77b4', '#ff7f0e']  # Blue for weighted, orange for unweighted
    patterns = ['/', '\\']  # Different hatch patterns
    
    # Get the LLMs in order
    llms = results_df['LLM'].tolist()
    x = np.arange(len(llms))
    width = 0.35  # Width of the bars
    
    # Create the bars manually for more control
    ax = plt.gca()
    weighted_bars = ax.bar(x - width/2, results_df['Weighted_Cohens_Kappa'], 
                          width, color=colors[0], label='Weighted', edgecolor='black', linewidth=0.5)
    unweighted_bars = ax.bar(x + width/2, results_df['Unweighted_Cohens_Kappa'], 
                            width, color=colors[1], label='Unweighted', edgecolor='black', linewidth=0.5)
    
    # Add patterns to bars
    for bar, pattern in zip([weighted_bars, unweighted_bars], patterns):
        for b in bar:
            b.set_hatch(pattern)
    
    # Set x-axis ticks and labels
    plt.xticks(x, llms, rotation=45, ha='right', fontsize=7)
    
    # Set y-axis label and limits
    plt.ylabel('Cohen\'s κ', fontsize=8)
    plt.ylim(0, 0.3)  # Set y-axis limit to 0.3 as specified
    
    # Format y-axis to not show redundant zeros
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Add data labels on top of the bars
    for bars in [weighted_bars, unweighted_bars]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:  # Only add label if not NaN and positive
                ax.annotate(f'{height:.3f}', 
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 1),  # 1 point vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, rotation=0)
    
    # Create custom legend with colors and patterns
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='Weighted', hatch=patterns[0]),
        Patch(facecolor=colors[1], edgecolor='black', label='Unweighted', hatch=patterns[1])
    ]
    
    # Add the legend with custom elements
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6, frameon=True)
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    # Save the figure in multiple formats suitable for LaTeX
    plt.savefig('dual_cohens_kappa_comparison.pdf', bbox_inches='tight', dpi=600)
    plt.savefig('dual_cohens_kappa_comparison.png', bbox_inches='tight', dpi=600)
    
    cprint(f"Results saved to 'weighted_cohens_kappa_results.csv'", "green")
    cprint(f"Visualizations saved as PDF and PNG files", "green")
else:
    cprint("No results to save or visualize", "red")
