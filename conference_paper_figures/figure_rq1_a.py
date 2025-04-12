import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from adjustText import adjust_text  # You may need to install this: pip install adjustText

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

# Load the data
df = pd.read_csv('llm_pt_sdp_f1_harmonic_means.csv', delimiter='\t')

# Get unique values for categorical variables
df['llm'] = df['llm'].str.split('-', n=1).str[0].str.capitalize()
llms = df['llm'].unique()
prompts = df['prompt'].unique()
decision_points = df['ssvc_decision_point'].unique()

# Create abbreviations for prompting techniques using first letter of each word
prompt_abbrev = {}
for prompt in prompts:
    # Split by underscore and take first letter of each word
    words = prompt.split('_')
    abbrev = ''.join([word[0].upper() for word in words])
    prompt_abbrev[prompt] = abbrev

# Filter to keep only the top 5 F1 scores for each LLM
top_df = pd.DataFrame()
for llm in llms:
    llm_data = df[df['llm'] == llm]
    top_5_llm = llm_data.nlargest(5, 'f1_harmonic_mean')
    top_df = pd.concat([top_df, top_5_llm])

# Create a mapping of LLMs to numeric values for x-axis
llm_mapping = {llm: i for i, llm in enumerate(sorted(llms))}
top_df['llm_numeric'] = top_df['llm'].map(llm_mapping)

# Get the unique prompts and decision points that appear in the top 5
top_prompts = top_df['prompt'].unique()
top_decision_points = top_df['ssvc_decision_point'].unique()

# Define marker shapes for prompting techniques
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
prompt_markers = {prompt: markers[i % len(markers)] for i, prompt in enumerate(sorted(prompts))}

# Define a professional color palette for decision points
# Using a colorblind-friendly palette that works well in print
colors = sns.color_palette("viridis", len(decision_points))
decision_colors = {dp: colors[i] for i, dp in enumerate(sorted(decision_points))}

# Create the figure with appropriate dimensions for a journal article
# Typical column width in journals is around 3.5 inches, full page width is around 7 inches
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)  # 7 inches wide, 300 dpi

# Create scatter plot with different colors and shapes
texts = []  # To store text objects for adjust_text
for prompt in sorted(prompts):
    for decision in sorted(decision_points):
        subset = top_df[(top_df['prompt'] == prompt) & (top_df['ssvc_decision_point'] == decision)]
        if not subset.empty:
            # Create the scatter plot
            scatter = ax.scatter(
                subset['llm_numeric'], 
                subset['f1_harmonic_mean'],
                c=[decision_colors[decision]] * len(subset),
                marker=prompt_markers[prompt],
                s=100,  # Slightly larger markers since we have fewer points
                alpha=0.8,  # Higher alpha for better visibility in print
                label=f"{prompt} - {decision}",
                edgecolors='black',  # Add black edge to markers for better definition
                linewidth=0.5
            )
            
            # Add data labels positioned to the left of points
            for i, row in subset.iterrows():
                # Format the F1 score to 2 decimal places
                label_text = f"{row['f1_harmonic_mean']:.2f}"
                
                # Create text object positioned to the left of the point
                text = ax.text(
                    row['llm_numeric'] - 0.15,  # Position to the left of the point
                    row['f1_harmonic_mean'], 
                    label_text,
                    fontsize=7, 
                    ha='right',  # Right-align text so it's closer to the point
                    va='center',  # Center vertically with the point
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                    zorder=5  # Ensure text is above other elements
                )
                texts.append(text)

# Use adjust_text to prevent overlapping labels while maintaining left positioning
adjust_text(
    texts,
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
    expand_points=(1.5, 1.5),
    force_points=(0.1, 0.1),
    x_only=False  # Allow vertical adjustment while trying to maintain horizontal position
)

# Create custom legend elements with abbreviated text for prompting techniques
legend_elements = []

# Add markers for prompting techniques that appear in the top 5
for prompt in sorted(top_prompts):
    abbrev = prompt_abbrev[prompt]
    legend_elements.append(plt.Line2D([0], [0], marker=prompt_markers[prompt], color='w', 
                          markerfacecolor='gray', markersize=8, label=abbrev))

# Add colors for decision points that appear in the top 5 (keeping full names)
for decision in sorted(top_decision_points):
    # Replace underscores with spaces for better readability
    readable_decision = decision.replace('_', ' ')
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=decision_colors[decision], markersize=8, label=readable_decision))

# Create two separate legends with appropriate font sizes
prompt_legend = ax.legend(handles=legend_elements[:len(top_prompts)], 
                         title="Prompting Technique", 
                         loc='upper left', 
                         bbox_to_anchor=(1.01, 1),
                         fontsize=8,
                         title_fontsize=9)
ax.add_artist(prompt_legend)
decision_legend = ax.legend(handles=legend_elements[len(top_prompts):], 
                           title="SSVC Decision Point", 
                           loc='upper left', 
                           bbox_to_anchor=(1.01, 0.5),
                           fontsize=8,
                           title_fontsize=9)

# Set x-axis ticks to LLM names
ax.set_xticks(range(len(llms)))
ax.set_xticklabels([llm.split('/')[-1].split('-', 1)[0].capitalize() for llm in sorted(llms)], 
                   rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelsize=9)

# Add labels and title with appropriate font sizes
ax.set_xlabel('LLM Model', fontsize=10)
ax.set_ylabel('Harmonic Mean of Trial F1-Scores', fontsize=10)
# ax.set_title('Top 5 F1 Scores per LLM by SSVC Decision Point and Prompting Technique', fontsize=11)

# Set y-axis limits to start from 0.5 as requested
y_min = 0.5
y_max = max(top_df['f1_harmonic_mean']) * 1.05  # Slight padding at the top
ax.set_ylim(y_min, y_max)

# Enhanced grid with more gridlines both vertical and horizontal
# Remove existing grid
ax.grid(False)

# Add more detailed grid
# Vertical gridlines at each x-tick and midpoints
x_major_ticks = np.arange(0, len(llms))
x_minor_ticks = np.arange(-0.5, len(llms))
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)

# Horizontal gridlines with smaller intervals
y_major_ticks = np.arange(0.5, y_max, 0.05)  # Major gridlines every 0.05
y_minor_ticks = np.arange(0.5, y_max, 0.01)  # Minor gridlines every 0.01
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)

# Add the enhanced grid
ax.grid(which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.3, color='lightgray', alpha=0.5)

# Adjust layout to make room for the legend
plt.tight_layout()
fig.subplots_adjust(right=0.75)

# Save the figure in multiple formats suitable for LaTeX
# Vector format (PDF) - best for LaTeX
plt.savefig('llm_top5_performance_plot.pdf', bbox_inches='tight', dpi=600)

# High-resolution raster format (PNG)
plt.savefig('llm_top5_performance_plot.png', bbox_inches='tight', dpi=600)

# TIFF format (sometimes required by journals)
plt.savefig('llm_top5_performance_plot.tiff', bbox_inches='tight', dpi=600)

# EPS format (alternative vector format)
plt.savefig('llm_top5_performance_plot.eps', bbox_inches='tight', dpi=600)

print("Professional figure with left-aligned data labels and enhanced grid saved in multiple formats suitable for LaTeX journal articles.")
