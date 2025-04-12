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

# Set a professional style with minimal gridlines
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid", {'grid.linestyle': ':'})

# Load the data
df = pd.read_csv('llm_pt_sdp_f1_harmonic_means.csv', delimiter='\t')

# Get unique values for categorical variables
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

# Create a mapping of decision points to numeric values for x-axis
decision_mapping = {dp: i for i, dp in enumerate(sorted(decision_points))}
df['decision_numeric'] = df['ssvc_decision_point'].map(decision_mapping)

# Filter to keep only the top 5 F1 scores for each SSVC decision point
top_df = pd.DataFrame()
for decision in decision_points:
    decision_data = df[df['ssvc_decision_point'] == decision]
    # Get top 5 combinations of LLM and prompting technique for this decision point
    top_5_decision = decision_data.nlargest(5, 'f1_harmonic_mean')
    top_df = pd.concat([top_df, top_5_decision])

# Get the unique LLMs and prompting techniques that appear in the top 5
top_llms = top_df['llm'].unique()
top_prompts = top_df['prompt'].unique()

# Define marker shapes for LLMs - using a variety of distinct shapes
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
llm_markers = {llm: markers[i % len(markers)] for i, llm in enumerate(sorted(top_llms))}

# Define a professional color palette for prompting techniques
# Using a colorblind-friendly palette from seaborn
distinct_colors = [
    '#FF4500',  # Orange Red
    '#1E90FF',  # Dodger Blue
    '#32CD32',  # Lime Green
    '#FF1493',  # Deep Pink
    '#FFD700',  # Gold
    '#8A2BE2',  # Blue Violet
    '#00FFFF',  # Cyan
    '#FF8C00',  # Dark Orange
    '#008000',  # Green
    '#E6194B',  # Red
    '#000000',  # Black
]

# Convert to RGB format for matplotlib (if needed)
distinct_colors_rgb = []
for color in distinct_colors:
    # Convert hex to RGB
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    distinct_colors_rgb.append((r, g, b))

prompt_colors = {prompt: distinct_colors_rgb[i % len(distinct_colors_rgb)] for i, prompt in enumerate(sorted(top_prompts))}

# Create the figure with appropriate dimensions
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Add jitter to x-coordinates to prevent overlapping
np.random.seed(42)  # For reproducibility
jitter_amount = 0.15  # Controls the amount of jitter

# Create scatter plot with different colors, shapes, and jitter
texts = []  # To store text objects for adjust_text
for llm in sorted(top_llms):
    for prompt in sorted(top_prompts):
        subset = top_df[(top_df['llm'] == llm) & (top_df['prompt'] == prompt)]
        if not subset.empty:
            # Apply jitter to x-coordinates - store as a regular list to avoid indexing issues
            jittered_x = list(subset['decision_numeric'] + np.random.uniform(-jitter_amount, jitter_amount, len(subset)))
            
            # Plot points
            scatter = ax.scatter(
                jittered_x, 
                subset['f1_harmonic_mean'],
                c=[prompt_colors[prompt]] * len(subset),
                marker=llm_markers[llm],
                s=100,  # Slightly larger markers for better visibility
                alpha=0.8,
                label=f"{llm.split('/')[-1]} - {prompt_abbrev[prompt]}",
                edgecolors='black',  # Add black edge to markers for better definition
                linewidth=0.5,
                zorder=2  # Ensure points are above grid lines
            )
            
            # Add data value labels for each point
            for i, (_, row) in enumerate(subset.iterrows()):
                # Format the F1 score to 2 decimal places
                label_text = f"{row['f1_harmonic_mean']:.2f}"
                
                # Create text object with a white background for better visibility
                # Position to the top-left of the point
                text = ax.text(
                    jittered_x[i] - 0.05,  # Position slightly to the left of the point
                    row['f1_harmonic_mean'] + 0.01,  # Position slightly above the point
                    label_text,
                    fontsize=8,
                    ha='right',  # Right-align text (end of text aligns with position)
                    va='bottom',  # Bottom-align text (bottom of text aligns with position)
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                    zorder=5  # Ensure text is above other elements
                )
                texts.append(text)

# Use adjust_text to prevent overlapping labels
adjust_text(
    texts,
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
    expand_points=(1.5, 1.5),
    force_points=(0.1, 0.1)
)

# Create custom legend elements for LLMs and prompting techniques
legend_elements = []

# Add markers for LLMs that appear in the top 5
for llm in sorted(top_llms):
    # Use just the model name (after the last slash) for cleaner labels
    model_name = llm.split('/')[-1]
    
    # Extract the first word before the first hyphen
    if '-' in model_name:
        model_name = model_name.split('-', 1)[0]
    
    # Capitalize the first letter
    if model_name:
        model_name = model_name[0].upper() + model_name[1:]
    
    legend_elements.append(plt.Line2D([0], [0], marker=llm_markers[llm], color='w', 
                          markerfacecolor='gray', markersize=8, label=model_name))

# Add colors for prompting techniques that appear in the top 5
for prompt in sorted(top_prompts):
    # Use abbreviation for cleaner labels
    abbrev = prompt_abbrev[prompt]
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=prompt_colors[prompt], markersize=8, label=abbrev))

# Create two separate legends with appropriate font sizes
llm_legend = ax.legend(handles=legend_elements[:len(top_llms)], 
                     title="LLM Model", 
                     loc='upper left', 
                     bbox_to_anchor=(1.01, 1),
                     fontsize=8,
                     title_fontsize=9)
ax.add_artist(llm_legend)
prompt_legend = ax.legend(handles=legend_elements[len(top_llms):], 
                         title="Prompting Technique", 
                         loc='upper left', 
                         bbox_to_anchor=(1.01, 0.5),
                         fontsize=8,
                         title_fontsize=9)

# Set x-axis ticks to SSVC decision point names with diagonal labels
ax.set_xticks(range(len(decision_points)))
# Replace underscores with spaces for better readability
readable_decisions = [dp.replace('_', ' ') for dp in sorted(decision_points)]
ax.set_xticklabels(readable_decisions, 
                   rotation=45,  # Diagonal labels
                   ha='right',   # Horizontal alignment
                   fontsize=9)

# Set y-axis limits to start from 0.5 for better focus on the data range
y_min = 0.35
y_max = max(top_df['f1_harmonic_mean']) * 1.05  # Slight padding at the top
ax.set_ylim(y_min, y_max)

# Add labels with appropriate font sizes
ax.set_xlabel('SSVC Decision Point', fontsize=10)
ax.set_ylabel('F1 Score', fontsize=10)

# Customize grid for better readability - horizontal lines only, lighter color
ax.grid(True, axis='y', linestyle=':', alpha=0.7, color='lightgray', zorder=1)
ax.grid(False, axis='x')  # Remove vertical grid lines

# Adjust layout to make room for the legends
plt.tight_layout()
fig.subplots_adjust(right=0.75)  # Only adjust right margin for legends

# Save the figure in multiple formats suitable for publications
plt.savefig('decision_top5_performance_plot.pdf', bbox_inches='tight', dpi=600)
plt.savefig('decision_top5_performance_plot.png', bbox_inches='tight', dpi=600)
plt.savefig('decision_top5_performance_plot.tiff', bbox_inches='tight', dpi=600)
plt.savefig('decision_top5_performance_plot.eps', bbox_inches='tight', dpi=600)

print("Professional figure showing top 5 LLM and prompting technique combinations per SSVC decision point with top-left aligned data labels saved in multiple formats.")
