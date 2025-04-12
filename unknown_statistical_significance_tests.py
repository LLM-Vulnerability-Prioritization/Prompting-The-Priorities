# ========================================================================================
# Run statistical significance tests between responses including and excluding Unknown
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# Compare statistical significance of results with and without unknown responses
def compare_metrics(df1, df2):
    # Merge dataframes on common columns
    merged = df1.merge(df2, 
                      on=['llm', 'prompt', 'aspect'],
                      suffixes=('_with_unknown', '_without_unknown'))
    
    # For each metric column
    metrics = ['f1_harmonic_mean', 'mcc_harmonic_mean']
    results = {}
    
    for metric in metrics:
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(
            merged[f'{metric}_with_unknown'],
            merged[f'{metric}_without_unknown']
        )
        results[metric] = {'t_stat': t_stat, 'p_value': p_value}
    
    return results