# ========================================================================================
# Evaluate LLM performance, both overall, and for detail and cutoff categories
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_7_1_Evaluate_LLM_Performance(df, file_dump_directory):
    cprint("Step_7_1_Evaluate_LLM_Performance", "magenta")

    def calculate_metrics(y_true, y_pred):
        """Calculate all required metrics for a single pair of true/predicted values."""
        # Convert all values to lowercase for consistency
        y_true = y_true.str.lower()
        y_pred = y_pred.str.lower()
        
        # Get unique classes from ground truth only
        true_classes = np.unique(y_true)
        
        # Get all unique values including predictions
        all_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        print(f"Ground truth classes: {true_classes}")
        print(f"All classes (including predictions): {all_classes}")
        
        # Encode labels to numeric values
        le = LabelEncoder()
        # Fit encoder on all possible classes (including 'unknown')
        le.fit(all_classes)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        
        # Calculate probabilities for AUC-PR (one-hot encoding)
        y_true_binary = pd.get_dummies(y_true)
        y_pred_binary = pd.get_dummies(y_pred)
        
        # Ensure both have same columns
        all_classes_set = set(y_true_binary.columns) | set(y_pred_binary.columns)
        for col in all_classes_set:
            if col not in y_true_binary.columns:
                y_true_binary[col] = 0
            if col not in y_pred_binary.columns:
                y_pred_binary[col] = 0
        
        # Align columns
        y_true_binary = y_true_binary[sorted(all_classes_set)]
        y_pred_binary = y_pred_binary[sorted(all_classes_set)]
        
        # Calculate AUC-PR carefully
        try:
            # Calculate average precision for each class and take mean
            class_auc_prs = []
            for col in y_true_binary.columns:
                if y_true_binary[col].sum() > 0:  # Only if there are positive samples
                    class_auc_pr = average_precision_score(y_true_binary[col], 
                                                        y_pred_binary[col])
                    class_auc_prs.append(class_auc_pr)
            
            # If we have any valid AUC-PR scores, take their mean
            if class_auc_prs:
                auc_pr = np.mean(class_auc_prs)
            else:
                auc_pr = 0.0  # Default value when no valid AUC-PR can be calculated
        except Exception as e:
            print(f"Error calculating AUC-PR: {str(e)}")
            auc_pr = 0.0
        
        # Calculate metrics with multiclass support
        metrics = {
            'precision': precision_score(y_true_encoded, y_pred_encoded, 
                                    average='weighted', zero_division=0),
            'recall': recall_score(y_true_encoded, y_pred_encoded, 
                                average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true_encoded, y_pred_encoded),
            'f1': f1_score(y_true_encoded, y_pred_encoded, 
                        average='weighted', zero_division=0),
            'f2': fbeta_score(y_true_encoded, y_pred_encoded, beta=2, 
                            average='weighted', zero_division=0),
            'auc_pr': auc_pr,
            'mcc': matthews_corrcoef(y_true_encoded, y_pred_encoded)
        }
        
        # Add additional information
        metrics['n_classes_true'] = len(true_classes)
        metrics['n_classes_total'] = len(all_classes)
        metrics['unknown_predictions'] = (y_pred == 'unknown').sum()
        metrics['unknown_ratio'] = (y_pred == 'unknown').mean()
        
        # Add class distribution information
        for cls in all_classes:
            metrics[f'true_{cls}_count'] = (y_true == cls).sum()
            metrics[f'pred_{cls}_count'] = (y_pred == cls).sum()
        
        return metrics

    def calculate_harmonic_mean(group, metric_columns):
        """Calculate harmonic mean for a group of metrics, handling zeros appropriately."""
        results = {}
        for col in metric_columns:
            # Replace 0 with NaN to avoid division by zero in harmonic mean
            values = group[col].replace(0, np.nan)
            # Calculate harmonic mean only for non-NaN values
            if values.notna().any():
                harmonic_mean = len(values) / (1/values).sum()
                results[f'{col}_harmonic_mean'] = harmonic_mean
            else:
                results[f'{col}_harmonic_mean'] = 0
        return pd.Series(results)

    # Map fields to SSVC Decision Point
    column_pairs = {
        'Exploitation': ('vulnrichment_ssvc_exploitation', 'llm-Exploitation'),
        'Automatable': ('vulnrichment_ssvc_automatable', 'llm-Automatable'),
        'Technical_Impact': ('vulnrichment_ssvc_technical_impact', 'llm-Technical_Impact'),
        'Mission_Wellbeing': ('scenario_system_role_risk', 'llm-Mission_Wellbeing')
    }
    
    # Initialize list to store results
    results = []
    
    # Group by specified columns
    grouped = df.groupby(['llm', 'prompt', 'trial_number'])
    
    # Calculate metrics for each group
    for (llm, prompt, trial), group in grouped:
        for aspect, (true_col, pred_col) in column_pairs.items():
            try:
                # Skip if either column has all NaN values
                if group[true_col].isna().all() or group[pred_col].isna().all():
                    print(f"Skipping {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial} - All NaN values")
                    continue
                
                # Remove rows where either column has NaN
                valid_mask = ~(group[true_col].isna() | group[pred_col].isna())
                if valid_mask.sum() == 0:
                    print(f"Skipping {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial} - No valid pairs")
                    continue
                
                # Calculate metrics for this aspect
                metrics = calculate_metrics(
                    group[true_col][valid_mask],
                    group[pred_col][valid_mask]
                )
                
                # Add group identifiers and aspect
                result_row = {
                    'llm': llm,
                    'prompt': prompt,
                    'trial_number': trial,
                    'aspect': aspect,
                    **metrics
                }
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error processing {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial}")
                print(f"Error message: {str(e)}")
                continue
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(results)
    
    base_columns = ['llm', 'prompt', 'trial_number', 'aspect', 
                    'precision', 'recall', 'accuracy', 'f1', 
                    'f2', 'auc_pr', 'mcc']
        
    metrics_df = metrics_df[base_columns]

    # Calculate harmonic means across trials
    metric_columns = ['precision', 'recall', 'accuracy', 'f1', 'f2', 'auc_pr', 'mcc']
    
    # Group by llm, prompt, and aspect to calculate harmonic means
    harmonic_means_df = metrics_df.groupby(['llm', 'prompt', 'aspect']).apply(
        calculate_harmonic_mean, metric_columns=metric_columns
    ).reset_index()
    
    # Save harmonic means
    harmonic_means_path = os.path.join(file_dump_directory, "Step_7_1_Evaluate_LLM_Performance.csv")
    harmonic_means_df.to_csv(harmonic_means_path, sep='\t', index=False, encoding='utf-8')

    return harmonic_means_df

# ========================================================================================

def Step_7_2_Evaluate_Detail_Cutoff_Performance(df, file_dump_directory):
    cprint("Step_7_2_Evaluate_Detail_Cutoff_Performance", "magenta")

    def calculate_metrics(y_true, y_pred):
        """Calculate all required metrics for a single pair of true/predicted values."""
        # Convert all values to lowercase for consistency
        y_true = y_true.str.lower()
        y_pred = y_pred.str.lower()
        
        # Get unique classes from ground truth only
        true_classes = np.unique(y_true)
        
        # Get all unique values including predictions
        all_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        print(f"Ground truth classes: {true_classes}")
        print(f"All classes (including predictions): {all_classes}")
        
        # Encode labels to numeric values
        le = LabelEncoder()
        # Fit encoder on all possible classes (including 'unknown')
        le.fit(all_classes)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        
        # Calculate probabilities for AUC-PR (one-hot encoding)
        y_true_binary = pd.get_dummies(y_true)
        y_pred_binary = pd.get_dummies(y_pred)
        
        # Ensure both have same columns
        all_classes_set = set(y_true_binary.columns) | set(y_pred_binary.columns)
        for col in all_classes_set:
            if col not in y_true_binary.columns:
                y_true_binary[col] = 0
            if col not in y_pred_binary.columns:
                y_pred_binary[col] = 0
        
        # Align columns
        y_true_binary = y_true_binary[sorted(all_classes_set)]
        y_pred_binary = y_pred_binary[sorted(all_classes_set)]
        
        # Calculate AUC-PR carefully
        try:
            # Calculate average precision for each class and take mean
            class_auc_prs = []
            for col in y_true_binary.columns:
                if y_true_binary[col].sum() > 0:  # Only if there are positive samples
                    class_auc_pr = average_precision_score(y_true_binary[col], 
                                                         y_pred_binary[col])
                    class_auc_prs.append(class_auc_pr)
            
            # If we have any valid AUC-PR scores, take their mean
            if class_auc_prs:
                auc_pr = np.mean(class_auc_prs)
            else:
                auc_pr = 0.0  # Default value when no valid AUC-PR can be calculated
        except Exception as e:
            print(f"Error calculating AUC-PR: {str(e)}")
            auc_pr = 0.0
        
        # Calculate metrics with multiclass support
        metrics = {
            'precision': precision_score(y_true_encoded, y_pred_encoded, 
                                       average='weighted', zero_division=0),
            'recall': recall_score(y_true_encoded, y_pred_encoded, 
                                 average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true_encoded, y_pred_encoded),
            'f1': f1_score(y_true_encoded, y_pred_encoded, 
                          average='weighted', zero_division=0),
            'f2': fbeta_score(y_true_encoded, y_pred_encoded, beta=2, 
                             average='weighted', zero_division=0),
            'auc_pr': auc_pr,
            'mcc': matthews_corrcoef(y_true_encoded, y_pred_encoded)
        }
        
        # Add additional information
        metrics['n_classes_true'] = len(true_classes)
        metrics['n_classes_total'] = len(all_classes)
        metrics['unknown_predictions'] = (y_pred == 'unknown').sum()
        metrics['unknown_ratio'] = (y_pred == 'unknown').mean()
        
        # Add class distribution information
        for cls in all_classes:
            metrics[f'true_{cls}_count'] = (y_true == cls).sum()
            metrics[f'pred_{cls}_count'] = (y_pred == cls).sum()
        
        return metrics


    def calculate_harmonic_mean(group, metric_columns):
        """Calculate harmonic mean for a group of metrics, handling zeros and negative values appropriately."""
        results = {}
        for col in metric_columns:
            values = group[col]
            if col == 'mcc':  # Special handling for MCC which can be negative
                # For MCC, use arithmetic mean instead of harmonic mean
                mean_value = values.mean()
                results[f'{col}_harmonic_mean'] = mean_value
            else:
                # Replace 0 with NaN to avoid division by zero in harmonic mean
                values = values.replace(0, np.nan)
                # Calculate harmonic mean only for non-NaN values
                if values.notna().any():
                    harmonic_mean = len(values) / (1/values).sum()
                    results[f'{col}_harmonic_mean'] = harmonic_mean
                else:
                    results[f'{col}_harmonic_mean'] = 0
        return pd.Series(results)



    def evaluate_detail_cutoff_performance(df, file_dump_directory):
        # Define column pairs (ground truth vs LLM prediction)
        column_pairs = {
            'Exploitation': ('vulnrichment_ssvc_exploitation', 'llm-Exploitation'),
            'Automatable': ('vulnrichment_ssvc_automatable', 'llm-Automatable'),
            'Technical_Impact': ('vulnrichment_ssvc_technical_impact', 'llm-Technical_Impact'),
            'Mission_Wellbeing': ('scenario_system_role_risk', 'llm-Mission_Wellbeing')
        }
        
        # Initialize list to store results
        results = []
        
        # Group by specified columns
        grouped = df.groupby(['llm', 'prompt', 'trial_number'])
        
        # Calculate metrics for each group
        for (llm, prompt, trial), group in grouped:
            for aspect, (true_col, pred_col) in column_pairs.items():
                try:
                    # Skip if either column has all NaN values
                    if group[true_col].isna().all() or group[pred_col].isna().all():
                        print(f"Skipping {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial} - All NaN values")
                        continue
                    
                    # Remove rows where either column has NaN
                    valid_mask = ~(group[true_col].isna() | group[pred_col].isna())
                    if valid_mask.sum() == 0:
                        print(f"Skipping {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial} - No valid pairs")
                        continue
                    
                    # Calculate metrics for this aspect
                    metrics = calculate_metrics(
                        group[true_col][valid_mask],
                        group[pred_col][valid_mask]
                    )
                    
                    # Add group identifiers and aspect
                    result_row = {
                        'llm': llm,
                        'prompt': prompt,
                        'trial_number': trial,
                        'aspect': aspect,
                        **metrics
                    }
                    
                    results.append(result_row)
                    
                except Exception as e:
                    print(f"Error processing {aspect} for LLM: {llm}, Prompt: {prompt}, Trial: {trial}")
                    print(f"Error message: {str(e)}")
                    continue
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(results)
                
        base_columns = ['llm', 'prompt', 'trial_number', 'aspect', 
                        'precision', 'recall', 'accuracy', 'f1', 
                        'f2', 'auc_pr', 'mcc']
                
        metrics_df = metrics_df[base_columns]

        # Calculate harmonic means across trials
        metric_columns = ['precision', 'recall', 'accuracy', 'f1', 'f2', 'auc_pr', 'mcc']
        
        # Group by llm, prompt, and aspect to calculate harmonic means
        harmonic_means_df = metrics_df.groupby(['llm', 'prompt', 'aspect']).apply(
            calculate_harmonic_mean, metric_columns=metric_columns
        ).reset_index()

        return harmonic_means_df

    # Create a folder to capture CSVs of dataframes from each step
    current_time = datetime.now().strftime("%Y%m%d-%H%M") 
    folder_name = f"file_dumps_{current_time}"
    os.makedirs(folder_name, exist_ok=True)

    def analyze_dataframe_by_columns(df, file_dump_directory):
        # Analysis for cutoff_status
        cutoff_results = {}
        for status in df['cutoff_status'].unique():
            filtered_df = df[df['cutoff_status'] == status]
            result_df = evaluate_detail_cutoff_performance(
                filtered_df, 
                file_dump_directory
            )
            cutoff_results[f'results_cutoff_{status}'] = result_df
        
        # Analysis for data_level
        level_results = {}
        for level in df['data_level'].unique():
            filtered_df = df[df['data_level'] == level]
            result_df = evaluate_detail_cutoff_performance(
                filtered_df, 
                file_dump_directory
            )
            level_results[f'results_level_{level}'] = result_df
        
        return cutoff_results, level_results
    
    cutoff_results, level_results = analyze_dataframe_by_columns(df, folder_name)

    # Access harmonic mean results
    pre_cutoff_performance = cutoff_results['results_cutoff_pre_cutoff'][1]
    post_cutoff_performance = cutoff_results['results_cutoff_post_cutoff'][1]
    high_detail_performance = level_results['results_level_high_detail'][1]
    medium_detail_performance = level_results['results_level_medium_detail'][1]
    low_detail_performance = level_results['results_level_low_detail'][1]

    # Export pre-cutoff results
    pre_cutoff_performance.to_csv(
        'Step_7_2_Pre_Cutoff_Performance.tsv', 
        sep='\t',
        index=False
    )

    # Export post-cutoff results
    post_cutoff_performance.to_csv(
        'Step_7_2_Post_Cutoff_Performance.tsv', 
        sep='\t',
        index=False
    )

    # Export high detail results
    high_detail_performance.to_csv(
        'Step_7_2_High_Detail_Performance.tsv', 
        sep='\t',
        index=False
    )
    
    # Export high detail results
    medium_detail_performance.to_csv(
        'Step_7_2_Medium_Detail_Performance.tsv', 
        sep='\t',
        index=False
    )

    # Export low detail results
    low_detail_performance.to_csv(
        'Step_7_2_Low_Detail_Performance.tsv', 
        sep='\t',
        index=False
    )
    
    return pre_cutoff_performance, post_cutoff_performance, high_detail_performance, medium_detail_performance, low_detail_performance


