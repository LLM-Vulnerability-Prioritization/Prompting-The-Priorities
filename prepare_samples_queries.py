# ========================================================================================
# Prepare samples and prompts for querying to LLMs
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_4_1_Assign_Detail_Levels(df, file_dump_directory):
    cprint("Step_4_1_Assign_Detail_Levels", "magenta")
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Drop columns that are not from VulZoo
    columns_to_drop = [
    'cve_id', 'vulnrichment_date_published',
    'vulnrichment_state', 'vulnrichment_ssvc_exploitation',
    'vulnrichment_ssvc_automatable', 'vulnrichment_ssvc_technical_impact', 'cutoff_status'
    ]

    # Drop the columns
    df_copy = df_copy.drop(columns=columns_to_drop, errors='ignore')
    
    # Replace empty strings and whitespace with NaN
    df_copy = df_copy.replace(r'^\s*$', np.nan, regex=True)
    
    # Count non-null values for each row
    valid_counts = df_copy.notna().sum(axis=1)
    
    # Get total number of columns
    total_columns = df_copy.shape[1]
    
    # Calculate the proportion of valid data
    valid_proportion = valid_counts / total_columns
    df['data_level_proportion'] = valid_proportion
    
    # Define thresholds for categorization
    def determine_data_level(proportion):
        for level, range_dict in kv.data_detail_levels.items():
            if range_dict['lower'] <= proportion <= range_dict['upper']:
                return level
        return "unknown_level"  # Return this if the proportion doesn't fall into any range
    
    # Add the data_level column to the original DataFrame
    df['data_level'] = valid_proportion.apply(determine_data_level)
    
    # Save the results
    output_path = os.path.join(file_dump_directory, "Step_4_1_Assign_Detail_Levels.csv")
    df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    # Print summary statistics
    level_counts = df['data_level'].value_counts()
    print("\nData Level Distribution:")
    print(level_counts)
    print("\nPercentage Distribution:")
    print((level_counts / len(df) * 100).round(2), "%")
        
    return df

# ========================================================================================

def Step_4_2_Sample_Detail_Cutoff(df, total_sample_size, file_dump_directory):
    cprint("Step_4_2_Sample_Detail_Cutoff", "magenta")
    
    # Identify unique classes in the 'data_level' column
    unique_classes = df['data_level'].unique()
    num_classes = len(unique_classes)
    
    # Calculate the number of samples needed for each class
    sample_size_per_class = total_sample_size // num_classes
    
    # Calculate the number of samples needed for each cutoff status within each class
    sample_size_per_cutoff_status = sample_size_per_class // 2
    
    # Create an empty list to store samples
    samples = []
    
    # Sample from each class
    for data_level in unique_classes:
        # Filter the DataFrame for the current class
        class_df = df[df['data_level'] == data_level]
        
        # Sample from "pre_cutoff"
        pre_cutoff_sample = class_df[class_df['cutoff_status'] == 'pre_cutoff'].sample(
            n=sample_size_per_cutoff_status, random_state=42
        )
        
        # Sample from "post_cutoff"
        post_cutoff_sample = class_df[class_df['cutoff_status'] == 'post_cutoff'].sample(
            n=sample_size_per_cutoff_status, random_state=42
        )
        
        # Append the samples to the list
        samples.append(pre_cutoff_sample)
        samples.append(post_cutoff_sample)
    
    # Combine the samples into one DataFrame
    sampled_df = pd.concat(samples, ignore_index=True)
        
    return sampled_df

# ========================================================================================

def Step_4_3_Generate_Queries(df, trials, llms, approach, file_dump_directory):
    cprint("Step_4_3_Generate_Queries", "magenta")
    
    # Create a list to store all expanded records
    expanded_records = []
    
    def format_tag(tag_name, value):
        """
        Helper function to format XML-style tags with comprehensive null checking
        
        Handles:
        - None values
        - numpy NaN
        - pandas NA/NaT
        - Empty strings
        - Whitespace-only strings
        - 'nan' strings (case insensitive)
        """
        # Convert value to string and strip whitespace
        str_value = str(value).strip()
        
        # Check for various null/empty conditions
        is_empty = (
            pd.isna(value) or                     # Handles pandas NA, NaT
            pd.isnull(value) or                   # Handles numpy NaN
            value is None or                      # Handles None
            str_value == '' or                    # Handles empty strings
            str_value.lower() == 'nan' or         # Handles 'nan' strings
            str_value.lower() == 'none' or        # Handles 'none' strings
            str_value.lower() == 'null'           # Handles 'null' strings
        )
        
        if not is_empty:
            return f"<{tag_name}>{str_value}</{tag_name}>"
        return f"<{tag_name}>No Data Available</{tag_name}>"
    
    # Iterate through the original dataframe
    for _, vuln in df.iterrows():
        # Create base record from vulnerability data
        base_record = vuln.to_dict()
        
        # Generate all tag combinations once per vulnerability
        tags = {
            'attackerkb_document_description': format_tag('AttackerKbDescription', vuln['attackerkb_document_description']),
            'attackerkb_mitre_tactics': format_tag('AttackerKbMitreTactics', vuln['attackerkb_mitre_tactics']),
            'attackerkb_timeline': format_tag('AttackerKbTimeline', vuln['attackerkb_timeline']),
            'attackerkb_vulnerable_versions': format_tag('AttackerKbVulnerableVersions', vuln['attackerkb_vulnerable_versions']),
            'attackerkb_vendor_product_names': format_tag('AttackerKbVendorProductNames', vuln['attackerkb_vendor_product_names']),
            'attackerkb_tags': format_tag('AttackerKbTags', vuln['attackerkb_tags']),
                        
            'bugtraq_advisory_email': format_tag('BugtraqAdvisory', vuln['bugtraq_advisory_email']),
            
            'kev_vendor': format_tag('KevVendor', vuln['kev_vendor']),
            'kev_product': format_tag('KevProduct', vuln['kev_product']),
            'kev_vulnerability_name': format_tag('KevVulnerabilityName', vuln['kev_vulnerability_name']),
            'kev_short_description': format_tag('KevShortDescription', vuln['kev_short_description']),
            'kev_required_action': format_tag('KevRequiredAction', vuln['kev_required_action']),
            'kev_known_ransomware_campaign_use': format_tag('KevKnownRansomwareCampaignUse', vuln['kev_known_ransomware_campaign_use']),
            
            'cve_description': format_tag('CveDescription', vuln['cve_description']),
            
            'exploit_description': format_tag('ExploitDescription', vuln['exploit_description']),
            'exploit_content': format_tag('ExploitContent', vuln['exploit_content']),
            
            'fulldis_advisory_email': format_tag('FulldisAdvisoryEmail', vuln['fulldis_advisory_email']),
            
            'linuxvul_descriptions': format_tag('LinuxVulDescriptions', vuln['linuxvul_descriptions']),
            'linuxvul_title': format_tag('LinuxVulTitle', vuln['linuxvul_title']),
            'linuxvul_email_advisory': format_tag('LinuxVulEmailAdvisory', vuln['linuxvul_email_advisory']),
            
            'oss_advisory_email': format_tag('OssAdvisoryEmail', vuln['oss_advisory_email']),
            
            'patch_code': format_tag('PatchCode', vuln['patch_code']),
            
            'zdi_title': format_tag('ZdiTitle', vuln['zdi_title']),
            'zdi_vendors': format_tag('ZdiVendors', vuln['zdi_vendors']),
            'zdi_products': format_tag('ZdiProducts', vuln['zdi_products']),
            'zdi_description': format_tag('ZdiDescription', vuln['zdi_description']),
            'zdi_additional_details': format_tag('ZdiDetails', vuln['zdi_additional_details']),
            'zdi_timeline': format_tag('ZdiTimeline', vuln['zdi_timeline'])
        }
        
        # Create the vulnerability data prompt once
        vulnerability_data_prompt = ''.join(tags.values())
        
        # Generate all combinations
        for app in approach:
            for llm in llms:
                for prompt_technique, prompt_technique_content in prompt_techniques.items():
                    for scenario in system_scenario_role:
                        for t in range(trials):
                            # Create a new record for this combination
                            new_record = base_record.copy()
                            
                            # Add the combination-specific fields
                            new_record.update({
                                'system_role_prompt': app["system_role"] + " " + scenario['scenario_description'],
                                'user_role_prompt': app['output_instruction'] + " " + prompt_technique_content.format(
                                    app['query_instruction'], 
                                    vulnerability_data_prompt,
                                    app["response_examples"]["act"],
                                    app["response_examples"]["track"]
                                ),
                                'scenario_system_role_risk': scenario['scenario_risk'],
                                'llm': llm['name'],
                                'llm_context_window': llm['context_window'],
                                'llm_cutoff_date': llm['cutoff_date'],
                                'prompt': prompt_technique,
                                'trial_number': t,
                                'llm-query-processed': 0
                            })
                            
                            expanded_records.append(new_record)
    
    # Create new DataFrame from all records
    expanded_df = pd.DataFrame(expanded_records)
    expanded_df = expanded_df.replace(r'^\s*$', np.nan, regex=True)  # Convert empty strings to NaN
    expanded_df = expanded_df.dropna(axis=1, how='all')
    
    # Save the results
    output_path = os.path.join(file_dump_directory, "Step_4_3_Generate_Queries.csv")
    expanded_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    try:
        # Your main function logic here
        return expanded_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del expanded_df
        gc.collect()