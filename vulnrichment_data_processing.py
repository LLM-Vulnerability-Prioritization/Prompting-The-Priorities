# ========================================================================================
# Import VULNRICHMENT and extract the SSVC Decision Point values for Ground Truth
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_1_1_Import_Vulnrichment(input_directory, file_dump_directory):
    cprint("Step_1_1_Import_Vulnrichment","magenta")

    # Initialize an empty list to store the rows of data
    vulnrichment_dictionary: list[dict] = []
    
    # Traverse directory for JSON Files
    def import_json_files(input_directory: str):
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        process_json_file(file_content)
    
    # Load data from JSON files
    def process_json_file(file_content: str):
        
        try:
            # Attempt to load the JSON content
            data = json.loads(file_content)
            # If the data is a list of JSON objects, process each object
            if isinstance(data, list):
                for obj in data:
                    record = parse_json_file(obj)
                    vulnrichment_dictionary.append(record)
            else:
                record = parse_json_file(data)
                vulnrichment_dictionary.append(record)
        except json.JSONDecodeError:
            print("Error decoding JSON from file content")
    
    def get_cvss_vector(version: str, data_str: str) -> str:
        pattern = rf'{version}.*?"vectorString":\s*"([^"]*)"' 
        match = re.search(pattern, data_str)
        return match.group(1) if match else ''
    
    # Parse data from JSON files
    def parse_json_file(data: dict) -> dict:
        
        # Section where CVSS String Vectors are located
        metrics = data.get('containers', {}).get('cna', {}).get('metrics', [])
        cvss_dict = {key: value for d in metrics for key, value in d.items()}
        # Extract SSVC metrics - Check multiple possible patterns 
        adp_containers = data.get('containers', {}).get('adp', []) 
        ssvc_other_dict = {} 
        
        cvssv2_0 = ''
        cvssv3_0 = ''
        cvssv3_1 = ''
        cvssv4_0 = ''
        
        for container in adp_containers: 
            metrics_list = container.get('metrics', []) 
            for metric in metrics_list: 
                # Handle SSVC data if present in 'other' key 
                other_content = metric.get('other', {}).get('content', {}) 
                if 'options' in other_content: 
                    ssvc_options = other_content['options'] 
                    ssvc_other_dict.update({key: value for d in ssvc_options for key, value in d.items()}) 
                
                else:
                    cvssv2_0 = metric.get('cvssV2_0', {}).get('vectorString', '')
                    cvssv3_0 = metric.get('cvssV3_0', {}).get('vectorString', '')
                    cvssv3_1 = metric.get('cvssV3_1', {}).get('vectorString', '')
                    cvssv4_0 = metric.get('cvssV4_0', {}).get('vectorString', '')
                                        
        # Extract vendor, product and cpe data  
        # Initialize lists to collect the extracted data 
        product = '' 
        vendor = ''
        cpes = ''
        # Loop through each element in the 'adp' list within 'containers' 
        for adp_item in data.get('containers', {}).get('adp', []): 
            # Extract product and vendor details 
            affected = adp_item.get('affected', None) or data.get('containers', {}).get('cna', {}).get('affected', [{}])
            for affect_item in affected: 
                product = affect_item.get('product', '') 
                vendor = affect_item.get('vendor', '') 
                cpes = '|'.join([cpe for item in affected for cpe in item.get('cpes', [])])
                        
        record = { 
            # CVE ID from Vulnrichment 
            'cve_id': data.get('cveMetadata', {}).get('cveId', ''), 
            'vulnrichment_date_published': data.get('cveMetadata', {}).get('datePublished', ''), 
            'vulnrichment_state': data.get('cveMetadata', {}).get('state', ''), 
            
            # Vulnerability details 
            'vulnrichment_product': product, 
            'vulnrichment_vendor': vendor, 
            'vulnrichment_cpes': cpes, 
            'vulnrichment_description': next((desc.get('value') for desc in data.get('containers', {}).get('cna', {}).get('descriptions', []) if desc.get('lang') == 'en'), ''), 
            'vulnrichment_cwe_description': next((desc.get('description') for desc in data.get('containers', {}).get('cna', {}).get('problemTypes', [{}])[0].get('descriptions', []) if desc.get('lang') == 'en' and desc.get('type') == 'CWE'), ''), 
            'vulnrichment_kev_date_added': data.get('containers', {}).get('adp', [{}])[0].get('metrics', [{}])[0].get('other', {}).get('content', {}).get('dateAdded', ''), 
            
            # Assessment details 
            'vulnrichment_cvss31': cvss_dict.get('cvssV3_1', {}).get('vectorString', '') or cvssv3_1, 
            'vulnrichment_cvss30': cvss_dict.get('cvssV3_0', {}).get('vectorString', '') or cvssv3_0, 
            'vulnrichment_cvss20': cvss_dict.get('cvssV2_0', {}).get('vectorString', '') or cvssv2_0, 
            'vulnrichment_cvss40': cvss_dict.get('cvssV4_0', {}).get('vectorString', '') or cvssv4_0,  
            
            # SSVC Grount Truth
            'vulnrichment_ssvc_exploitation': ssvc_other_dict.get('Exploitation', ''), 
            'vulnrichment_ssvc_automatable': ssvc_other_dict.get('Automatable', ''), 
            'vulnrichment_ssvc_technical_impact': ssvc_other_dict.get('Technical Impact', ''),             
            } 
        
        return record        
        
    # Run and cast to Data Frame
    import_json_files(input_directory)
    df = pd.DataFrame(vulnrichment_dictionary)
        
    # Export to CSV
    output_path = os.path.join(file_dump_directory, "Step_1_1_Import_Vulnrichment.csv")
    df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    return df

def Step_1_2_Clean_Vulnrichment(df: pd.DataFrame, file_dump_directory: str) -> pd.DataFrame:
    cprint("Step_1_2_Clean_Vulnrichment","magenta")
    
    # Remove pipes used for joins in Step_1_1
    def clean_pipe_characters(value): 
        if isinstance(value, str):
            value = value.strip('|') 
            value = re.sub(r'\|+', '|', value) 
        return value
    
    cleaned_df = df.applymap(clean_pipe_characters)
    cleaned_df['vulnrichment_ssvc_exp_aut_techimp_comb'] = (
        cleaned_df["vulnrichment_ssvc_exploitation"] + "-" +
        cleaned_df["vulnrichment_ssvc_automatable"] + "-" +
        cleaned_df["vulnrichment_ssvc_technical_impact"]
    )

    cleaned_df = cleaned_df[[
        'cve_id',
        'vulnrichment_date_published',
        'vulnrichment_state',
        'vulnrichment_ssvc_exploitation',
        'vulnrichment_ssvc_automatable',
        'vulnrichment_ssvc_technical_impact',
    ]]
    
    # Convert cutoff dates to datetime objects
    cutoff_dates = [datetime.strptime(item['cutoff_date'], '%Y-%m-%d') for item in kv.llms]
    min_date = min(cutoff_dates).replace(tzinfo=pytz.UTC)
    max_date = max(cutoff_dates).replace(tzinfo=pytz.UTC)
    
    # Convert 'vulnrichment_date_published' to datetime with flexible parsing
    cleaned_df['vulnrichment_date_published'] = pd.to_datetime(
        cleaned_df['vulnrichment_date_published'],
        format='mixed',  # Allow mixed formats
        utc=True        # Ensure UTC timezone
    )
    
    # Filter out records where 'vulnrichment_date_published' is between min_date and max_date
    filtered_df = cleaned_df[
        (cleaned_df['vulnrichment_date_published'] < min_date) |
        (cleaned_df['vulnrichment_date_published'] > max_date)
    ]
    
    # Add 'cutoff_status' column
    filtered_df['cutoff_status'] = filtered_df['vulnrichment_date_published'].apply(
        lambda x: 'pre_cutoff' if x < min_date else 'post_cutoff'
    )
    
    # Ensure only PUBLISHED vulnerabilities are accepted (discard REJECTED RESERVED DISPUTED)
    filtered_df = filtered_df[(filtered_df['vulnrichment_state']=='PUBLISHED')]
    
    # Drop where no SSVC ground truth exists (e.g. exploitation is empty)
    filtered_df.dropna(subset=['vulnrichment_ssvc_exploitation'], inplace=True)
    
    output_path = os.path.join(file_dump_directory, "phase_1_step_2_clean_vulnrichment.csv")
    filtered_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    return filtered_df
