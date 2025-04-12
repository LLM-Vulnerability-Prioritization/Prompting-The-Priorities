# ========================================================================================
# Import VulZoo, extract subfolders, and extract CVE ID, unstructured and semi-structured fields
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_2_1_Import_Vulzoo(root_dir):
    cprint("Step_2_1_Import_Vulzoo","magenta")

    # Dictionary to store the dataframes for each subfolder
    dataframes = {}
    
    # Iterate over each subfolder
    for subfolder in kv.vulzoo_subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        data = []
        # Check if the subfolder exists
        if os.path.isdir(subfolder_path):
            # Use os.walk to traverse directories recursively
            for dirpath, _, filenames in os.walk(subfolder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        # Read the content of the file
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                        # Create the source by combining directory and file name relative to root_dir
                        source = os.path.relpath(file_path, root_dir)
                        data.append({'Source': source, 'Content': content})
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
            # Create a DataFrame for the current subfolder
            df = pd.DataFrame(data)
            # Store the DataFrame in the dictionary with the subfolder name as the key
            dataframes[subfolder] = df
        else:
            print(f"Subfolder {subfolder_path} does not exist.")
    
    return dataframes

# ========================================================================================

def Step_2_2_Process_Attack(vulzoo_dfs):
    cprint("Step_2_2_Import_Attack","magenta")

    attack_database_raw_df=vulzoo_dfs["attack-database"]
    # Initialize a list to collect all objects
    all_objects = []
    # Iterate over the DataFrame to parse JSON and extract objects
    for index, row in attack_database_raw_df.iterrows():
        content_json_str = row['Content']
        # Parse the JSON content
        try:
            content_dict = json.loads(content_json_str)
            objects = content_dict.get('objects', [])
            all_objects.extend(objects)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    # Normalize the list of objects into a flat DataFrame
    attack_database_df = pd.json_normalize(all_objects, max_level=50)
    
    # Display the new DataFrame
    print(attack_database_df)
    
    # Define the CVE regex pattern
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Function to extract the first CVE ID from a row
    def extract_cve_id(row):
        for column in row:
            if isinstance(column, str):  # Ensure the column is a string
                match = re.search(cve_pattern, column)
                if match:
                    return match.group(0)
        return None
    
    # Apply the function to each row and create a new column 'cve_id'
    attack_database_df['cve_id'] = attack_database_df.apply(extract_cve_id, axis=1)
    
    # Display the updated DataFrame
    print(attack_database_df)
    # Count the number of rows with a non-null 'cve_id'
    attack_database_df['cve_id'].notna().sum()
    
    attack_database_df=attack_database_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return attack_database_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del attack_database_raw_df
        del all_objects
        gc.collect()

# ========================================================================================

def Step_2_3_Process_AttackKB(vulzoo_dfs):
    cprint("Step_2_3_Process_AttackKB","magenta")

    attackerkb_database_raw_df=vulzoo_dfs["attackerkb-database"]
    # Initialize a list to collect all data objects
    all_data_objects = []
    
    # Iterate over the DataFrame to parse JSON and extract data objects
    for index, row in attackerkb_database_raw_df.iterrows():
        content_json_str = row['Content']
        # Parse the JSON content
        try:
            content_dict = json.loads(content_json_str)
            data_objects = content_dict.get('data', [])
            all_data_objects.extend(data_objects)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    # Normalize the list of data objects into a flat DataFrame
    attackerkb_database_df = pd.json_normalize(all_data_objects, max_level=50)
    
    # Define the CVE regex pattern
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Function to extract the first CVE ID from a row
    def extract_cve_id(row):
        for column in row:
            if isinstance(column, str):  # Ensure the column is a string
                match = re.search(cve_pattern, column)
                if match:
                    return match.group(0)
        return None
    
    # Apply the function to each row and create a new column 'cve_id'
    attackerkb_database_df['cve_id'] = attackerkb_database_df.apply(extract_cve_id, axis=1)
    
    # Display the updated DataFrame
    print(attackerkb_database_df)
    # Count the number of rows with a non-null 'cve_id'
    attackerkb_database_df['cve_id'].notna().sum()
    
    attackerkb_database_df=attackerkb_database_df.dropna(subset=['cve_id'])
        
    try:
        # Your main function logic here
        return attackerkb_database_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del attackerkb_database_raw_df
        del all_data_objects
        gc.collect()

# ========================================================================================

def Step_2_4_Process_Bugtraq(vulzoo_dfs):
    cprint("Step_2_4_Process_Bugtraq","magenta")

    bugtraq_raw_df=vulzoo_dfs["bugtraq-database"]
    
    # Define the regex pattern to capture CVE and CAN IDs
    cve_pattern = r'(CVE-\d{4}-\d{4,7}|CAN-\d{4}-\d{4,7})'
    
    def extract_cves(content):
        matches = re.findall(cve_pattern, content)
        return matches if matches else []
    
    # Initialize a list to store new rows
    new_rows = []
    
    # Iterate over the DataFrame to extract CVE IDs and create new rows
    for index, row in bugtraq_raw_df.iterrows():
        cves = extract_cves(row['Content'])
        
        # Create new rows for each CVE found
        if cves:
            for cve in cves:
                new_row = row.copy()
                new_row['cve_id'] = cve
                new_rows.append(new_row)
    
    # Create a new DataFrame from the new rows
    bugtraq_df = pd.DataFrame(new_rows)
    
    # Define the regex pattern to capture CAN IDs
    can_pattern = r'CAN-(\d{4}-\d{4,7})'
    bugtraq_df['cve_id'] = bugtraq_df['cve_id'].str.replace(can_pattern, r'CVE-\1', regex=True)
    bugtraq_df['Content'] = bugtraq_df['Content'].str.replace(can_pattern, r'CVE-\1', regex=True)
    
    def clean_content(text):
        # Define the patterns to remove
        patterns = [
            r"Date:.*?Subject:",  # Remove everything from Date up to Subject
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",  # Remove PGP SIGNATURE block
        ]
        
        # Combine patterns into a single regex
        combined_pattern = "|".join(patterns)
        
        # Use re.sub to remove the patterns
        cleaned_text = re.sub(combined_pattern, "", text, flags=re.DOTALL)
        
        return cleaned_text.strip()

    # Apply the function to the 'Content' column
    bugtraq_df['Cleaned_Content'] = bugtraq_df['Content'].apply(clean_content)

    # Display the updated DataFrame
    print(bugtraq_df)
    
    bugtraq_df=bugtraq_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return bugtraq_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del bugtraq_raw_df
        del new_rows
        gc.collect()

# ========================================================================================

def Step_2_5_Process_Capec(vulzoo_dfs):
    cprint("Step_2_5_Process_Capec","magenta")

    capec_raw_df=vulzoo_dfs["capec-database"]
    print(capec_raw_df)    
    
    # Initialize a list to collect all data objects
    all_data_objects = []
    
    # Iterate over the DataFrame to parse JSON and extract data objects
    for index, row in capec_raw_df.iterrows():
        content_json_str = row['Content']
        # Parse the JSON content
        try:
            content_dict = json.loads(content_json_str)
            cprint(content_dict,"cyan")
            data_objects = content_dict.get('Attack_Pattern_Catalog', {}).get('Attack_Patterns', {}).get('Attack_Pattern', [])
            all_data_objects.extend(data_objects)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    cprint(all_data_objects,"cyan")
    # Normalize the list of data objects into a flat DataFrame
    capec_df = pd.json_normalize(all_data_objects, max_level=10)
    
    # Define the CVE regex pattern
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Function to extract the first CVE ID from a row
    def extract_cve_id(row):
        for column in row:
            if isinstance(column, str):  # Ensure the column is a string
                match = re.search(cve_pattern, column)
                if match:
                    return match.group(0)
        return None
    
    # Apply the function to each row and create a new column 'cve_id'
    capec_df['cve_id'] = capec_df.apply(extract_cve_id, axis=1)
    
    # Display the updated DataFrame
    print(capec_df)
    # Count the number of rows with a non-null 'cve_id'
    capec_df['cve_id'].notna().sum()
    
    capec_df=capec_df.dropna(subset=['cve_id'])
        
    try:
        # Your main function logic here
        return capec_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del capec_raw_df
        del all_data_objects
        gc.collect()

# ========================================================================================

def Step_2_6_Process_KEV(vulzoo_dfs):
    cprint("Step_2_6_Process_KEV","magenta")
    
    kev_raw_df=vulzoo_dfs["cisa-kev-database"]
    print(kev_raw_df)
    
    # Initialize a list to collect all vulnerabilities
    all_vulnerabilities = []
    
    # Iterate over the DataFrame to parse JSON and extract vulnerabilities
    for index, row in kev_raw_df.iterrows():
        content_json_str = row['Content']
        # Parse the JSON content
        try:
            content_dict = json.loads(content_json_str)
            vulnerabilities = content_dict.get('vulnerabilities', [])
            all_vulnerabilities.extend(vulnerabilities)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    # Normalize the list of vulnerabilities into a flat DataFrame
    kev_df = pd.json_normalize(all_vulnerabilities)
    
    # Add the cve_id column
    kev_df['cve_id'] = kev_df['cveID']
    
    # Display the updated DataFrame
    print(kev_df)
    # Count the number of rows with a non-null 'cve_id'
    kev_df['cve_id'].notna().sum()
    
    kev_df=kev_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return kev_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del kev_raw_df
        del all_vulnerabilities
        gc.collect()
    
# ========================================================================================

def Step_2_7_Process_CVE(vulzoo_dfs):
    cprint("Step_2_7_Process_CVE","magenta")
    
    cve_raw_df=vulzoo_dfs["cve-database"]
    print(cve_raw_df)
    
    # Initialize a list to collect all CVE objects
    all_cves = []
    
    # Iterate over the DataFrame to parse JSON and extract CVE objects
    for index, row in cve_raw_df.iterrows():
        content_json_str = row['Content']
        try:
            content_dict = json.loads(content_json_str)
            all_cves.append(content_dict)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    # Normalize the list of CVE objects into a flat DataFrame
    cve_df = pd.json_normalize(all_cves, max_level=20)
    
    # Extract CVE IDs using regex
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    cve_df['cve_id'] = cve_df.apply(lambda row: next((re.search(cve_pattern, str(value)).group(0) for value in row if isinstance(value, str) and re.search(cve_pattern, value)), None), axis=1)
    
    cve_df['cve_id'] =  cve_df['CVE_data_meta.ID']

    # Display the final DataFrame
    print(cve_df)
    # Count the number of rows with a non-null 'cve_id'
    cve_df['cve_id'].notna().sum()
    
    cve_df=cve_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return cve_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del cve_raw_df
        del all_cves
        gc.collect()

# ========================================================================================

def Step_2_8_Process_CWE(vulzoo_dfs):
    cprint("Step_2_8_Process_CWE","magenta")
    
    cwe_raw_df=vulzoo_dfs["cwe-database"]
    print(cwe_raw_df)
    
    # Initialize a list to collect all Weakness objects
    all_weaknesses = []
    
    # Iterate over the DataFrame to parse JSON and extract Weakness objects
    for index, row in cwe_raw_df.iterrows():
        content_json_str = row['Content']
        try:
            content_dict = json.loads(content_json_str)
            weaknesses = content_dict.get('Weakness_Catalog', {}).get('Weaknesses', {}).get('Weakness', [])
            all_weaknesses.extend(weaknesses)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in row {index}: {e}")
    
    # Normalize the list of Weakness objects into a flat DataFrame
    cwe_df = pd.json_normalize(all_weaknesses, max_level=5)
    
    # Extract CVE IDs and descriptions
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    cwe_df['cve_id'] = cwe_df.apply(lambda row: next((re.search(cve_pattern, str(value)).group(0) for value in row if isinstance(value, str) and re.search(cve_pattern, value)), None), axis=1)
    
    # Extract descriptions related to CVE IDs
    def extract_descriptions(row):
        cve_id = row['cve_id']
        if pd.notna(cve_id):
            for col in row.index:
                if isinstance(row[col], str) and cve_id in row[col]:
                    return row.get('Description', ''), row.get('Extended_Description', ''), row[col]
        return None, None, None
    
    cwe_df[['Description', 'Extended_Description', 'Reference_Description']] = cwe_df.apply(extract_descriptions, axis=1, result_type='expand')
    
    # Display the final DataFrame
    print(cwe_df)
    # Count the number of rows with a non-null 'cve_id'
    cwe_df['cve_id'].notna().sum()
    
    cwe_df=cwe_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return cwe_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del cwe_raw_df
        del all_weaknesses
        gc.collect()

# ========================================================================================

def Step_2_9_Process_Exploit(vulzoo_dfs):
    cprint("Step_2_9_Process_Exploit","magenta")
    
    exploit_raw_df=vulzoo_dfs["exploit-db-database"]
    print(exploit_raw_df)
    
    # Parse the JSON content from row 0
    content_json_str = exploit_raw_df.loc[0, 'Content']
    exploit_list = json.loads(content_json_str)
    
    # Normalize the list of exploits into a flat DataFrame
    exploit_df = pd.json_normalize(exploit_list)
    
    # Function to find and append the corresponding content
    def append_exploit_content(row, original_df):
        exploit_id = row['id']
        # Search for the row in the original DataFrame with the matching ID in the Source path
        matching_row = original_df[original_df['Source'].str.contains(f'/{exploit_id}.', regex=False, case=False)]
        if not matching_row.empty:
            return matching_row.iloc[0]['Content']
        return None
    
    cve_pattern = r'(CVE-\d{4}-\d{4,7})'
    # Extract the first CVE ID using regex and store it in a new column 'cve_id'
    exploit_df['cve_id'] = exploit_df['codes'].str.extract(cve_pattern, expand=False)

    # Apply the function to each row in exploit_dict_df
    exploit_df['exploit_content'] = exploit_df.apply(append_exploit_content, axis=1, original_df=exploit_raw_df)
    
    # Display the final DataFrame
    print(exploit_df)
    # Count the number of rows with a non-null 'cve_id'
    exploit_df['cve_id'].notna().sum()
    
    exploit_df=exploit_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return exploit_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del exploit_raw_df
        gc.collect()

# ========================================================================================

def Step_2_10_Process_FullDisclosure(vulzoo_dfs):
    cprint("Step_2_10_Process_FullDisclosure","magenta")
    
    fulldis_raw_df=vulzoo_dfs["full-disclosure-database"]
    print(fulldis_raw_df)
    
    # Define the regex pattern to capture CVE IDs
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Initialize a list to store the results
    results = []
    
    # Iterate over each row in the DataFrame
    for index, row in fulldis_raw_df.iterrows():
        # Extract all CVE IDs from the Content field
        cve_ids = re.findall(cve_pattern, row['Content'])
        # For each CVE ID found, create a new entry in the results list
        for cve_id in cve_ids:
            results.append({'cve_id': cve_id, 'Content': row['Content']})
    
    # Create a new DataFrame from the results
    fulldis_df = pd.DataFrame(results)
    print(fulldis_df)
    fulldis_df['cve_id'].notna().sum()
    
    fulldis_df=fulldis_df.dropna(subset=['cve_id'])
    
    def clean_content(text):
        # Define the patterns to remove
        patterns = [
            r"Date:.*?Subject:",  # Remove everything from Date up to Subject
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",  # Remove PGP SIGNATURE block
        ]
        
        # Combine patterns into a single regex
        combined_pattern = "|".join(patterns)
        
        # Use re.sub to remove the patterns
        cleaned_text = re.sub(combined_pattern, "", text, flags=re.DOTALL)
        
        return cleaned_text.strip()

    # Apply the function to the 'Content' column
    fulldis_df['Cleaned_Content'] = fulldis_df['Content'].apply(clean_content)
    
    try:
        # Your main function logic here
        return fulldis_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del fulldis_raw_df
        del results
        gc.collect()

# ========================================================================================

def Step_2_11_Process_LinuxVulnerabilities(vulzoo_dfs):
    cprint("Step_2_11_Process_LinuxVulnerabilities","magenta")
    
    linuxvul_raw_df=vulzoo_dfs["linux-vulns-database"]
    print(linuxvul_raw_df)
    
    # Separate JSON and MBOX entries
    json_entries = linuxvul_raw_df[linuxvul_raw_df['Source'].str.endswith('.json')]
    mbox_entries = linuxvul_raw_df[linuxvul_raw_df['Source'].str.endswith('.mbox')]
    
    # Initialize a list to store the results
    results = []
    
    # Process each JSON entry
    for index, row in json_entries.iterrows():
        # Parse the JSON content
        json_content = json.loads(row['Content'])
        # Normalize the JSON content
        linuxvul_df = pd.json_normalize(json_content, max_level=20)
        
        # Extract the base file name to match with MBOX
        base_name = row['Source'].rsplit('.', 1)[0]
        
        # Find the corresponding MBOX content
        mbox_content = mbox_entries[mbox_entries['Source'].str.startswith(base_name)]['Content'].values
        mbox_content = mbox_content[0] if len(mbox_content) > 0 else None
        
        # Add the MBOX content to the DataFrame
        linuxvul_df['mbox'] = mbox_content
        
        # Append the result to the list
        results.append(linuxvul_df)
    
    # Concatenate all results into a single DataFrame
    linuxvul_df = pd.concat(results, ignore_index=True)
    
    linuxvul_df['cve_id'] = linuxvul_df['cveMetadata.cveID']
    
    def clean_content(text):
        # Define the patterns to remove
        patterns = [
            r"From:.*?Subject:",  # Remove everything from Date up to Subject
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",  # Remove PGP SIGNATURE block
        ]
        
        # Combine patterns into a single regex
        combined_pattern = "|".join(patterns)
        
        # Use re.sub to remove the patterns
        cleaned_text = re.sub(combined_pattern, "", text, flags=re.DOTALL)
        
        return cleaned_text.strip()

    # Apply the function to the 'Content' column
    linuxvul_df['Cleaned_Mbox'] = linuxvul_df['mbox'].apply(clean_content)
    
    print(linuxvul_df)
    linuxvul_df['cve_id'].notna().sum()
    
    linuxvul_df=linuxvul_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return linuxvul_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del linuxvul_raw_df
        del results
        gc.collect()

# ========================================================================================

def Step_2_12_Process_OSS(vulzoo_dfs):
    cprint("Step_2_12_Process_OSS","magenta")
    
    oss_raw_df=vulzoo_dfs["oss-security-database"]
    print(oss_raw_df)
    
    # Define the regex pattern to capture CVE IDs
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Initialize a list to store the results
    results = []
    
    # Iterate over each row in the DataFrame
    for index, row in oss_raw_df.iterrows():
        # Extract all CVE IDs from the Content field
        cve_ids = re.findall(cve_pattern, row['Content'])
        # For each CVE ID found, create a new entry in the results list
        for cve_id in cve_ids:
            results.append({'cve_id': cve_id, 'Content': row['Content']})
    
    # Create a new DataFrame from the results
    oss_df = pd.DataFrame(results)
    
    print(oss_df)
    oss_df['cve_id'].notna().sum()
    
    oss_df=oss_df.dropna(subset=['cve_id'])
    
    def clean_content(text):
        # Define the patterns to remove
        patterns = [
            r"From:.*?Subject:",  # Remove everything from Date up to Subject
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",  # Remove PGP SIGNATURE block
        ]
        
        # Combine patterns into a single regex
        combined_pattern = "|".join(patterns)
        
        # Use re.sub to remove the patterns
        cleaned_text = re.sub(combined_pattern, "", text, flags=re.DOTALL)
        
        return cleaned_text.strip()

    # Apply the function to the 'Content' column
    oss_df['Cleaned_Content'] = oss_df['Content'].apply(clean_content)
    
    try:
        # Your main function logic here
        return oss_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del oss_raw_df
        del results
        gc.collect()

# ========================================================================================

def Step_2_13_Process_Patch(vulzoo_dfs):
    cprint("Step_2_13_Process_Patch","magenta")
    
    patch_raw_df=vulzoo_dfs["patch-database"]
    print(patch_raw_df)
    
    # Define the regex pattern to capture CVE IDs
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Initialize a list to store the results
    results = []
    
    # Iterate over each row in the DataFrame
    for index, row in patch_raw_df.iterrows():
        # Extract the CVE ID from the Source field
        match = re.search(cve_pattern, row['Source'])
        if match:
            cve_id = match.group(0)
            # Append the result to the list
            results.append({'cve_id': cve_id, 'Content': row['Content']})
    
    # Create a new DataFrame from the results
    patch_df = pd.DataFrame(results)
    
    # Display the new DataFrame
    print(patch_df)
    patch_df['cve_id'].notna().sum()
    
    patch_df=patch_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return patch_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del patch_raw_df
        del results
        gc.collect()

# ========================================================================================

def Step_2_14_Process_ZDI(vulzoo_dfs):
    cprint("Step_2_14_Process_ZDI","magenta")
    
    zdi_raw_df=vulzoo_dfs["zdi-advisory-database"]
    print(zdi_raw_df)
    
    # Initialize a list to store the results
    results = []
    
    # Iterate over each row in the DataFrame
    for index, row in zdi_raw_df.iterrows():
        # Parse the JSON content
        json_content = json.loads(row['Content'])
        # Normalize the JSON content
        normalized_df = pd.json_normalize(json_content, max_level=10)
        # Append the result to the list
        results.append(normalized_df)
    
    # Concatenate all results into a single DataFrame
    zdi_df = pd.concat(results, ignore_index=True)
    
    zdi_df['cve_id'] = zdi_df['cveId']
    
    # Display the new DataFrame
    print(zdi_df)
    zdi_df['cve_id'].notna().sum()
    
    zdi_df=zdi_df.dropna(subset=['cve_id'])
    
    try:
        # Your main function logic here
        return zdi_df
    finally:
        # Cleanup code that runs regardless of where the function exits
        del zdi_raw_df
        del results
        gc.collect()

# ========================================================================================

def Step_2_15_Merge_Cleaned_Vulzoo(attackerkb_df, bugtraq_df, kev_df, cve_df, exploit_df, fulldis_df, linuxvul_df, oss_df, patch_df, zdi_df, cve_id_list):
    cprint("Step_2_15_Merge_Cleaned_Vulzoo","magenta")

    # AttackerKB Repo
    attackerkb_subset_df = attackerkb_df[attackerkb_df['cve_id'].isin(cve_id_list)]
    attackerkb_subset_df = attackerkb_subset_df[['cve_id','document','metadata.mitre-tactics','metadata.timeline','metadata.vulnerable-versions','metadata.vendor.productNames','metadata.tags']]
    new_column_names = {
    'cve_id': 'cve_id',
    'document': 'attackerkb_document_description',
    'metadata.mitre-tactics': 'attackerkb_mitre_tactics',
    'metadata.timeline': 'attackerkb_timeline',
    'metadata.vulnerable-versions': 'attackerkb_vulnerable_versions',
    'metadata.vendor.productNames': 'attackerkb_vendor_product_names',
    'metadata.tags': 'attackerkb_tags',
    }
    attackerkb_subset_df.rename(columns=new_column_names, inplace=True)
    attackerkb_subset_df = attackerkb_subset_df.astype(str)
    attackerkb_subset_df = attackerkb_subset_df.drop_duplicates()
    
    bugtraq_subset_df = bugtraq_df[bugtraq_df['cve_id'].isin(cve_id_list)]
    bugtraq_subset_df = bugtraq_subset_df[['cve_id','Cleaned_Content']]
    new_column_names = {
    'cve_id': 'cve_id',
    'Cleaned_Content': 'bugtraq_advisory_email'
    }
    bugtraq_subset_df.rename(columns=new_column_names, inplace=True)
    bugtraq_subset_df = bugtraq_subset_df.astype(str)
    bugtraq_subset_df = bugtraq_subset_df.drop_duplicates()
    
    kev_subset_df = kev_df[kev_df['cve_id'].isin(cve_id_list)]
    kev_subset_df = kev_subset_df[['cve_id','vendorProject','product','vulnerabilityName','shortDescription','requiredAction','knownRansomwareCampaignUse']]
    new_column_names = {
    'cve_id': 'cve_id',
    'vendorProject': 'kev_vendor',
    'product': 'kev_product',
    'vulnerabilityName': 'kev_vulnerability_name',
    'shortDescription': 'kev_short_description',
    'requiredAction': 'kev_required_action',
    'knownRansomwareCampaignUse': 'kev_known_ransomware_campaign_use'
    }
    kev_subset_df.rename(columns=new_column_names, inplace=True)
    kev_subset_df = kev_subset_df.astype(str)
    kev_subset_df = kev_subset_df.drop_duplicates()
    
    cve_subset_df = cve_df[cve_df['cve_id'].isin(cve_id_list)]
    cve_subset_df = cve_subset_df[['cve_id','description.description_data']]
    new_column_names = {
    'cve_id': 'cve_id',
    'description.description_data': 'cve_description'
    }
    cve_subset_df.rename(columns=new_column_names, inplace=True)
    cve_subset_df = cve_subset_df.astype(str)
    cve_subset_df = cve_subset_df.drop_duplicates()
    
    exploit_subset_df = exploit_df[exploit_df['cve_id'].isin(cve_id_list)]
    exploit_subset_df = exploit_subset_df[['cve_id','description','exploit_content']]
    new_column_names = {
    'cve_id': 'cve_id',
    'description': 'exploit_description',
    'exploit_content': 'exploit_content'
    }
    exploit_subset_df.rename(columns=new_column_names, inplace=True)
    exploit_subset_df = exploit_subset_df.astype(str)
    exploit_subset_df = exploit_subset_df.drop_duplicates()
    
    fulldis_subset_df = fulldis_df[fulldis_df['cve_id'].isin(cve_id_list)]
    fulldis_subset_df = fulldis_subset_df[['cve_id','Cleaned_Content']]
    new_column_names = {
    'cve_id': 'cve_id',
    'Cleaned_Content': 'fulldis_advisory_email'
    }
    fulldis_subset_df.rename(columns=new_column_names, inplace=True)
    fulldis_subset_df = fulldis_subset_df.astype(str)
    fulldis_subset_df = fulldis_subset_df.drop_duplicates()
    
    linuxvul_subset_df = linuxvul_df[linuxvul_df['cve_id'].isin(cve_id_list)]
    linuxvul_subset_df = linuxvul_subset_df[['cve_id','containers.cna.descriptions','containers.cna.title','Cleaned_Mbox']]
    new_column_names = {
    'cve_id': 'cve_id',
    'containers.cna.descriptions': 'linuxvul_descriptions',
    'containers.cna.title': 'linuxvul_title',
    'Cleaned_Mbox': 'linuxvul_email_advisory'
    }
    linuxvul_subset_df.rename(columns=new_column_names, inplace=True)
    linuxvul_subset_df = linuxvul_subset_df.astype(str)
    linuxvul_subset_df = linuxvul_subset_df.drop_duplicates()
    
    oss_subset_df = oss_df[oss_df['cve_id'].isin(cve_id_list)]
    oss_subset_df = oss_subset_df[['cve_id','Cleaned_Content']]
    new_column_names = {
    'cve_id': 'cve_id',
    'Cleaned_Content': 'oss_advisory_email'
    }
    oss_subset_df.rename(columns=new_column_names, inplace=True)
    oss_subset_df = oss_subset_df.astype(str)
    oss_subset_df = oss_subset_df.drop_duplicates()
    
    # Patch repo
    patch_subset_df = patch_df[patch_df['cve_id'].isin(cve_id_list)]
    patch_subset_df = patch_subset_df[['cve_id','Content']]
    new_column_names = {
    'cve_id': 'cve_id',
    'Content': 'patch_code'
    }
    patch_subset_df.rename(columns=new_column_names, inplace=True)
    patch_subset_df = patch_subset_df.astype(str)
    patch_subset_df = patch_subset_df.drop_duplicates()
    
    zdi_df_subset_df = zdi_df[zdi_df['cve_id'].isin(cve_id_list)]
    zdi_df_subset_df = zdi_df_subset_df[['cve_id','title','vendors','products','description','addtionnal_details','timeline']]
    new_column_names = {
    'cve_id': 'cve_id',
    'title': 'zdi_title',
    'vendors': 'zdi_vendors',
    'products': 'zdi_products',
    'description': 'zdi_description',
    'addtionnal_details': 'zdi_additional_details',
    'timeline': 'zdi_timeline'
    }
    zdi_df_subset_df.rename(columns=new_column_names, inplace=True)
    zdi_df_subset_df = zdi_df_subset_df.astype(str)
    zdi_df_subset_df = zdi_df_subset_df.drop_duplicates()
    
    # List of DataFrames and their names
    dataframes = [
        ('attackerkb', attackerkb_subset_df),
        ('bugtraq', bugtraq_subset_df),
        ('kev', kev_subset_df),
        ('cve', cve_subset_df),
        ('exploit', exploit_subset_df),
        ('fulldis', fulldis_subset_df),
        ('linuxvul', linuxvul_subset_df),
        ('oss',oss_subset_df),
        ('patch', patch_subset_df),
        ('zdi', zdi_df_subset_df)
    ]
    
    # Start with the first DataFrame
    merged_df = dataframes[0][1]
    
    # Merge each DataFrame with the suffix
    for name, df in dataframes[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='cve_id',
            how='outer',  # Choose the type of join you need
            suffixes=('', f'_{name}')
        )

    # Convert list-type columns to tuples
    # Function to concatenate differing values
    # Function to convert lists to strings
    def list_to_string(x):
        if isinstance(x, list):
            return ', '.join(map(str, x))
        return x
    
    # Apply the conversion to all columns
    merged_df = merged_df.applymap(list_to_string)
    
    # Function to concatenate unique values
    def concat_unique(series):
        return '|'.join(series.dropna().unique())
    
    # Group by 'cve_id' and apply the concatenation function
    result_df = merged_df.groupby('cve_id').agg(concat_unique).reset_index()

    return result_df
