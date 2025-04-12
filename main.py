# ========================================================================================
# Import modules and environment variables
# ========================================================================================
from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================
# Import script files
# ========================================================================================
import vulnrichment_data_processing as vrdp
import vulzoo_data_processing as vzdp
import prepare_samples_queries as psq
import openrouter_api_call as oac
import clean_llm_response as clr
import evaluate_llm_performance as elp

# ========================================================================================
# Initialise and run
# ========================================================================================

# VULNRICHMENT and VULZOO directories
vulnrichment_directory = "vulnrichment"
vulzoo_directory = "vulzoo_processed"
    
# Create a folder to capture CSVs of dataframes from each step
current_time = datetime.now().strftime("%Y%m%d-%H%M") 
folder_name = f"file_dumps_{current_time}"
os.makedirs(folder_name, exist_ok=True)

# ========================================================================================
# Step 1: VULNRICHMENT Processing

Step_1_1_Import_Vulnrichment_df = vrdp.Step_1_1_Import_Vulnrichment(vulnrichment_directory,folder_name)
Step_1_2_Clean_Vulnrichment_df = vrdp.Step_1_2_Clean_Vulnrichment(Step_1_1_Import_Vulnrichment_df,folder_name)

cve_list = Step_1_2_Clean_Vulnrichment_df['cve_id'].unique()
vulnrichment_cve_count = Step_1_2_Clean_Vulnrichment_df['cve_id'].nunique()

# ========================================================================================
# Step 2: VulZoo Processing

Step_2_1_Import_Vulzoo_df = vzdp.Step_2_1_Import_Vulzoo(vulzoo_directory)

Step_2_2_Process_Attack_df = vzdp.Step_2_2_Process_Attack(Step_2_1_Import_Vulzoo_df)
Step_2_3_Process_AttackKB_df = vzdp.Step_2_3_Process_AttackKB(Step_2_1_Import_Vulzoo_df)
Step_2_4_Process_Bugtraq_df = vzdp.Step_2_4_Process_Bugtraq(Step_2_1_Import_Vulzoo_df)
Step_2_5_Process_Capec_df = vzdp.Step_2_5_Process_Capec(Step_2_1_Import_Vulzoo_df)
Step_2_6_Process_KEV_df = vzdp.Step_2_6_Process_KEV(Step_2_1_Import_Vulzoo_df)
Step_2_7_Process_CVE_df = vzdp.Step_2_7_Process_CVE(Step_2_1_Import_Vulzoo_df)
Step_2_8_Process_CWE_df = vzdp.Step_2_8_Process_CWE(Step_2_1_Import_Vulzoo_df)
Step_2_9_Process_Exploit_df = vzdp.Step_2_9_Process_Exploit(Step_2_1_Import_Vulzoo_df)
Step_2_10_Process_FullDisclosure_df = vzdp.Step_2_10_Process_FullDisclosure(Step_2_1_Import_Vulzoo_df)
Step_2_11_Process_LinuxVulnerabilities_df = vzdp.Step_2_11_Process_LinuxVulnerabilities(Step_2_1_Import_Vulzoo_df)
Step_2_12_Process_OSS_df = vzdp.Step_2_12_Process_OSS(Step_2_1_Import_Vulzoo_df)
Step_2_13_Process_Patch_df = vzdp.Step_2_13_Process_Patch(Step_2_1_Import_Vulzoo_df)
Step_2_14_Process_ZDI_df = vzdp.Step_2_14_Process_ZDI(Step_2_1_Import_Vulzoo_df)

Step_2_15_Merge_Cleaned_Vulzoo_df = vzdp.Step_2_15_Merge_Cleaned_Vulzoo(Step_2_3_Process_AttackKB_df, Step_2_4_Process_Bugtraq_df, Step_2_6_Process_KEV_df, Step_2_7_Process_CVE_df, Step_2_9_Process_Exploit_df, Step_2_10_Process_FullDisclosure_df, Step_2_11_Process_LinuxVulnerabilities_df, Step_2_12_Process_OSS_df, Step_2_13_Process_Patch_df, Step_2_14_Process_ZDI_df, cve_list)

# ========================================================================================
# Step 3: Merge clean VULNRICHMENT and VulZoo

Step_3_1_Merge_Vulnrichment_Vulzoo_df = pd.merge(Step_2_15_Merge_Cleaned_Vulzoo_df,Step_1_2_Clean_Vulnrichment_df,on='cve_id',how='left')  

# ========================================================================================
# Step 4: Select samples and prepare queries

Step_4_1_Assign_Detail_Levels_df = psq.Step_4_1_Assign_Detail_Levels(Step_3_1_Merge_Vulnrichment_Vulzoo_df,folder_name)
Step_4_2_Sample_Detail_Cutoff_df = psq.Step_4_2_Sample_Detail_Cutoff(Step_4_1_Assign_Detail_Levels_df,384,folder_name)
Step_4_3_Generate_Queries_df= psq.Step_4_3_Generate_Queries(Step_4_2_Sample_Detail_Cutoff_df,kv.trials,kv.llms,kv.approaches,folder_name)

# ========================================================================================
# Step 5: Prepare API call and run queries

Step_5_1_Run_LLM_Queries_df = oac.Step_5_1_Run_LLM_Queries(Step_4_3_Generate_Queries_df,folder_name,openrouter_url,openrouter_key)

# ========================================================================================
# Step 6: Clean responses from LLMs, and create a second dataframe without Unknown responses

Step_6_Clean_LLM_Response_df = clr.Step_6_Clean_LLM_Response(Step_5_1_Run_LLM_Queries_df,folder_name)
Step_6_Clean_LLM_Response_Excluding_Unknown_df = Step_6_Clean_LLM_Response_df[Step_6_Clean_LLM_Response_df['llm-ssvc-decision']!='unknown']

# ========================================================================================
# Step 7: Evaluate LLM performance, both overall, and for detail and cutoff categories. Evaluate for datasets with and without Unknown responses

# Including unknown responses
Step_7_1_Evaluate_LLM_Performance_df = elp.Step_7_1_Evaluate_LLM_Performance(Step_6_Clean_LLM_Response_df,folder_name)
Step_7_2_Pre_Cutoff_Performance_df, Step_7_2_Post_Cutoff_Performance_df, Step_7_2_High_Detail_Performance_df, Step_7_2_Medium_Detail_Performance_df, Step_7_2_Low_Detail_Performance_df = elp.Step_7_2_Evaluate_Detail_Cutoff_Performance(Step_6_Clean_LLM_Response_df,folder_name)

# Excluding unknown responses
Step_7_1_Evaluate_LLM_Excluding_Unknown_Performance_df = elp.Step_7_1_Evaluate_LLM_Performance(Step_6_Clean_LLM_Response_df,folder_name)
Step_7_2_Pre_Cutoff_Performance_Excluding_Unknown_df, Step_7_2_Post_Cutoff_Performance_Excluding_Unknown_df, Step_7_2_High_Detail_Performance_Excluding_Unknown_df, Step_7_2_Medium_Detail_Performance_Excluding_Unknown_df, Step_7_2_Low_Detail_Performance_Excluding_Unknown_df = elp.Step_7_2_Evaluate_Detail_Cutoff_Performance(Step_6_Clean_LLM_Response_Excluding_Unknown_df,folder_name)
