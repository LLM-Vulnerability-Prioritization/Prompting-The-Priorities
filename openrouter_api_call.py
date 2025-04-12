# ========================================================================================
# Prepare API call and run queries
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_5_OpenRouter_API_Call(llm, system_role, user_role, openrouter_url, openrouter_key):
    cprint("Step_5_OpenRouter_API_Call","magenta")

    try:
        response = requests.post(
          url=openrouter_url,
          headers={
            "Authorization": f"Bearer {openrouter_key}",
          },
          data=json.dumps({
            "model": llm,
            "response_format": { 'type': 'json_object' },
            "temperature": 0.7,
            "messages": [
              { "role": "system", "content": system_role }, # or ssvc_context_prompt
              { "role": "user", "content": user_role }
                  ],
            "top_k": 50,
            "top_p": 0.7,
                }
            )
        )
        response_json = response.json()
    except requests.exceptions.JSONDecodeError:
        # If a JSONDecodeError occurs, return the default response
        response_json = {
            'Exploitation': 'unknown',
            'Automatable': 'unknown',
            'Technical_Impact': 'unknown',
            'Mission_Wellbeing': 'unknown'
        }
    except requests.exceptions.RequestException as e:
        # Handle other potential request exceptions
        print(f"An error occurred: {e}")
        response_json = {
            'Exploitation': 'unknown',
            'Automatable': 'unknown',
            'Technical_Impact': 'unknown',
            'Mission_Wellbeing': 'unknown'
        }
    # OpenRouter reponse
    content = response_json['choices'][0]['message']['content']
    
    return content, str(response_json)

# ========================================================================================

def Step_5_Process_Chunk(chunk, output_file_prefix, openrouter_url, openrouter_key):
    cprint("Step_5_OpenRouter_API_Call","magenta")

    for index, vuln in chunk.iterrows():
        if vuln['llm-query-processed'] == 1:
            cprint(f"Skipped. Already processed row {index}", "yellow")
            continue
        
        try:
            llm_response, llm_raw_response = Step_5_OpenRouter_API_Call(
                vuln['llm'],
                vuln['system_role_prompt'],
                vuln['user_role_prompt'], 
                openrouter_url, 
                openrouter_key
            )
            
            chunk.loc[index, 'llm-response'] = llm_response
            chunk.loc[index, 'llm-raw-response'] = llm_raw_response
            chunk.loc[index, 'llm-query-processed'] = 1
            
            cprint(f"Processed row {index}", "green")
            cprint(f"LLM Response: {llm_response}", "cyan")
            
            # Export after each query
            output_file = f"{output_file_prefix}_{multiprocessing.current_process().name}.tsv"
            chunk.to_csv(output_file, sep='\t', index=False)
            cprint(f"Saved interim result to {output_file}", "blue")
            
        except Exception as e:
            cprint(f"Error processing row {index}: {str(e)}", "red")
            continue
    
    return chunk

# ========================================================================================

def Step_5_1_Run_LLM_Queries(df, output_file_prefix, openrouter_url, openrouter_key):
    cprint("Step_5_1_Run_LLM_Queries","magenta")

    num_cores = multiprocessing.cpu_count()
    cprint(f"Using {num_cores} CPU cores for parallel processing", "magenta")
    
    chunks = np.array_split(df, num_cores)
    
    process_chunk_partial = partial(Step_5_Process_Chunk, output_file_prefix=output_file_prefix)
    
    with multiprocessing.Pool(num_cores) as pool:
        processed_chunks = pool.map(process_chunk_partial, chunks)
    
    result_df = pd.concat(processed_chunks, ignore_index=True)
    return result_df