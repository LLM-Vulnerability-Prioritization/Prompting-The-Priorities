# ========================================================================================
# Clean responses from LLMs
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_6_1_Clean_LLM_Response(df, file_dump_directory):
    cprint("Step_6_1_Clean_LLM_Response", "magenta")

    # Define constants
    DECISION_PATTERN = r"\{\s*['\"]Exploitation['\"]\s*:\s*['\"](none|poc|active|unknown)['\"]\s*,\s*['\"]Automatable['\"]\s*:\s*['\"](yes|no|unknown)['\"]\s*,\s*['\"]Technical_Impact['\"]\s*:\s*['\"](partial|total|unknown)['\"]\s*,\s*['\"]Mission_Wellbeing['\"]\s*:\s*['\"](low|medium|high|unknown)['\"]\s*\}"
    
    # Initialize new columns with default values
    decision_columns = ['llm-Exploitation', 'llm-Automatable', 'llm-Technical_Impact', 'llm-Mission_Wellbeing']
    for col in decision_columns:
        df[col] = 'unknown'

    def standardize_value(field: str, value: str) -> str:
        """Standardize values based on field-specific rules."""
        value = value.lower()
        if field == 'Exploitation':
            return 'poc' if re.sub(r'[^a-zA-Z]', '', value) == 'publicpoc' else value
        elif field == 'Technical_Impact':
            if value == 'high':
                return 'total'
            elif value == 'low':
                return 'partial'
        return value

    def extract_decision_points(text: str) -> tuple[str, str, str, str]:
        """Extract and standardize decision points from text."""
        try:
            if isinstance(text, str):
                # Try to find pattern match first
                if match := re.search(DECISION_PATTERN, text):
                    values = match.groups()
                else:
                    # Try JSON parsing as fallback
                    data = json.loads(text)
                    values = (
                        data.get('Exploitation', 'error'),
                        data.get('Automatable', 'error'),
                        data.get('Technical_Impact', 'error'),
                        data.get('Mission_Wellbeing', 'error')
                    )
                
                # Standardize values
                return (
                    standardize_value('Exploitation', values[0]),
                    standardize_value('Automatable', values[1]),
                    standardize_value('Technical_Impact', values[2]),
                    standardize_value('Mission_Wellbeing', values[3])
                )
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            cprint(f"Error processing response: {e}", "red")
            cprint(text,"cyan")
        
        return ('unknown', 'unknown', 'unknown', 'unknown')

    def get_ssvc_decision(e: str, a: str, ti: str, mw: str) -> str:
        """Get SSVC decision if all values are known."""
        if any(val == 'unknown' for val in (e, a, ti, mw)):
            return 'unknown'
        
        if any(val == 'error' for val in (e, a, ti, mw)):
            return 'error'
        
        try:
            decision = ssvc.Decision(
                exploitation=e,
                automatable=a,
                technical_impact=ti,
                mission_wellbeing=mw,
            )
            return str(decision.evaluate().action.value)
        except Exception as e:
            cprint(f"Error in SSVC decision: {e}", "red")
            return 'unknown'

    # Process each row
    for index, row in df.iterrows():
        if 'llm-response' not in row:
            continue
            
        # Extract and store decision points
        exploitation, automatable, tech_impact, mission_wellbeing = extract_decision_points(row['llm-response'])
        
        df.loc[index, 'llm-Exploitation'] = exploitation
        df.loc[index, 'llm-Automatable'] = automatable
        df.loc[index, 'llm-Technical_Impact'] = tech_impact
        df.loc[index, 'llm-Mission_Wellbeing'] = mission_wellbeing
        
        # Calculate SSVC decision
        df.loc[index, 'llm-ssvc-decision'] = get_ssvc_decision(
            exploitation, automatable, tech_impact, mission_wellbeing
        )


    def extract_tokens(response_str):
        """Safely parse the Python dictionary and extract 'usage' info."""
        try:
            # Parse the Python-style dictionary
            response_dict = ast.literal_eval(response_str)
            
            usage = response_dict.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            return pd.Series({
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            })
        except (SyntaxError, ValueError) as e:
            # Fallback: try to convert single quotes to double quotes, then parse as JSON
            # This is optional, depending on your data. If data is consistently Python syntax,
            # you may omit the fallback.
            try:
                response_str_json = re.sub(r"(?<!\\)'", '"', response_str)
                response_dict = json.loads(response_str_json)
                usage = response_dict.get('usage', {})
                return pd.Series({
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                })
            except Exception as e:
                print(f"Error parsing tokens: {e}")
                return pd.Series({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        except Exception as e:
            print(f"Unexpected error: {e}")
            return pd.Series({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

    # Create the new token columns
    df[['prompt_tokens', 'completion_tokens', 'total_tokens']] = df['llm-raw-response'].apply(extract_tokens)

    # Make sure they are integers
    df[['prompt_tokens', 'completion_tokens', 'total_tokens']] = df[[
        'prompt_tokens', 
        'completion_tokens', 
        'total_tokens'
    ]].astype(int)

    cprint(f"Final dataframe head:\n{df.head()}", "green")

    # Export results
    output_path = os.path.join(file_dump_directory, "Step_6_1_Clean_LLM_Response.csv")
    df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')

    return df