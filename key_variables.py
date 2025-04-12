# LLM identifiers for OpenRouter API.
llms = [
    {"name": "openai/gpt-4o-mini", "cutoff_date": "2023-10-01", "context_window": 128000}, # https://platform.openai.com/docs/models
    {"name": "anthropic/claude-3-haiku", "cutoff_date": "2023-08-01", "context_window": 200000}, # https://context.ai/model/claude-3-haiku
    {"name": "google/gemini-flash-1.5", "cutoff_date": "2023-11-01", "context_window": 1000000}, # https://docsbot.ai/models/gemini-1-5-flash-001
    {"name": "deepseek/deepseek-r1", "cutoff_date": "2024-07-01", "context_window": 128000}, # https://docsbot.ai/models/deepseek-r1. No knowledge cutoff date provided. Asking DeepSeek returns 2024-07-01 and it can accurately recall events just prior to this cutoff (e.g. 2024 UK election outcome)
]

# Number of times to run a prompt against on LLM (since LLMs not deterministic)
trials = 3

# Level of detail available for a vulnerability and a category (i.e. proportion of fields with a value)
data_detail_levels = {
    'high_detail': {'lower': 0.4, 'upper': 1.0},
    'medium_detail': {'lower': 0.2, 'upper': 0.4},
    'low_detail': {'lower': 0.0, 'upper': 0.2}
}

# Instruction for LLM to analyse vulnerabilities using the SSVC system and to return SSVC decision point values. Prior versions asked LLMs to return final Outcome
approaches = [                    
                {"approach": "decision_points",
                 "query_instruction": "return responses for the four SSVC decision points",
                 "response_examples": {"act": "{'Exploitation': ’active’, 'Automatable': 'yes', 'Technical_Impact': 'total', 'Mission_Wellbeing': 'high'}", "track": "{'Exploitation': ’none’, 'Automatable': 'no', 'Technical_Impact': 'partial', 'Mission_Wellbeing': 'low'}"},
                 "output_instruction": """<Instruction>Return a JSON object response and only a JSON object response. Refer to this JSON object example if you know the response.</Instruction><FormatOutputExample>{'Exploitation': ’active’, 'Automatable': 'yes', 'Technical_Impact': 'total', 'Mission_Wellbeing': 'high'}</FormatOutputExample><Instruction>Refer to this JSON object example if you are unsure of the response.</Instruction><FormatOutputExample>{'Exploitation': 'unknown', 'Automatable': 'unknown', 'Technical_Impact': 'unknown', 'Mission_Wellbeing': 'unknown'}</FormatOutputExample>""",  
                 "system_role": "Act as a Stakeholder-Specific Vulnerability Categorization (SSVC) system expert. Analyse vulnerabilities and return values for Exploitation:  ['active','poc','none']  , Automatable: ['no', 'yes'], Technical_Impact: ['partial','total'], and Mission_Wellbeing: ['low','medium','high']. Admit ignorance if you are unsure of the response.",
                 }
            ]

# Given VULNRICHMENT ground truth does not have Mission & Wellbeing ground truth, representative stand-ins are needed
mission_wellbeing_standins = [
            {
                "scenario_risk": "low",
                "scenario_description" : "Assume you work for a small town leisure centre that provides sport and recreational facilities and services to the community."
            },
            {
                "scenario_risk": "medium",
                "scenario_description" : "Assume you work for a large supermarket chain that operates outlets across the state."
            },
            {
                "scenario_risk": "high",
                "scenario_description" : "Assume you work for an electric power generation company that is the sole provider for the country."
            },
    ]

# List of the 17 VulZoo subfolders
vulzoo_subfolders = [
    'attack-database', 'attackerkb-database', 'bugtraq-database', 'capec-database', 'cisa-kev-database',
    'cve-database', 'cwe-database', 'd3fend-database', 'exploit-db-database', 'full-disclosure-database',
    'github-advisory-database', 'linux-vulns-database', 'nvd-database', 'oss-security-database',
    'patch-database', 'relationships', 'zdi-advisory-database'
]

# List of VulZoo vulnerability repos and fields used in queries for dashboard display
vulzoo_repos_fields = ['attackerkb-database | document_description','attackerkb-database | mitre_tactics','attackerkb-database | timeline','attackerkb-database | vulnerable_versions','attackerkb-database | vendor_product_names','attackerkb-database | tags',
                        'bugtraq | advisory_email',
                        'cisa-kev-database | vendor','cisa-kev-database | product','cisa-kev-database | vulnerability_name','cisa-kev-database | short_description','cisa-kev-database | required_action','cisa-kev-database | known_ransomware_campaign_use',
                        'cve-database | description',
                        'exploit-db-database | description','exploit-db-database | content',
                        'full-disclosure-database | advisory_email',
                        'linux-vulns-database | descriptions','linux-vulns-database | title','linux-vulns-database | email_advisory',
                        'oss-security-database | advisory_email',
                        'patch-database | code',
                        'zdi-advisory-database | title','zdi-advisory-database | vendors','zdi-advisory-database | products','zdi-advisory-database | description','zdi-advisory-database | additional_details','zdi-advisory-database | timeline'
                    ]
