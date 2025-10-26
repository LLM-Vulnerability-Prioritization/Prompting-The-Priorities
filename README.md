_Prompting the Priorities: Evaluating LLMs for Organization-Specific Vulnerability Prioritization_ - Workshop Paper for the Workshop on AI for Cyber Threat Intelligence (WAITI) at the 2025 Annual Computer Security Applications Conference (ACSAC) [TBC].

Python script that: (1) processes ground truth SSVC data from Vulnrichment; (2) processes vulnerability data from VulZoo; (3) parses the vulnerability data by prompting technique and prompting best practice; (4) selects sample of vulnerabilities common to both Vulnrichment and VulZoo; (5) queries LLMs with the sample prompts via OpenRouter API; and (6) evaluates LLM and prompting technique performance.

Run from main.py.

**Data Sources**

Vulnrichment Ground Truth: https://github.com/cisagov/vulnrichment 

VulZoo Vulnerability Data: https://github.com/NUS-Curiosity/VulZoo

**Data Files**

Script analysis data files: [https://mega.nz/folder/2fRVkS5C#P3J22tluXrTjYZihhZgQKQ](https://mega.nz/folder/2fRVkS5C#P3J22tluXrTjYZihhZgQKQ)

- prompt_queries_llm_responses.csv
  
  Contains the queries sent to the LLMs and the responses received from the LLMs. Notable fields include:
  - cve_id: Vulnerability identifier
  - vulnrichment_ssvc_exploitation: Exploitation ground truth from Vulnrichment
  - vulnrichment_ssvc_automatable: Automatable ground truth from Vulnrichment
  - vulnrichment_ssvc_technical_impact: Technical Impact ground truth from Vulnrichment
  - scenario_system_role_risk: Mission & Wellbeing stand-in scenario as not included in Vulnrichment
  - system_role_prompt: Provided to the system role of the LLMs. Sets the SSVC context, as well as the Mission & Wellbeing organisation context
  - llm: LLM identifier to instruct OpenRouter which LLM to use
  - prompt: Stipulates prompting technique used
  - user_role_prompt: Provided to the user role of the LLMs. Contains the vulnerability data, parsed with the selected prompting technique and best practice (e.g. XML tags, imperative tense, etc) and the instruction to the LLM to analyse the data and return SSVC decision point values (or unknown if unsure)
  - trial: Stipulates which of three trials the query is part of
  - llm-raw-response: Records the entire response from the LLM
  - llm-Exploitation: Extracts the LLM response for the Exploitation decision point
  - llm-Automatable: Extracts the LLM response for the Automatable decision point
  - llm-Technical_Impact: Extracts the LLM response for the Technical_Impact decision point
  - llm-Mission_Wellbeing: Extracts the LLM response for the Mission & Wellbeing decision point
  - llm-ssvc-decision: The decision outcome generated when the llm-Exploitation, llm-Automatable, llm-Technical_Impact and llm-Mission_Wellbeing are parsed through the SSVC decision tree
  - ground_truth: The decision outcome generated when the vulnrichment_ssvc_exploitation, vulnrichment_ssvc_automatable, vulnrichment_ssvc_technical_impact and scenario_system_role_risk are parsed through the SSVC decision tree

- llm_pt_sdp_f1_harmonic_means.csv

  F1-Scores are calculated based on the performance of an LLM and prompting technique combination to accurately predict values for an SSVC decision point in a given trial. The F1-Scores from the three trials are then used to determine the harmonic mean, thereby providing a single score for the LLM, prompting technique and SSVC decision point combination.

- cohens_kappa_results.csv

  Calculation of Weighted and Unweighted Cohen's Kappas when comparing llm-ssvc-decision and ground_truth from the prompt_queries_llm_responses.csv. Weighted provides insights into similarity (e.g. an LLM responding Attend is more performant than an LLM responding Track when the ground truth is Act).
