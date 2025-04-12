# ========================================================================================
# Evaluate LLM performance, both overall, and for detail and cutoff categories
# ========================================================================================

from module_env_import import *
from prompting_technique_templates import prompt_techniques
import key_variables as kv

# ========================================================================================

def Step_8_1_Create_Dashboard(vulnrichment_cve_count, df_vuln, harmonic_means_df, pre_cutoff_df, post_cutoff_df, high_detail_df, medium_level_df, low_detail_df, folder_path):

    # Create the output file path
    output_filename = "llm_performance_dashboard.html"
    output_path = os.path.join(folder_path, output_filename)
    # Set the output file
    output_file(output_path, title="LLM Performance Dashboard")
    # List of metrics to create plots for
    metrics = ['precision', 'recall', 'accuracy', 'f1', 'f2', 'auc_pr', 'mcc']
    # Create all tabs
    tabs = []

    # Key Details Tab
    # ====================================================
    # Number of vulnerabilities by a given date
    df_vuln['datetime'] = pd.to_datetime(df_vuln['vulnrichment_date_published'])
    latest_datetime = df_vuln['datetime'].max()
    query_count = len(df_vuln)
    distinct_cve_count = df_vuln['cve_id'].nunique()

    def generate_llm_list_html(llms):
        list_items = "".join([f"<li>{llm['name']}</li>" for llm in llms])
        return f"<h3>LLMs</h3><ul>{list_items}</ul>"
    
    def generate_prompting_techniques_html(techniques):
        list_items = "".join([f"<li>{technique}</li>" for technique in techniques.keys()])
        return f"<h3>Prompting Techniques</h3><ul>{list_items}</ul>"
        
    def generate_risk_scenarios_html(scenarios):
        list_items = "".join([f"<li>{scenario['scenario_risk'].capitalize()}: {scenario['scenario_description']}</li>" for scenario in scenarios])
        return f"""
        <h3>Organization Risk Scenarios</h3>
        <ul>{list_items}</ul>
        <h5>Stand-in for Mission Wellbeing as not present in VULNRICHMENT.</h5>
        """

    def find_min_max_dates(llms):
        dates = [datetime.strptime(llm['cutoff_date'], '%Y-%m-%d') for llm in llms]
        return min(dates), max(dates)

    def format_date(date, format_type):
        if format_type == 'pre':
            return date.strftime('%b-%Y')
        else:  # post
            return date.strftime('%B %Y')
            
    def generate_data_detail_content(levels):
        content = "<h2>Data Detail Levels</h2>"
        for level, range_dict in levels.items():
            level_name = level.replace('_', ' ').title()
            lower = range_dict['lower'] * 100
            upper = range_dict['upper'] * 100
            
            if level == 'high_detail':
                description = f"{lower}% or more Data Available"
            elif level == 'low_detail':
                description = f"Less than {upper}% Data Available"
            else:
                description = f"{lower}% to less than {upper}% Data Available"
            
            content += f"<h3>{level_name}</h3>{description}"
        
        return content
        
    def generate_vulzoo_repos_content(fields):
        content = "<h2>VulZoo Repository Fields</h2>"
        
        # Group fields by repository
        repos = {}
        for field in fields:
            repo, field_name = field.split(' | ')
            if repo not in repos:
                repos[repo] = []
            repos[repo].append(field_name)
    
        # Generate content for each repository
        for repo, fields in repos.items():
            repo_name = repo.replace('-', ' ').title()
            content += f"<h3>{repo_name}</h3>"
            content += "<ul>"
            for field in fields:
                content += f"<li>{field}</li>"
            content += "</ul>"
        
        return content

    # Color palette for grouping related tiles
    colors = Category10[10]
    min_date, max_date = find_min_max_dates(kv.llms)
    
    # Helper function to create a styled div
    def create_styled_div(text, width, height, border_color):
        return Div(
            text=text,
            width=width,
            height=height,
            styles={
                'background-color': '#f0f0f0',
                'padding': '10px',
                'border': f'2px solid {border_color}',
                'border-radius': '5px',
                'overflow-y': 'auto'
            }
        )
    
    # First row: small tiles
    tile1 = create_styled_div(f'''<h3 style="font-size: 14px;">Number of VULNRICHMENT vulnerabilities</h3>
                                 <p style="font-size: 18px; font-weight: bold;">{vulnrichment_cve_count}</p>
                                 <p style="font-size: 12px;">(as of {latest_datetime})</p>''', 200, 150, colors[0])
    tile2 = create_styled_div("<h3 style='font-size: 14px;'>Confidence Interval</h3><p style='font-size: 18px; font-weight: bold;'>95%</p>", 200, 150, colors[1])
    tile3 = create_styled_div("<h3 style='font-size: 14px;'>Margin of Error</h3><p style='font-size: 18px; font-weight: bold;'>5%</p>", 200, 150, colors[1])
    tile4 = create_styled_div("<h3 style='font-size: 14px;'>Proportion</h3><p style='font-size: 18px; font-weight: bold;'>50%</p>", 200, 150, colors[1])
    tile5 = create_styled_div(f"<h3 style='font-size: 14px;'>Representative Sample Size</h3><p style='font-size: 18px; font-weight: bold;'>{distinct_cve_count}</p>", 200, 150, colors[2])
    
    # Second row: medium and large tiles
    tile6 = create_styled_div(generate_llm_list_html(kv.llms), 250, 300, colors[3])
    tile7 = create_styled_div(generate_prompting_techniques_html(prompt_techniques), 250, 300, colors[4])
    tile8 = create_styled_div(generate_risk_scenarios_html(kv.system_scenario_role), 300, 350, colors[5])
    tile9 = create_styled_div(f"<h3 style='font-size: 14px;'>Number of Trials</h3><p style='font-size: 18px; font-weight: bold;'>{kv.trials}</p>", 200, 250, colors[6])
    tile10 = create_styled_div(f"<h3 style='font-size: 14px;'>Number of Queries</h3><p style='font-size: 18px; font-weight: bold;'>{query_count}</p><p style='font-size: 12px;'>[Representative Sample Size] x [LLMs] x [Prompting Techniques] x [Organization Risk Scenarios] x [Number of Trials]</p>", 200, 250, colors[6])
    
    # Third row: additional tiles
    tile11 = create_styled_div(f"<h3 style='font-size: 14px;'>Pre Cutoff Date</h3><p style='font-size: 18px; font-weight: bold;'>{format_date(min_date, 'pre')}</p><p style='font-size: 12px;'>Pre-cutoff vulnerabilities were published during the training sets of the LLMs</p>", 200, 150, colors[7])
    tile12 = create_styled_div(f"<h3 style='font-size: 14px;'>Post Cutoff Date</h3><p style='font-size: 18px; font-weight: bold;'>{format_date(max_date, 'post')}</p><p style='font-size: 12px;'>Post-cutoff vulnerabilities were published after the training sets of the LLMs</p>", 200, 150, colors[7])
    tile13 = create_styled_div(generate_data_detail_content(kv.data_detail_levels), 300, 300, colors[8])
    tile14 = create_styled_div(generate_vulzoo_repos_content(kv.vulzoo_repos_fields), 400, 600, colors[9])
    
    # Create the layout
    grid_top = gridplot([
        [tile1, tile2, tile3, tile4, tile5],
        [tile6, tile7, tile8, tile9, tile10]
    ], width=300, height=350)
    
    row_bottom = row([tile11, tile12, tile13, tile14], width=400, height=600)
    
    key_details_layout = column([grid_top, row_bottom])
    key_details_tab = TabPanel(child=key_details_layout, title="Key Details")
    tabs.append(key_details_tab)

    # Tab 1: LLM, PT and SDP Performance
    # ====================================================
    harmonic_columns = [TableColumn(field=col, title=col) for col in harmonic_means_df.columns]
    harmonic_source = ColumnDataSource(harmonic_means_df)
    harmonic_table = DataTable(source=harmonic_source, columns=harmonic_columns, width=1200, height=600)
    tabs.append(TabPanel(child=harmonic_table, title="Tab 1: LLM, PT and SDP Performance"))

    # Tab 2: Input Tokens Five Number Summary
    # ====================================================
    stats = df_vuln.groupby('llm')['prompt_tokens'].agg([
            ('Min', 'min'),
            ('Q1', lambda x: x.quantile(0.25)),
            ('Median', 'median'),
            ('Q3', lambda x: x.quantile(0.75)),
            ('Max', 'max')
        ]).reset_index()
    
    output_path = os.path.join(folder_path, f"Step_8_Input_Tokens_Five_Number_Summary.csv")
    stats.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    # Create a ColumnDataSource
    source = ColumnDataSource(stats)
    
    # Create columns for the DataTable
    columns = [
        TableColumn(field="llm", title="LLM"),
        TableColumn(field="Min", title="Min", formatter=NumberFormatter(format="0,0")),
        TableColumn(field="Q1", title="Q1", formatter=NumberFormatter(format="0,0")),
        TableColumn(field="Median", title="Median", formatter=NumberFormatter(format="0,0")),
        TableColumn(field="Q3", title="Q3", formatter=NumberFormatter(format="0,0")),
        TableColumn(field="Max", title="Max", formatter=NumberFormatter(format="0,0"))
    ]
    
    # Create the DataTable
    data_table = DataTable(source=source, columns=columns, width=800, height=280)
    
    # Create a new TabPanel for the DataTable
    new_tab = TabPanel(child=data_table, title="Tab 2: Input Tokens Five Number Summary")
    
    # Add the new tab to your existing tabs
    tabs.append(new_tab)

    # Tab 3: Cutoff Data Performance
    # ====================================================
    def create_data_table(df):
        source = ColumnDataSource(df)
        columns = [TableColumn(field=col, title=col) for col in df.columns]
        return DataTable(source=source, columns=columns, width=1200, height=600)

    # Create DataTables for each dataframe
    pre_cutoff_table = create_data_table(pre_cutoff_df)
    post_cutoff_table = create_data_table(post_cutoff_df)
        
    # Create subtabs
    pre_cutoff_tab = TabPanel(child=pre_cutoff_table, title="Pre-Cutoff Data")
    post_cutoff_tab = TabPanel(child=post_cutoff_table, title="Post-Cutoff Data")
    
    # Create the main tab with subtabs
    cutoff_data_tabs = Tabs(tabs=[pre_cutoff_tab, post_cutoff_tab])
    cutoff_data_panel = TabPanel(child=cutoff_data_tabs, title="Tab 3: Cutoff Data Performance")
    
    # Add the new tab to your existing tabs
    tabs.append(cutoff_data_panel)

    # Tab 4: Data Detail Levels Performance
    # ====================================================
    def create_data_table(df):
        source = ColumnDataSource(df)
        columns = [TableColumn(field=col, title=col) for col in df.columns]
        return DataTable(source=source, columns=columns, width=1200, height=600)

    # Create DataTables for each dataframe
    high_detail_table = create_data_table(high_detail_df)
    medium_level_table = create_data_table(medium_level_df)
    low_detail_table = create_data_table(low_detail_df)
    
    # Create subtabs
    high_detail_tab = TabPanel(child=high_detail_table, title="High Detail Data")
    medium_level_tab = TabPanel(child=medium_level_table, title="Medium Level Data")
    low_detail_tab = TabPanel(child=low_detail_table, title="Low Detail Data")
    
    # Create the main tab with subtabs
    detail_data_tabs = Tabs(tabs=[high_detail_tab, medium_level_tab, low_detail_tab])
    detail_data_panel = TabPanel(child=detail_data_tabs, title="Tab 4: Data Detail Levels Performance")
    
    tabs.append(detail_data_panel)

    # Tab 5: SDP Five Number Summary
    # ====================================================
    harmonic_metrics = ['precision_harmonic_mean', 'recall_harmonic_mean', 'accuracy_harmonic_mean','f1_harmonic_mean', 'f2_harmonic_mean', 'auc_pr_harmonic_mean', 'mcc_harmonic_mean']
    
    # List of aspect types
    ssvc_decision_points = harmonic_means_df['aspect'].unique()

    def create_combined_summary_table(df, metrics, aspects):
    """Create combined summary for all metrics and aspects"""
        
        # Initialize lists to store all data
        all_data = {
            'Aspect': [],
            'Metric': [],
            'Min': [],
            'Q1': [],
            'Median': [],
            'Q3': [],
            'Max': []
        }
        
        # Process each metric for all aspects
        for metric in metrics:
            for aspect in aspects:
                # Get data for this metric and aspect
                aspect_data = df[df['aspect'] == aspect][metric]
                stats = aspect_data.describe()
                
                # Add data to the dictionary
                all_data['Aspect'].append(aspect)
                all_data['Metric'].append(metric.replace('_harmonic_mean', '').upper())
                all_data['Min'].append(round(stats['min'], 4))
                all_data['Q1'].append(round(stats['25%'], 4))
                all_data['Median'].append(round(stats['50%'], 4))
                all_data['Q3'].append(round(stats['75%'], 4))
                all_data['Max'].append(round(stats['max'], 4))
        
        # Create DataFrame and export to TSV
        summary_df = pd.DataFrame(all_data)
        
        # Export to TSV
        output_path = os.path.join(folder_path, "Step_8_SDP_Five_Number_Summary.tsv")
        summary_df.to_csv(output_path, 
                  sep='\t',           # Tab separator
                  index=False,        # Don't include index
                  encoding='utf-8'    # UTF-8 encoding
        )
        
        # Create ColumnDataSource
        source = ColumnDataSource(all_data)
        
        # Define columns for the DataTable
        columns = [
            TableColumn(field='Aspect', title='SDP'),
            TableColumn(field='Metric', title='Metric'),
            TableColumn(field='Min', title='Min'),
            TableColumn(field='Q1', title='Q1'),
            TableColumn(field='Median', title='Median'),
            TableColumn(field='Q3', title='Q3'),
            TableColumn(field='Max', title='Max')
        ]
        
        # Create DataTable
        data_table = DataTable(
            source=source,
            columns=columns,
            width=800,
            height=400,
            index_position=None  # Remove index column
        )
        
        return data_table
    
    # Create the combined summary table
    summary_table = create_combined_summary_table(harmonic_means_df, harmonic_metrics, ssvc_decision_points)
    
    # Add to your existing tabs
    tabs.append(TabPanel(child=summary_table, title="Tab 5: SDP Five Number Summary"))

    # Tab 6: Data Detail Level Five Number Summary
    # ====================================================
    def calculate_box_whisker_stats(df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats = []
        
        for col in numeric_columns:
            col_stats = df[col].agg(['min', 'max', 'median'])
            col_stats['q1'] = df[col].quantile(0.25)
            col_stats['q3'] = df[col].quantile(0.75)
            col_stats = col_stats.reset_index()
            col_stats.columns = ['Statistic', col]
            stats.append(col_stats)
        
        return pd.concat(stats, axis=1)
    
    high_detail_stats = calculate_box_whisker_stats(high_detail_df)
    medium_level_stats = calculate_box_whisker_stats(medium_level_df)
    low_detail_stats = calculate_box_whisker_stats(low_detail_df)
    
    high_detail_stats['Detail Level'] = 'High Detail'
    medium_level_stats['Detail Level'] = 'Medium Level'
    low_detail_stats['Detail Level'] = 'Low Detail'
    
    combined_stats = pd.concat([high_detail_stats, medium_level_stats, low_detail_stats])
    combined_stats = combined_stats.loc[:,~combined_stats.columns.duplicated()]
    
    output_path = os.path.join(folder_path, f"Step_8_Data_Detail_Level_Five_Number_Summary.csv")
    combined_stats.to_csv(output_path, sep='\t', index=False, encoding='utf-8')

    def create_box_whisker_table(df):
        source = ColumnDataSource(df)
        columns = [
            TableColumn(field='Detail Level', title='Detail Level'),
            TableColumn(field='Statistic', title='Statistic')
        ]
        
        for column in df.columns:
            if column not in ['Detail Level', 'Statistic']:
                columns.append(TableColumn(field=column, title=column))
        
        data_table = DataTable(source=source, columns=columns, width=1200, height=400)
        return data_table

    box_whisker_table = create_box_whisker_table(combined_stats)
    
    # Add the new tab to the existing detail_data_tabs
    box_whisker_tab = TabPanel(child=box_whisker_table, title="Tab 6: Data Detail Level Five Number Summary")
    # Add the new tab to your existing tabs
    tabs.append(box_whisker_tab)

    # Tab 7: Cutoff Status Five Number Summary
    # ====================================================
    def create_combined_summary(pre_df, post_df, metrics):
        """Create combined summary for all metrics and both cutoff periods"""
        
        # Initialize lists to store all data
        all_data = {
            'Metric': [],
            'Cutoff_Status': [],
            'Min': [],
            'Q1': [],
            'Median': [],
            'Q3': [],
            'Max': []
        }
        
        # Process each metric for both pre and post cutoff
        for metric in metrics:
            # Pre-cutoff statistics
            pre_stats = pre_df[metric].describe()
            all_data['Metric'].append(metric.replace('_harmonic_mean', '').upper())
            all_data['Cutoff_Status'].append('Pre-Cutoff')
            all_data['Min'].append(round(pre_stats['min'], 4))
            all_data['Q1'].append(round(pre_stats['25%'], 4))
            all_data['Median'].append(round(pre_stats['50%'], 4))
            all_data['Q3'].append(round(pre_stats['75%'], 4))
            all_data['Max'].append(round(pre_stats['max'], 4))
            
            # Post-cutoff statistics
            post_stats = post_df[metric].describe()
            all_data['Metric'].append(metric.replace('_harmonic_mean', '').upper())
            all_data['Cutoff_Status'].append('Post-Cutoff')
            all_data['Min'].append(round(post_stats['min'], 4))
            all_data['Q1'].append(round(post_stats['25%'], 4))
            all_data['Median'].append(round(post_stats['50%'], 4))
            all_data['Q3'].append(round(post_stats['75%'], 4))
            all_data['Max'].append(round(post_stats['max'], 4))
        
        # Create ColumnDataSource
        source = ColumnDataSource(all_data)
        
        # Export to TSV
        cutoff_status_five_number_summaries = pd.DataFrame(all_data)

        # Export to TSV
        cutoff_status_five_number_summaries.to_csv('Step_8_Cutoff_Status_Five_Number_Summary.tsv', 
                  sep='\t',           # Tab separator
                  index=False,        # Don't include index
                  encoding='utf-8'    # UTF-8 encoding
        )
        
        # Define columns for the DataTable
        columns = [
            TableColumn(field='Cutoff_Status', title='Cutoff Status'),
            TableColumn(field='Metric', title='Metric'),
            TableColumn(field='Min', title='Min'),
            TableColumn(field='Q1', title='Q1'),
            TableColumn(field='Median', title='Median'),
            TableColumn(field='Q3', title='Q3'),
            TableColumn(field='Max', title='Max')
        ]
        
        # Create DataTable
        data_table = DataTable(
            source=source,
            columns=columns,
            width=800,
            height=400,
            index_position=None  # Remove index column
        )
        
        return data_table
    
    # Create the combined summary table
    summary_table = create_combined_summary(pre_cutoff_df, post_cutoff_df, harmonic_metrics)
    
    # Add to your existing tabs
    tabs.append(TabPanel(child=summary_table, title="Tab 7: Cutoff Status Five Number Summary"))

    # Tab 8: LLM Five Number Summary
    # ====================================================
    def create_harmonic_summary_table(harmonic_means_df, harmonic_metrics):
        # Initialize empty lists to store results
        rows = []
        
        # Calculate summary statistics for each metric
        for metric in harmonic_metrics:
            # Calculate basic statistics
            stats = harmonic_means_df.groupby('llm')[metric].agg(['min', 'median', 'max']).reset_index()
            
            # Calculate Q1 and Q3 separately
            q1 = harmonic_means_df.groupby('llm')[metric].quantile(0.25).reset_index()
            q3 = harmonic_means_df.groupby('llm')[metric].quantile(0.75).reset_index()
            
            # Merge all statistics
            stats = stats.merge(q1, on='llm', suffixes=('', '_q1'))
            stats = stats.merge(q3, on='llm', suffixes=('', '_q3'))
            
            # Rename columns
            stats.columns = ['llm', 'Min', 'Median', 'Max', 'Q1', 'Q3']
            
            # Add metric name
            stats['Metric'] = metric.replace('_harmonic_mean', '').title()
            
            # Reorder columns to desired format
            stats = stats[['llm', 'Metric', 'Min', 'Q1', 'Median', 'Q3', 'Max']]
            
            rows.append(stats)
        
        # Combine all results
        summary = pd.concat(rows, ignore_index=True)
        
        # Save to file
        output_path = os.path.join(folder_path, f"Step_8_LLM_Five_Number_Summary.csv")
        summary.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        
        # Create ColumnDataSource
        source = ColumnDataSource(summary)
        
        # Define columns for the DataTable
        columns = [
            TableColumn(field='llm', title='LLM'),
            TableColumn(field='Metric', title='Metric'),
            TableColumn(field='Min', title='Min'),
            TableColumn(field='Q1', title='Q1'),
            TableColumn(field='Median', title='Median'),
            TableColumn(field='Q3', title='Q3'),
            TableColumn(field='Max', title='Max')
        ]
        
        # Create DataTable
        data_table = DataTable(
            source=source, 
            columns=columns, 
            width=1000, 
            height=400,
            index_position=None
        )
        
        # Create a new TabPanel for the DataTable
        new_tab = TabPanel(child=data_table, title="Tab 8: LLM Five Number Summary")
        
        return new_tab

    new_tab = create_harmonic_summary_table(harmonic_means_df, harmonic_metrics)
    tabs.append(new_tab)

    # Tab 9: PT Five Number Summary
    # ====================================================
    def create_harmonic_summary_table(harmonic_means_df, harmonic_metrics):
        # Initialize empty lists to store results
        rows = []
        
        # Calculate summary statistics for each metric
        for metric in harmonic_metrics:
            # Calculate basic statistics
            stats = harmonic_means_df.groupby('prompt')[metric].agg(['min', 'median', 'max']).reset_index()
            
            # Calculate Q1 and Q3 separately
            q1 = harmonic_means_df.groupby('prompt')[metric].quantile(0.25).reset_index()
            q3 = harmonic_means_df.groupby('prompt')[metric].quantile(0.75).reset_index()
            
            # Merge all statistics
            stats = stats.merge(q1, on='prompt', suffixes=('', '_q1'))
            stats = stats.merge(q3, on='prompt', suffixes=('', '_q3'))
            
            # Rename columns
            stats.columns = ['Prompt', 'Min', 'Median', 'Max', 'Q1', 'Q3']
            
            # Add metric name
            stats['Metric'] = metric.replace('_harmonic_mean', '').title()
            
            rows.append(stats)
    
        # Combine all results
        summary = pd.concat(rows, ignore_index=True)
        
        # Save to file
        output_path = os.path.join(folder_path, f"Step_8_PT_Five_Number_Summary.csv")
        summary.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        
        # Create ColumnDataSource
        source = ColumnDataSource(summary)
        
        # Define columns for the DataTable
        columns = [
            TableColumn(field='Prompt', title='Prompt'),
            TableColumn(field='Metric', title='Metric'),
            TableColumn(field='Min', title='Min', formatter=NumberFormatter(format="0.000")),
            TableColumn(field='Q1', title='Q1', formatter=NumberFormatter(format="0.000")),
            TableColumn(field='Median', title='Median', formatter=NumberFormatter(format="0.000")),
            TableColumn(field='Q3', title='Q3', formatter=NumberFormatter(format="0.000")),
            TableColumn(field='Max', title='Max', formatter=NumberFormatter(format="0.000"))
        ]
        
        # Create DataTable
        data_table = DataTable(
            source=source, 
            columns=columns, 
            width=1000, 
            height=400,
            index_position=None
        )
        
        # Create a new TabPanel for the DataTable
        new_tab = TabPanel(child=data_table, title="Tab 9: PT Five Number Summary")
        
        return new_tab

    # Create the harmonic metrics summary table and add it to tabs
    harmonic_metrics = [
        'precision_harmonic_mean', 'recall_harmonic_mean', 'accuracy_harmonic_mean',
        'f1_harmonic_mean', 'f2_harmonic_mean', 'auc_pr_harmonic_mean', 'mcc_harmonic_mean'
    ]
    
    new_tab = create_harmonic_summary_table(harmonic_means_df, harmonic_metrics)
    tabs.append(new_tab)

    # Generate Dashboard
    # ====================================================
    dashboard = Tabs(tabs=tabs)
    save(dashboard)


    
    

