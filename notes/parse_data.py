#!/usr/bin/env python3

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

# Suppress FutureWarning about fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')


def load_benchmark_results(base_path):
    results = defaultdict(list)
    
    # Find all database folders (postgresql*, redis)
    base_path_obj = Path(base_path)
    db_folders = []
    
    # # # Add all postgresql* folders
    # postgresql_folders = list(base_path_obj.glob('postgresql*'))
    # for pg_folder in postgresql_folders:
    #     if pg_folder.is_dir():
    #         db_folders.append(pg_folder.name)
    
    # # Add redis folder
    # if (base_path_obj / 'redis-gcc13').exists():
    #     db_folders.append('redis-gcc13')

    # # Add redis folder
    # if (base_path_obj / 'redis').exists():
    #     db_folders.append('redis')
    
    # # # Add elasticsearch folder
    # if (base_path_obj / 'elasticsearch').exists():
    #     db_folders.append('elasticsearch')

    # # # Add qdrant folder
    if (base_path_obj / 'qdrant').exists():
        db_folders.append('qdrant')

    print(f"Found database folders: {db_folders}")
    
    # Process all found database folders
    for db_type in db_folders:
        db_path = Path(base_path) / db_type
        if not db_path.exists():
            print(f"Warning: Folder {db_path} does not exist")
            continue
            
        # Process all architecture folders (ARM, Intel, Intel-2xlarge, etc.)
        for arch_folder in db_path.iterdir():
            if arch_folder.is_dir():
                arch = arch_folder.name
                
                # Load all JSON files (skip summary)
                for json_file in arch_folder.glob('*.json'):
                    if 'summary' in json_file.name:
                        continue
                    
                    try:
                        # Try with utf-8-sig to handle BOM
                        with open(json_file, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)
                            
                        # Add file information
                        data['file_path'] = str(json_file)
                        data['file_name'] = json_file.name
                        data['db_type'] = db_type
                        data['architecture'] = arch
                        
                        # Determine test type (search/upload)  
                        if 'upload' in json_file.name:
                            data['test_type'] = 'upload'
                        elif 'search' in json_file.name:
                            data['test_type'] = 'search'
                        else:
                            data['test_type'] = 'unknown'
                            
                        results[f"{db_type}_{arch}"].append(data)
                        
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                    
    return results


def extract_hnsw_params(data):
    """Extract HNSW parameters (m, ef_construct) from the data"""
    params = data.get('params', {})
    hnsw_config = params.get('hnsw_config', {})
    index_options = params.get('index_options', {})  # For Elasticsearch
    
    # For Redis: M and EF_CONSTRUCTION
    # For PostgreSQL: m and ef_construct  
    # For Elasticsearch: m and ef_construction in index_options
    m_value = (hnsw_config.get('M') or hnsw_config.get('m') or 
               index_options.get('m'))
    ef_value = (hnsw_config.get('EF_CONSTRUCTION') or hnsw_config.get('ef_construct') or 
                index_options.get('ef_construction'))
    
    if m_value is not None and ef_value is not None:
        return f"M{m_value}E{ef_value}"
    
    # Fallback to experiment name if HNSW params not found
    experiment = params.get('experiment', 'unknown')
    return experiment.replace('dbpedia-cal-', '').replace('-dbpedia-openai-1M-1536-angular', '')


def extract_test_case_key(data):
    params = data.get('params', {})
    
    # Extract data type from experiment name or search_params
    data_type = 'unknown'
    experiment = params.get('experiment', '')
    if 'float32' in experiment:
        data_type = 'float32'
    elif 'float16' in experiment:
        data_type = 'float16'
    else:
        # Try to get from search_params
        search_params = params.get('search_params', {})
        if 'data_type' in search_params:
            data_type = search_params['data_type'].lower()
    
    # For search tests
    if data['test_type'] == 'search':
        calibration_precision = params.get('calibration_precision', 'unknown')
        parallel = params.get('parallel', 'unknown')
        top = params.get('top', 'unknown')
        experiment = params.get('experiment', 'unknown')
        
        return ('search', data_type, experiment, calibration_precision, parallel, top)
    
    # For upload tests
    elif data['test_type'] == 'upload':
        parallel = params.get('parallel', 'unknown')
        experiment = params.get('experiment', 'unknown')
        
        return ('upload', data_type, experiment, parallel)
    
    return ('unknown',)


def find_best_rps_for_search_tests(results_data):
    # Group search tests by test case keys
    search_groups = defaultdict(list)
    
    for data in results_data:
        if data['test_type'] == 'search' and 'rps' in data.get('results', {}):
            key = extract_test_case_key(data)
            search_groups[key].append(data)
    
    # Find best RPS for each case
    best_results = {}
    
    for test_case_key, test_runs in search_groups.items():
        if len(test_runs) == 0:
            continue
            
        # Find run with highest RPS
        best_run = max(test_runs, key=lambda x: x['results'].get('rps', 0))
        best_results[test_case_key] = best_run
        
        print(f"  Case {test_case_key}: {len(test_runs)} runs, best RPS: {best_run['results']['rps']:.2f}")
    
    return best_results


def create_results_dataframe(all_results):
    db_dataframes = {}
    
    # First, create a mapping of experiment -> hnsw_config from upload files
    experiment_to_hnsw = {}
    for key, results_list in all_results.items():
        for result in results_list:
            if result['test_type'] == 'upload':
                params = result['params']
                experiment = params.get('experiment', '')
                hnsw_config = params.get('hnsw_config', {})
                index_options = params.get('index_options', {})  # For Elasticsearch
                
                # Handle Redis (M, EF_CONSTRUCTION), PostgreSQL (m, ef_construct), and Elasticsearch (m, ef_construction in index_options)
                m_value = hnsw_config.get('M') or hnsw_config.get('m') or index_options.get('m', 'unknown')
                ef_construct_value = hnsw_config.get('EF_CONSTRUCTION') or hnsw_config.get('ef_construct') or index_options.get('ef_construction', 'unknown')
                if m_value != 'unknown' and ef_construct_value != 'unknown':
                    experiment_to_hnsw[experiment] = f"M{m_value}E{ef_construct_value}"
    
    for key, results_list in all_results.items():
        parts = key.split('_')
        db_type = parts[0]
        arch = '_'.join(parts[1:])  # Handle cases like postgresql-biggersharedbuff_Intel
        
        if db_type not in db_dataframes:
            db_dataframes[db_type] = []
        
        # Find best results for search tests
        best_search = find_best_rps_for_search_tests(results_list)
        
        # Add upload tests (without grouping)
        upload_tests = [data for data in results_list if data['test_type'] == 'upload']
        
        # Prepare data for DataFrame
        for test_case_key, best_result in best_search.items():
            params = best_result['params']
            results = best_result['results']
            
            # Extract data type and experiment from test case key
            data_type = test_case_key[1] if len(test_case_key) > 1 else 'unknown'
            experiment = test_case_key[2] if len(test_case_key) > 2 else 'unknown'
            
            # Extract HNSW parameters for the identifier
            hnsw_params = extract_hnsw_params(best_result)
            
            # Get m and ef_construct values from the experiment mapping
            experiment = test_case_key[2] if len(test_case_key) > 2 else 'unknown'
            hnsw_m_ef = experiment_to_hnsw.get(experiment, 'MunknownEunknown')
            
            row_data = {
                'db_config': db_type,  # e.g., 'postgresql' or 'postgresql-biggersharedbuff'
                'architecture': arch,
                'test_type': 'search',
                'data_type': data_type,
                'experiment': experiment,
                'hnsw_params': hnsw_params,
                'hnsw_m_ef': hnsw_m_ef,
                'calibration_precision': params.get('calibration_precision', 'unknown'),
                'parallel': params.get('parallel', 'unknown'),
                'top': params.get('top', 'unknown'),
                'rps': results.get('rps', 0),
                'mean_time': results.get('mean_time', 0),
                'mean_precisions': results.get('mean_precisions', 0),
                'p99_time': results.get('p99_time', 0),
                'file_name': best_result['file_name']
            }
            
            db_dataframes[db_type].append(row_data)
        
        # Add upload tests
        for upload_result in upload_tests:
            params = upload_result['params']
            results = upload_result['results']
            
            # Extract data type and experiment for upload test
            experiment = params.get('experiment', '')
            data_type = 'unknown'
            if 'float32' in experiment:
                data_type = 'float32'
            elif 'float16' in experiment:
                data_type = 'float16'
            
            # Extract HNSW parameters for the identifier
            hnsw_params = extract_hnsw_params(upload_result)
            
            # Get m and ef_construct values
            hnsw_config = params.get('hnsw_config', {})
            index_options = params.get('index_options', {})  # For Elasticsearch
            
            # Handle Redis (M, EF_CONSTRUCTION), PostgreSQL (m, ef_construct), and Elasticsearch (m, ef_construction in index_options)
            m_value = hnsw_config.get('M') or hnsw_config.get('m') or index_options.get('m', 'unknown')
            ef_construct_value = hnsw_config.get('EF_CONSTRUCTION') or hnsw_config.get('ef_construct') or index_options.get('ef_construction', 'unknown')
            hnsw_m_ef = f"M{m_value}E{ef_construct_value}"
            
            row_data = {
                'db_config': db_type,
                'architecture': arch,
                'test_type': 'upload',
                'data_type': data_type,
                'experiment': experiment,
                'hnsw_params': hnsw_params,
                'hnsw_m_ef': hnsw_m_ef,
                'calibration_precision': 'N/A',
                'parallel': params.get('parallel', 'unknown'),
                'rps': 'N/A',  # Upload tests don't have RPS
                'mean_time': 'N/A',
                'mean_precisions': 'N/A',
                'total_time': results.get('total_time', 0),
                'file_name': upload_result['file_name']
            }
            
            db_dataframes[db_type].append(row_data)
    
    # Convert lists to DataFrames
    result_dfs = {}
    for db_type, data_list in db_dataframes.items():
        if data_list:
            result_dfs[db_type] = pd.DataFrame(data_list)
        else:
            result_dfs[db_type] = pd.DataFrame()
    
    return result_dfs


def print_combined_results_table(db_dataframes):
    # Combine all search data
    all_search_data = []
    
    # Add data from all database configurations
    for db_type, df in db_dataframes.items():
        if not df.empty:
            search_data = df[df['test_type'] == 'search'].copy()
            if not search_data.empty:
                # Create simple db_arch identifier (for search tests)
                search_data['db_arch'] = search_data['db_config'] + '-' + search_data['architecture']
                all_search_data.append(search_data)
    
    if not all_search_data:
        print("No search data found!")
        return
    
    combined_df = pd.concat(all_search_data, ignore_index=True)
    
    # Get unique data types, top values, and HNSW configurations
    data_types = sorted(combined_df['data_type'].unique())
    top_values = sorted(combined_df['top'].unique())
    hnsw_configs = sorted(combined_df['hnsw_m_ef'].unique())
    
    # Create separate tables for each data type, top value, and HNSW config
    for data_type in data_types:
        if data_type == 'unknown':
            continue
        
        for top_value in top_values:
            if top_value == 'unknown':
                continue
            
            for hnsw_config in hnsw_configs:
                if hnsw_config == 'unknown':
                    continue
                    
                print(f"\n{'='*120}")
                print(f"(RPS) - {data_type.upper()} TOP={top_value} {hnsw_config}")
                print(f"{'='*120}")
                
                # Filter data for this data type, top value, and HNSW config
                type_df = combined_df[(combined_df['data_type'] == data_type) & 
                                      (combined_df['top'] == top_value) & 
                                      (combined_df['hnsw_m_ef'] == hnsw_config)].copy()
                
                if type_df.empty:
                    print(f"No data found for {data_type} with top={top_value} and {hnsw_config}")
                    continue
                
                # Create column names for each combination
                type_df['combination'] = type_df.apply(
                    lambda row: f"precision_{row['calibration_precision']}_parallel_{row['parallel']}", 
                    axis=1
                )
                
                # Create pivot table
                pivot_table = type_df.pivot_table(
                    index='db_arch',
                    columns='combination',
                    values='rps',
                    aggfunc='first'
                ).fillna('-')
                
                # Sort columns by precision then parallel
                column_order = []
                unique_combinations = type_df[['calibration_precision', 'parallel']].drop_duplicates()
                unique_combinations = unique_combinations.sort_values(['calibration_precision', 'parallel'])
                
                for _, row in unique_combinations.iterrows():
                    col_name = f"precision_{row['calibration_precision']}_parallel_{row['parallel']}"
                    if col_name in pivot_table.columns:
                        column_order.append(col_name)
                
                pivot_table = pivot_table[column_order]
                
                # Format numbers to 2 decimal places
                for col in pivot_table.columns:
                    pivot_table[col] = pivot_table[col].apply(
                        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x != '-' else x
                    )
                
                # Sort rows: Qdrant, Elasticsearch, Redis, Redis-gcc13, PostgreSQL
                all_rows = list(pivot_table.index)
                qdrant_rows = [row for row in all_rows if row.startswith('qdrant')]
                elasticsearch_rows = [row for row in all_rows if row.startswith('elasticsearch')]
                redis_gcc13_rows = [row for row in all_rows if row.startswith('redis-gcc13')]
                redis_rows = [row for row in all_rows if row.startswith('redis') and not row.startswith('redis-gcc13')]
                postgresql_rows = [row for row in all_rows if row.startswith('postgresql')]
                
                # Sort each group
                qdrant_rows.sort()
                elasticsearch_rows.sort()
                redis_gcc13_rows.sort()
                redis_rows.sort()
                postgresql_rows.sort()
                
                # Combine in desired order
                row_order = qdrant_rows + elasticsearch_rows + redis_rows + redis_gcc13_rows + postgresql_rows
                
                # Add any remaining rows
                seen_rows = set(row_order)
                remaining_rows = [row for row in all_rows if row not in seen_rows]
                remaining_rows.sort()
                row_order.extend(remaining_rows)
                
                pivot_table = pivot_table.reindex(row_order)
                
                # Transpose the table (swap rows and columns)
                pivot_table = pivot_table.T
                
                print(pivot_table.to_string())
    
    # Create p99_time tables for each data type, top value, and HNSW config
    for data_type in data_types:
        if data_type == 'unknown':
            continue
        
        for top_value in top_values:
            if top_value == 'unknown':
                continue
            
            for hnsw_config in hnsw_configs:
                if hnsw_config == 'unknown':
                    continue
                    
                print(f"\n{'='*120}")
                print(f"(P99_TIME) {data_type.upper()} TOP={top_value} {hnsw_config}")
                print(f"{'='*120}")
                
                # Filter data for this data type, top value, and HNSW config
                type_df = combined_df[(combined_df['data_type'] == data_type) & 
                                      (combined_df['top'] == top_value) & 
                                      (combined_df['hnsw_m_ef'] == hnsw_config)].copy()
                
                if type_df.empty:
                    print(f"No data found for {data_type} with top={top_value} and {hnsw_config}")
                    continue
                
                # Create column names for each combination
                type_df['combination'] = type_df.apply(
                    lambda row: f"precision_{row['calibration_precision']}_parallel_{row['parallel']}", 
                    axis=1
                )
                
                # Create pivot table for p99_time
                pivot_table_p99 = type_df.pivot_table(
                    index='db_arch',
                    columns='combination',
                    values='p99_time',
                    aggfunc='first'
                ).fillna('-')
                
                # Sort columns by precision then parallel
                column_order = []
                unique_combinations = type_df[['calibration_precision', 'parallel']].drop_duplicates()
                unique_combinations = unique_combinations.sort_values(['calibration_precision', 'parallel'])
                
                for _, row in unique_combinations.iterrows():
                    col_name = f"precision_{row['calibration_precision']}_parallel_{row['parallel']}"
                    if col_name in pivot_table_p99.columns:
                        column_order.append(col_name)
                
                pivot_table_p99 = pivot_table_p99[column_order]
                
                # Format numbers to 4 decimal places for time values
                for col in pivot_table_p99.columns:
                    pivot_table_p99[col] = pivot_table_p99[col].apply(
                        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x != '-' else x
                    )
                
                # Sort rows: Qdrant, Elasticsearch, Redis, Redis-gcc13, PostgreSQL
                all_rows = list(pivot_table_p99.index)
                qdrant_rows = [row for row in all_rows if row.startswith('qdrant')]
                elasticsearch_rows = [row for row in all_rows if row.startswith('elasticsearch')]
                redis_gcc13_rows = [row for row in all_rows if row.startswith('redis-gcc13')]
                redis_rows = [row for row in all_rows if row.startswith('redis') and not row.startswith('redis-gcc13')]
                postgresql_rows = [row for row in all_rows if row.startswith('postgresql')]
                
                # Sort each group
                qdrant_rows.sort()
                elasticsearch_rows.sort()
                redis_gcc13_rows.sort()
                redis_rows.sort()
                postgresql_rows.sort()
                
                # Combine in desired order
                row_order = qdrant_rows + elasticsearch_rows + redis_rows + redis_gcc13_rows + postgresql_rows
                
                # Add any remaining rows
                seen_rows = set(row_order)
                remaining_rows = [row for row in all_rows if row not in seen_rows]
                remaining_rows.sort()
                row_order.extend(remaining_rows)
                
                pivot_table_p99 = pivot_table_p99.reindex(row_order)
                
                # Transpose the table (swap rows and columns)
                pivot_table_p99 = pivot_table_p99.T
                
                print(pivot_table_p99.to_string())
    
    # Upload summary tables by data type and HNSW config
    all_upload_data = []
    
    # Collect all upload data
    for db_type, df in db_dataframes.items():
        if not df.empty:
            upload_df_filtered = df[df['test_type'] == 'upload'].copy()
            if not upload_df_filtered.empty:
                # Create db_arch identifier without HNSW parameters (they will be used for grouping)
                upload_df_filtered['db_arch'] = (upload_df_filtered['db_config'] + '-' + 
                                               upload_df_filtered['architecture'])
                all_upload_data.append(upload_df_filtered)
    
    if all_upload_data:
        combined_upload_df = pd.concat(all_upload_data, ignore_index=True)
        upload_data_types = sorted(combined_upload_df['data_type'].unique())
        upload_hnsw_configs = sorted(combined_upload_df['hnsw_m_ef'].unique())
        
        for data_type in upload_data_types:
            if data_type == 'unknown':
                continue
            
            for hnsw_config in upload_hnsw_configs:
                if hnsw_config == 'MunknownEunknown':
                    continue
                    
                print(f"\n{'='*60}")
                print(f"(UPLOAD) - {data_type.upper()} - {hnsw_config}")
                print(f"{'='*60}")
                
                type_upload_df = combined_upload_df[(combined_upload_df['data_type'] == data_type) & 
                                                     (combined_upload_df['hnsw_m_ef'] == hnsw_config)].copy()
                
                if not type_upload_df.empty:
                    # Create pivot table with db_arch as columns
                    display_data = type_upload_df[['db_arch', 'total_time']].copy()
                    display_data['total_time_seconds'] = display_data['total_time'].apply(lambda x: f"{x:.2f}")
                    display_data = display_data[['db_arch', 'total_time_seconds']].sort_values(['db_arch'])
                    
                    # Set db_arch as index and transpose
                    display_data = display_data.set_index('db_arch')
                    display_data = display_data.T
                    
                    print(display_data.to_string())
                else:
                    print(f"No upload data found for {data_type} with {hnsw_config}")
    else:
        print("No upload data found")
    
    print(f"\n{'='*60}")
    print("COPY-PASTE INSTRUCTIONS FOR EXCEL:")
    print("1. Select and copy the table data above")
    print("2. Paste into Excel as 'Text' to preserve formatting")
    print("3. Use 'Text to Columns' with space delimiter if needed")
    print(f"{'='*60}")


def main():
    print("Vector-db-benchmark results analysis")
    print("=" * 60)
    
    # Base path (current directory)
    base_path = "."
    
    # Load all results
    print("Loading results...")
    all_results = load_benchmark_results(base_path)
    
    if not all_results:
        print("No data to analyze!")
        return
    
    print(f"Loaded results from {len(all_results)} folders")
    
    # Display summary of loaded data
    for key, results_list in all_results.items():
        search_count = len([r for r in results_list if r['test_type'] == 'search'])
        upload_count = len([r for r in results_list if r['test_type'] == 'upload'])
        print(f"  {key}: {search_count} search, {upload_count} upload")
    
    print("\nFinding best RPS results for search tests...")
    
    # Create DataFrames
    db_dataframes = create_results_dataframe(all_results)
    
    # Display results
    print_combined_results_table(db_dataframes)
    
    print(f"\nAnalysis completed!")


if __name__ == "__main__":
    main()