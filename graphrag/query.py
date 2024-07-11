from graphrag.query.cli import *
from graphrag.query.cli import _configure_paths_and_settings, __get_embedding_description_store

from graphrag.vector_stores import VectorStoreType
import argparse
import time
from myFactories import *
# Example usage

# Modify:
# /home/adamtay/miniforge3/envs/graphrag/lib/python3.10/site-packages/graphrag/query/structured_search/local_search/search.py

def main(query):
    # parser = argparse.ArgumentParser(description='Process a query string.')
    # parser.add_argument('query', type=str, help='The query string to process')
    
    # args = parser.parse_args()
    
    # query = args.query
    print(f"Received query: {query}")
    
    start_time = time.time()
    response, context_text = my_run_global_search(data_dir="/home/adamtay/graphrag-accelerator/graphrag/output/20240710-163051/artifacts",
                    root_dir="/home/adamtay/graphrag-accelerator/graphrag",
                    community_level=2,
                    response_type="clear and concise", # Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report
                    query=query)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return response, context_text
    
def my_run_local_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a local search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_community_reports = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )
    final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
    final_relationships = pd.read_parquet(
        data_path / "create_final_relationships.parquet"
    )
    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
    final_covariates_path = data_path / "create_final_covariates.parquet"
    final_covariates = (
        pd.read_parquet(final_covariates_path)
        if final_covariates_path.exists()
        else None
    )

    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )
    vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

    description_embedding_store = __get_embedding_description_store(
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )
    covariates = (
        read_indexer_covariates(final_covariates)
        if final_covariates is not None
        else []
    )

    search_engine = my_get_local_search_engine(
        config,
        reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        ),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )
    result, context_text = search_engine.search(query=query)
    print("LLm Calls:", result.llm_calls)
    # print("Contexts: ", context_text)
    reporter.success(f"Local Search Response: {result.response}")
    return result.response, context_text

def my_run_global_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a global search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    final_nodes: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_nodes.parquet"
    )
    final_entities: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_entities.parquet"
    )
    final_community_reports: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )

    reports = read_indexer_reports(
        final_community_reports, final_nodes, community_level
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    search_engine = my_get_global_search_engine(
        config,
        reports=reports,
        entities=entities,
        response_type=response_type,
    )
    
    result, context_text = search_engine.search(query=query)
    # print("Contexts: ", context_text)
    print("LLM Calls:", result.llm_calls)
    reporter.success(f"Global Search Response: {result.response}")
    return result.response, context_text

import json
def parse_dataset(dataset_path):
    # Load the JSON data
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    # Initialize a list to hold the parsed data
    query_set = []
    
    # Iterate through the dataset
    for item in data['dataset']:
        # Get the document name (assuming each item has only one document)
        documents = item['docs']
        name = item['name']
        # Iterate through the tasks
        for task in item['tasks']:
            # Extract the query and reference answer
            query = task['query']
            ref_answer = task['ref_answer']
            
            # Combine into a dictionary and append to the list
            query_set.append({
                'Dataset': name,
                'Document': documents,
                'Query': query,
                'ref_answer': ref_answer
            })

    return query_set

import pandas as pd
from datetime import datetime
from openpyxl import load_workbook
from format import format_xlsx
if __name__ == "__main__":
    query_set = parse_dataset('/home/adamtay/graphrag-accelerator/graphrag/sa_dataset_v2.json')
    today = datetime.now().strftime('%Y%m%d_%H%M')
    df = pd.DataFrame()
    for query in query_set:
        response, context = main(query["Query"])
        result_df = pd.DataFrame({"context_chunks": [context], "response": [response]})
        query_df = pd.DataFrame([query])
        
        combined_df = pd.concat([query_df, result_df], axis=1)
        df = pd.concat([df, combined_df], ignore_index=True)
    
    # Define the Excel file name
    excel_path = f'results/{today}'
    if os.path.exists(excel_path):
        print(f"Directory {excel_path} exists")
    else: 
        os.makedirs(excel_path)
        print(f"Directory {excel_path} created")
    excel_file = f'{excel_path}/graphrag-sa.xlsx'
    # Check if the Excel file exists
    file_exists = os.path.isfile(excel_file)
    
    # Writing to the Excel file
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        print(f"Data successfully written to {excel_file}")
        print("Existing sheets: \n", writer.sheets)

    except:
        print(f"Error writing to {excel_file}")
        
    print("Formatting sheet...")
    format_xlsx(excel_file)