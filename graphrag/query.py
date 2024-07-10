from graphrag.query.cli import run_local_search
import argparse
import time
# Example usage

def main():
    parser = argparse.ArgumentParser(description='Process a query string.')
    parser.add_argument('query', type=str, help='The query string to process')
    
    args = parser.parse_args()
    
    query = args.query
    print(f"Received query: {query}")
    
    start_time = time.time()
    run_local_search(data_dir="/home/adamtay/graphrag-accelerator/graphrag/output/20240710-163051/artifacts",
                    root_dir="/home/adamtay/graphrag-accelerator/graphrag",
                    community_level=2,
                    response_type="clear and conciose", # Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report
                    query=query)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()