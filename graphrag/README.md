To run the graph rag library:

`conda create -n graphrag python=3.10`

`pip install graphrag`

`python -m graphrag.index --init --root ./graphrag`

- output/, prompts/, settings.yaml will be created. Insert your openai api key in `GRAPHRAG_API_KEY`.

To create knowledge graph:
- Configure settings in settings.yaml, api type, models, chunk size, file type, etc.

`cd graphrag`

`python -m graphrag.index --root .`

- Results will be saved into ./graphrag/output/datetime. Contains some log files and multiple .parquet files.

To Query:

 - Global mode:

`
python -m graphrag.query \
--root ./graphrag \
--method global \
"What are the three ai engines?"
`



Supports local models.
Can be used locally or with azure resource group.