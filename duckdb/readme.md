# Introduction

The explanation of what this does is described in this [article](https://www.linkedin.com/pulse/using-openai-embeddings-data-engineering-fabric-miguel-peredo-z%C3%BCrcher/)

The original article refers to running the embeddings in a Fabric Notebook, now this is a python program you can run on your machine and uses duckdb to store and search vectors.

## Requirements

- You need an Embedding Model in Azure OpenAI of type text-embedding-3-large and the corresponding key
- Python 3.9 or higher

## How to run 

### Create virtual enviroment

0. Create a .env file, this file should contain the following: 
(This step is mandatory, create this file before deploying to Azure)

    ```
    openai.api_type = "azure"
    AZURE_API_KEY= yourkey
    openai.api_version = "2024-02-15-preview"
    AZURE_OPENAI_API_KEY = yourkey
    AZURE_OPENAI_ENDPOINT = "https://yoururl.openai.azure.com/"
    ADA = "emb3l"
    ```
Note: the value of ADA is the name of the deployment of a model of type: text-embedding-3-large

1. Create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) and activate it. 
    You can name the environment .venv for example: 
    ```log
    python -m venv .venv
    ```
    This name .venv keeps the directory typically hidden in your shell and thus out of the way while giving it a name that explains why the directory exists. 
    Activate it following the instructions from the [link](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments).

2. Run the command below to install the necessary requirements.

    ```log
    python3 -m pip install -r requirements.txt
     ```

3. Make sure you are in the folder duckdb run
   
    ```log
    python main.py
     ```
   

