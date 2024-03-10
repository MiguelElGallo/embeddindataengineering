from dotenv import load_dotenv
import pandas as pd
import os
from langchain_openai import AzureOpenAIEmbeddings
import openai
import duckdb
import numpy as np


def create_new_concatenate_column(records: pd.DataFrame):
    if records is None:
        return None
    # Create a new column
    records["concatenated"] = records["product_hier"].astype(str) + records[
        "description"
    ].astype(str)
    return records


def read_dairy_prods(filename):
    # Read as pandas dataframe
    df = pd.read_csv(filename)
    return df


def init_azureopenai():

    load_dotenv()
    os.environ["openai.api_type"] = os.getenv("openai.api_type")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

    global api_version
    api_version = os.getenv("openai.api_version")

    global ada_deployed_model
    ada_deployed_model = os.getenv("ADA")

    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=ada_deployed_model,
        openai_api_version=api_version,  # e.g., "2023-07-01-preview"
    )
    return aoai_embeddings


def embed(df, aoai_embeddings: AzureOpenAIEmbeddings):
    # We have two important fields id and concatenated
    # We will embed the concatenated field
    # The aoai_embeddings of type OpenAIEmbeddings has method embed_documents
    # The input is a list of strings
    # Lets get the list of strings from the Data frame, field concatenated
    # and then call the method
    list_of_strings = df["concatenated"].tolist()
    # Embed the list of strings
    embeddings = aoai_embeddings.embed_documents(list_of_strings)
    # Add the embeddings to the DataFrame
    df["embeddings"] = embeddings
    return df


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def embedd_single_string(aoai_embeddings: AzureOpenAIEmbeddings, string: str):
    # Embedd a single string
    # The aoai_embeddings of type OpenAIEmbeddings has method embed_documents
    # The input is a list of strings
    # Lets get the list of strings from the Data frame, field concatenated
    # and then call the method
    # Embed the list of strings
    embeddings = aoai_embeddings.embed_query(string)
    return embeddings


def init_duckdb():

    conn = duckdb.connect("store.db", read_only=False)
    conn.execute(
        "CREATE TABLE data (id varchar, product_hier varchar, description varchar, concatenated varchar, vector FLOAT4[3072]);"
    )
    # text-embedding-3-large is the model used for embedding, it is a 3072 dimensional vector

    return conn


def normalize_vec_embeddings(df):
    df["embeddings"] = df["embeddings"].apply(lambda x: normalize(x))
    return df


def save_to_duckdb(df, conn):
    # Save the dataframe to duckdb
    # insert into the table "my_table" from te DataFrame "my_df"
    result = duckdb.sql("INSERT INTO data SELECT * FROM df", connection=conn)
    print(result)


def duckdb_cosine_similarity(conn, vector1):
    result = conn.execute(
        "select id, product_hier, description,  array_cosine_similarity(vector, $query_vector::FLOAT4[3072]) as cosim from data order by cosim desc limit 10",
        {"query_vector": vector1},
    ).fetchdf()
    print(result.to_string())


def main():
    duck_conn = init_duckdb()
    aoai_embeddings = init_azureopenai()
    df = read_dairy_prods("dairy_products.csv")
    df = create_new_concatenate_column(df)
    df = embed(df, aoai_embeddings)
    df = normalize_vec_embeddings(df)
    print(df.head(5))
    save_to_duckdb(df, duck_conn)
    to_search_emb = embedd_single_string(
        aoai_embeddings,
        "Emmental Cheese - Swiss cheese, world-famous for its distinctive holes and one-of-a-kind flavor. Aged minimum for 4 four months.",
    )
    duckdb_cosine_similarity(duck_conn, to_search_emb)


if __name__ == "__main__":
    main()
