from whoosh.fields import Schema, TEXT, KEYWORD, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
import os, whoosh
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import cs635_assignment_1_using_sklearn as util

# Download NLTK data
nltk.download("stopwords")

# Define a function to preprocess text
def preprocess_text(text, is_hugging_face):
    if is_hugging_face:
        model_name = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        parts = []
        chunk_size = 512
        for i in range(0, len(text), chunk_size):
            part = text[i:i+chunk_size]
            parts.append(part)
        tokens = []
        for i, part in enumerate(parts):
            tokens += tokenizer.tokenize(part)
    else:
        tokens = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [word for word in tokens if not word.isnumeric() and len(word) > 2]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# Define a function to create a Whoosh index
def create_whoosh_index(index_dir, doc_dict, query_dict, is_hugging_face):
    schema = Schema(id=ID(stored=True), content=TEXT)
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    ix = create_in(index_dir, schema)

    writer = ix.writer()
    for doc_id, doc in doc_dict.items():
        content = preprocess_text(doc["title"] + " " + doc["abstract"] + " " + doc["author"], is_hugging_face)
        writer.add_document(id=doc_id, content=content)

    for query_id, query in query_dict.items():
        content = preprocess_text(query["title"] + " " + query["query"], is_hugging_face)
        writer.add_document(id=query_id, content=content)

    writer.commit()

# Define a function to perform a basic Whoosh search
def perform_whoosh_search(index_dir, query, top_k=10):
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        query_parser = MultifieldParser(["title", "abstract", "author"],
                                        schema=ix.schema,
                                        group=OrGroup)
        query = query_parser.parse(query)
        results = searcher.search(query, limit=top_k, scored=True)
        return [result['id'] for result in results]

# Define a function to get top-k ranked documents against all queries
def get_topk_results(doc_dict, query_dict, index_dir, top_k, is_hugging_face):
    create_whoosh_index(index_dir, doc_dict, query_dict, is_hugging_face)
    results = {}
    for query_id, query in query_dict.items():
        query_text = preprocess_text(query["title"] + " " + query["query"], is_hugging_face)
        search_results = perform_whoosh_search(index_dir, query_text, top_k)
        results[query_id] = search_results
    return results

# Define a function to calculate relevance for the retrieved documents
def calculate_relevance_ranked_doc(retrieval_results, rel_df):
    top = len(retrieval_results["queryid_1"])
    ranked_df = pd.DataFrame(0, columns=[f"rank_{j+1}" for j in range(top)], index=list(retrieval_results.keys()))

    for query_id, doc_ids in retrieval_results.items():
        ranked_df.loc[query_id, :] = doc_ids

    ranked_rel_df = pd.DataFrame(0, index=list(ranked_df.index), columns=[f"rank_{i+1}" for i in range(top)])

    for query_id in ranked_df.index:
        for rank_id in ranked_df.columns:
            doc_id = ranked_df.loc[query_id, rank_id].replace("docid_", "")
            if rel_df[(rel_df["qryid"] == query_id.replace("queryid_", "")) & (rel_df["docid"] == doc_id)].shape[0] != 0:
                ranked_rel_df.loc[query_id, rank_id] = 1

    return ranked_df, ranked_rel_df

# Define a function to calculate MAP
def calculate_MAP(ranked_rel_df, ranked_df):
    top_k = ranked_df.shape[1]
    prec_df = pd.DataFrame(0, columns=[f"P@{i+1}" for i in range(top_k)], index=ranked_df.index)

    for i in range(ranked_rel_df.shape[0]):
        sorted_rel = ranked_rel_df.iloc[i, :].sort_values(ascending=False)
        for k in range(1, top_k+1):
            prec_at_k = (sorted_rel.iloc[:k].sum()) / k
            prec_df.iloc[i, k-1] = prec_at_k

    avg_prec_series = prec_df.mean(axis=1)
    mean_avg_prec = avg_prec_series.mean()

    print(mean_avg_prec)

    return mean_avg_prec

# Define a function to calculate NDCG
def calculate_NDCG(ranked_rel_df, ranked_df):
    top_k = ranked_df.shape[1]
    ndcg_df = pd.DataFrame(0, columns=["dcg", "idcg", "ndcg"], index=ranked_df.index)

    for i in range(ranked_rel_df.shape[0]):
        sorted_rel = ranked_rel_df.iloc[i, :].sort_values(ascending=False)
        unsorted_rel = ranked_rel_df.iloc[i, :]

        dcg = unsorted_rel.iloc[0]
        idcg = sorted_rel.iloc[0]
        for k in range(2, top_k+1):
            dcg += unsorted_rel.iloc[k-1] / np.log2(k+1)
            idcg += sorted_rel.iloc[k-1] / np.log2(k+1)

        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg

        ndcg_df.iloc[i, 0] = dcg
        ndcg_df.iloc[i, 1] = idcg
        ndcg_df.iloc[i, 2] = ndcg

    avg_ndcg = ndcg_df.mean(axis=0)[-1]
    return avg_ndcg

# Main code
if __name__ == "__main__":
    doc_dict = util.get_documents_from_corpus("cisi/cisi.all")
    query_dict = util.get_documents_from_corpus("cisi/cisi.qry", "query")

    retrieval_results = get_topk_results(doc_dict, query_dict, "index_directory", 10, False)

    # rel_df = util.get_relevance("cisi/cisi.rel")
    # ranked_df, ranked_rel_df = calculate_relevance_ranked_doc(retrieval_results, rel_df)

    # mean_avg_prec = calculate_MAP(ranked_rel_df, ranked_df)
    # avg_ndcg = calculate_NDCG(ranked_rel_df, ranked_df)

    # retrieval_results_hf = get_topk_results(doc_dict, query_dict, "index_directory_hf", 10, True)

    # rel_df = util.get_relevance("cisi/cisi.rel")
    # ranked_df_hf, ranked_rel_df_hf = calculate_relevance_ranked_doc(retrieval_results_hf, rel_df)

    # mean_avg_prec_hf = calculate_MAP(ranked_rel_df_hf, ranked_df_hf)
    # avg_ndcg_hf = calculate_NDCG(ranked_rel_df_hf, ranked_df_hf)
