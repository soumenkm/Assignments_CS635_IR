#%% Import
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer

#%% Download NLTK data
nltk.download("stopwords")

#%% Get the documents from corpus
def get_documents_from_corpus(corpus_src, corpus_type = "doc"):

    # Extract the documents ID position
    with open(corpus_src, 'r') as file:
        cisi_all_contents = file.read()

    doc_start_pos_list = []
    digits_str = ["0","1","2","3","4","5","6","7","8","9"]
    for i, char in enumerate(cisi_all_contents):
        try:
            char_p1 = cisi_all_contents[i+1]
            char_p2 = cisi_all_contents[i+2]
            char_p3 = cisi_all_contents[i+3]
        except IndexError:
            break # The end of the corpus

        if char == '.' and char_p1 == "I" and char_p2 == " " and char_p3 in digits_str:
            doc_start_pos_list.append(i)

    # Extract the document attributes
    doc_list = [""] * len(doc_start_pos_list)
    for i in range(len(doc_start_pos_list)):
        start_pos = doc_start_pos_list[i]
        if i == len(doc_start_pos_list) - 1:
            doc_list[i] = cisi_all_contents[start_pos:]
        else:
            end_pos = doc_start_pos_list[i+1]
            doc_list[i] = cisi_all_contents[start_pos:end_pos]

    # Store all the documents in a dictionary
    doc_dict = {}
    for doc in doc_list:
        doc_lines = doc.splitlines()

        if corpus_type == "doc":
            pos_dotI = [i for i,j in enumerate(doc_lines) if ".I" in j][0]
            pos_dotT = [i for i,j in enumerate(doc_lines) if ".T" in j][0]
            pos_dotA = [i for i,j in enumerate(doc_lines) if ".A" in j][0]
            pos_dotW = [i for i,j in enumerate(doc_lines) if ".W" in j][0]
            pos_dotX = [i for i,j in enumerate(doc_lines) if ".X" in j][0]

            docid = ("\n".join(doc_lines[pos_dotI:pos_dotT])).replace(".I ","docid_")
            title = "\n".join(doc_lines[pos_dotT+1:pos_dotA])
            author = "\n".join(doc_lines[pos_dotA+1:pos_dotW])
            abstract = "\n".join(doc_lines[pos_dotW+1:pos_dotX])
            ref = "\n".join(doc_lines[pos_dotX+1:])

            doc_dict[docid] = {
                "id": docid,
                "title": title,
                "author": author,
                "abstract": abstract,
                "ref": ref
            }
        elif corpus_type == "query":
            pos_dotI = [i for i,j in enumerate(doc_lines) if ".I" in j][0]
            pos_dotW = [i for i,j in enumerate(doc_lines) if ".W" in j][0]
            try:
                pos_dotT = [i for i,j in enumerate(doc_lines) if ".T" in j][0]
                pos_dotA = [i for i,j in enumerate(doc_lines) if ".A" in j][0]
            except IndexError:
                qid = ("\n".join(doc_lines[pos_dotI:pos_dotW])).replace(".I ","queryid_")
                query = "\n".join(doc_lines[pos_dotW+1:])
                doc_dict[qid] = {
                    "id": qid,
                    "title": "",
                    "author": "",
                    "query": query
                }
            else:
                qid = ("\n".join(doc_lines[pos_dotI:pos_dotT])).replace(".I ","queryid_")
                title = "\n".join(doc_lines[pos_dotT+1:pos_dotA])
                author = "\n".join(doc_lines[pos_dotA+1:pos_dotW])
                query = "\n".join(doc_lines[pos_dotW+1:])

                doc_dict[qid] = {
                    "id": qid,
                    "title": title,
                    "author": author,
                    "query": query
                }
        else:
            raise ValueError("Invalid corpus type. Valid: ['doc','query']")

    return doc_dict

#%% Text preprocessing and tokenize using nltk or hugging face
def preprocess_text(text, ishf):

    # Tokenize
    if ishf:
        model_name = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
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

#%% Get top k ranked documents against all query
def get_topk_results(doc_dict, query_dict, topk, ishf):

    # Preprocess the documents and queries
    queries = {qid: qry["title"] + " " + qry["query"] for qid, qry in query_dict.items()}
    corpus = {docid: doc["title"] + " " + doc["abstract"] + " " + doc["author"] for docid, doc in doc_dict.items()}

    # Apply preprocessing to corpus and queries
    corpus = {docid: preprocess_text(doc, ishf) for docid, doc in corpus.items()}
    queries = {qid: preprocess_text(query, ishf) for qid, query in queries.items()}
    corpus.update(queries)

    # Tokenize the corpus
    tokenized_corpus = {i: nltk.word_tokenize(doc.lower()) for i, doc in corpus.items()}

    # Implement TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(doc) for doc in tokenized_corpus.values()])
    similarities = linear_kernel(tfidf_matrix[-len(queries):], tfidf_matrix[:-len(queries)])

    # Get the top k document IDs for each query
    retrieval_results = {}
    for i, query in enumerate(queries):
        query_similarities = similarities[i]
        top_indices = query_similarities.argsort()[::-1][:topk]
        top_docs = [list(doc_dict.keys())[idx] for idx in top_indices]
        retrieval_results[f"queryid_{i + 1}"] = top_docs

    return retrieval_results

#%% Get the relevance or ground truth
def get_relevance(rel_src):

    with open(rel_src, "r") as f:
        cisi_rel_contents = f.read()

    line_list = cisi_rel_contents.split("\n")
    rel_data = []
    for i in line_list:
        i = i.replace("\t"," ")
        line = i.split()
        rel_data.append(line)

    rel_df = pd.DataFrame(rel_data,columns=["qryid","docid","rel","rel_score"])

    return rel_df

#%% Calculate relevance for the retrieved docs against all query
def calculate_relevance_ranked_doc(retrieval_results, rel_df):

    top = len(retrieval_results["queryid_1"])
    ranked_df = pd.DataFrame(0,
                             columns=[f"rank_{j+1}" for j in range(top)],
                             index=list(retrieval_results.keys()))

    for i,j in retrieval_results.items():
        ranked_df.loc[i,:] = j

    ranked_rel_df = pd.DataFrame(0,
                                 index=list(ranked_df.index),
                                 columns=[f"rank_{i+1}" for i in range(top)])

    for qid in ranked_df.index:
        for rid in ranked_df.columns:
            docid = ranked_df.loc[qid,rid].replace("docid_","")
            if (rel_df[(rel_df["qryid"] == qid.replace("queryid_","")) &
                       (rel_df["docid"] == docid)]).shape[0] != 0:
                ranked_rel_df.loc[qid,rid] = 1

    return ranked_df, ranked_rel_df

#%% Calculate MAP
def calculate_MAP(ranked_rel_df, ranked_df):

    topk = ranked_df.shape[1]
    prec_df = pd.DataFrame(0,
                           columns=[f"P@{i+1}" for i in range(topk)],
                           index=ranked_df.index)
    for i in range(ranked_rel_df.shape[0]):
        sorted_rel = ranked_rel_df.iloc[i,:].sort_values(ascending=False)
        for k in range(1,topk+1):
            prec_at_k = (sorted_rel.iloc[:k].sum())/k
            prec_df.iloc[i,k-1] = prec_at_k

    avg_prec_series = prec_df.mean(axis=1)
    mean_avg_prec = avg_prec_series.mean()

    return mean_avg_prec

#%% Calculate NDCG
def calculate_NDCG(ranked_rel_df, ranked_df):

    topk = ranked_df.shape[1]
    ndcg_df = pd.DataFrame(0, columns=["dcg","idcg","ndcg"], index=ranked_df.index)
    for i in range(ranked_rel_df.shape[0]):
        sorted_rel = ranked_rel_df.iloc[i,:].sort_values(ascending=False)
        unsorted_rel = ranked_rel_df.iloc[i,:]

        dcg = unsorted_rel.iloc[0]
        idcg = sorted_rel.iloc[0]
        for k in range(2,topk+1):
            dcg += unsorted_rel.iloc[k-1]/np.log2(k+1)
            idcg += sorted_rel.iloc[k-1]/np.log2(k+1)

        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg/idcg

        ndcg_df.iloc[i,0] = dcg
        ndcg_df.iloc[i,1] = idcg
        ndcg_df.iloc[i,2] = ndcg

    avg_ndcg = ndcg_df.mean(axis=0)[-1]
    return avg_ndcg


#%% Main code
if __name__ == "__main__":

    doc_dict = get_documents_from_corpus("cisi/cisi.all")
    query_dict = get_documents_from_corpus("cisi/cisi.qry", "query")

    retrieval_results = get_topk_results(doc_dict, query_dict, 10, False)

    rel_df = get_relevance("cisi/cisi.rel")
    ranked_df, ranked_rel_df = calculate_relevance_ranked_doc(retrieval_results, rel_df)

    mean_avg_prec = calculate_MAP(ranked_rel_df, ranked_df)
    avg_ndcg = calculate_NDCG(ranked_rel_df, ranked_df)

    retrieval_results_hf = get_topk_results(doc_dict, query_dict, 10, True)

    rel_df = get_relevance("cisi/cisi.rel")
    ranked_df, ranked_rel_df = calculate_relevance_ranked_doc(retrieval_results, rel_df)

    mean_avg_prec = calculate_MAP(ranked_rel_df, ranked_df)
    avg_ndcg = calculate_NDCG(ranked_rel_df, ranked_df)

#%%



