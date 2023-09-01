#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:40:42 2023

@author: soumensmacbookair
"""

# Imports
import numpy as np
import pandas as pd

# Get the documents from corpus
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

# Remove punctuations from text
def remove_punctuation(text, to_include = ""):

    punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')',
                         '*', '+', ',', '-', '.', '/', ':', ';', '<',
                         '=', '>', '?', '@', '[', '\\', ']', '^', '_',
                         '`', '{', '|', '}', '~']
    text_mod = ""
    for char in text:
        if char not in punctuation_chars or char == to_include:
            text_mod += char

    return text_mod

# Get stopword list from text file
def get_stopword_list(stopword_src):

    with open(stopword_src, "r") as f:
        content = f.read()
        content = remove_punctuation(content, ",")
        stop_word_list = content.split(",")

    return stop_word_list

# Count in Linear search
def count_linear_search(search_item, src_list):
    count = 0
    for i in src_list:
        if i == search_item:
            count += 1

    return count

# Tokenize the documents
def tokenize_document(document,
                      remove_stopword = True,
                      remove_number = True):

    # Convert document to lowercase
    abstract = document["abstract"].lower()
    title = document["title"].lower()
    author = document["author"].lower()

    # Remove punctuations from the documents
    abstract = remove_punctuation(abstract)
    title = remove_punctuation(title)
    author = " ".join([i.strip() for i in author.split(",") if "." not in i])

    # Split the document text to get tokens
    text = title + "\n" + author + "\n" + abstract
    text = text.replace("\n"," ")
    token_list = text.split(" ")
    token_list = [i.strip() for i in token_list if i]

    # Remove stopwords
    stop_word_list = get_stopword_list("stopwords.txt")

    if remove_stopword:
        token_list = [i for i in token_list if i not in stop_word_list]

    # Remove numbers
    digits_str = ["0","1","2","3","4","5","6","7","8","9"]
    if remove_number:
        token_list = [i for i in token_list if not i[0] in digits_str]

    # Remove single character tokens
    token_list = [i for i in token_list if len(i) > 1]

    # Get the unique tokens in sorted order
    token_list = sorted(list(set(token_list)))
    token_dict = {}
    text_list = text.split(" ")
    for i in token_list:
        token_dict[i] = count_linear_search(i, text_list)

    return token_dict

# Gnerate the tokens list for whole corpus
def generate_corpus_tokens(doc_dict):

    # Tokenize the documents
    all_token_dict = {}
    index_list = []
    for docid, doc in doc_dict.items():
        all_token_dict[docid] = tokenize_document(doc)
        index_list += list(all_token_dict[docid].keys())

    # Get sorted token dict for whole corpus
    index_list = sorted(list(set(index_list)))

    return index_list, all_token_dict

# Create the term - tf array for all documents
def create_tf_matrix(index_list, all_token_dict):

    # Get the term occurrence matrix
    toc_df = pd.DataFrame(0, columns=list(all_token_dict.keys()), index=index_list)

    for docid, token_dict in all_token_dict.items():
         for token, toc in token_dict.items():
             toc_df.loc[token, docid] = toc

    # Fill the NaN by 0 which means the toc is 0 for the (term,docid) pair
    toc_df.fillna(0, inplace=True)

    # Get the term frequency matrix
    tf_df = toc_df.div(toc_df.sum(axis=0, numeric_only=True), axis=1)
    tf_df = tf_df.astype(float)

    return tf_df

# Create the term - idf vector for all terms
def create_idf_vector(tf_df):

    # Create the doc_freq dataframe
    docfreq_df = pd.DataFrame(index=tf_df.index, columns=["doc_freq","idf"])
    docfreq_series = (tf_df != 0).sum(axis=1)
    docfreq_df["doc_freq"] = docfreq_series
    docfreq_df["idf"] = -np.log10(docfreq_series.to_numpy()) + np.log10(len(doc_dict))

    return docfreq_df.loc[:,"idf"].to_frame()

# Create the term - document tf-idf for all (term, doc) pair
def create_tf_idf_matrix(tf_df, idf_df):

    # Calculate the tf-idf
    tf_idf_df = pd.DataFrame(tf_df.to_numpy() * idf_df.to_numpy(),
                             columns=tf_df.columns, index=tf_df.index)

    return tf_idf_df

# Preprocess query for counting frequency of corpus tokens
def preprocess_query(query_dict):

    query_mod_dict = {}
    for qid, query_dict in query_dict.items():
        query = query_dict["query"]
        query = query.lower()
        query = remove_punctuation(query)

        query = query.replace("\n"," ")
        query_token_list = query.split(" ")
        query_token_list = [i.strip() for i in query_token_list if i]

        stop_word_list = get_stopword_list("stopwords.txt")
        query_token_list = [i for i in query_token_list if i not in stop_word_list]

        digits_str = ["0","1","2","3","4","5","6","7","8","9"]
        query_token_list = [i for i in query_token_list if not i[0] in digits_str]
        query_token_list = [i for i in query_token_list if len(i) > 1]
        query_token_list = list(set(query_token_list))

        query = " ".join(query_token_list)
        query_mod_dict[qid] = query

    return query_mod_dict

# Create the tf-query matrix
def create_tf_query_matrix(query_data_dict, index_list):
    query_toc_df = pd.DataFrame(0, columns=list(query_data_dict.keys()),
                               index=index_list)
    for index in index_list:
        for qid, query in query_data_dict.items():
            query_list = query.split(" ")
            query_toc_df.loc[index,qid] = count_linear_search(index, query_list)

    query_tf_df = query_toc_df.div(query_toc_df.sum(axis=0, numeric_only=True), axis=1)
    query_tf_df = query_tf_df.astype(float)

    return query_tf_df

# Calculate the similarity score for all the queries
def calculate_similarity_score(query_tf_idf_df, doc_tf_idf_df):

    q_arr = query_tf_idf_df.to_numpy()
    q_arr = q_arr/np.linalg.norm(q_arr,axis=0)
    d_arr = doc_tf_idf_df.to_numpy()

    sim_arr = np.dot(q_arr.T, d_arr)
    sim_arr = sim_arr/np.sum(sim_arr,axis=1,keepdims=True)*100
    sim_df = pd.DataFrame(sim_arr,
                          columns=doc_tf_idf_df.columns,
                          index=query_tf_idf_df.columns)
    return sim_df

# Retrieve highly ranked documents for all query
def retrieve_ranked_document(sim_df, doc_dict):

    ranked_dict = {}
    for i in range(sim_df.shape[0]):
        doc_rank_list = list(sim_df.iloc[i,:].sort_values(ascending=False).index)[0:10]
        top10_dict = {}
        for j,k in enumerate(doc_rank_list):
            top10_dict[f"rank_{j}"] = doc_dict[k]

        ranked_dict[sim_df.index[i]] = top10_dict

    return ranked_dict


# Main code
if __name__ == "__main__":

    # Get the tf-idf matrix for the documents
    doc_dict = get_documents_from_corpus('cisi/cisi.all')
    index_list, all_token_dict = generate_corpus_tokens(doc_dict)
    tf_df = create_tf_matrix(index_list, all_token_dict)
    idf_df = create_idf_vector(tf_df)
    doc_tf_idf_df = create_tf_idf_matrix(tf_df, idf_df)

    # Get the tf-idf matrix for the queries
    query_dict = get_documents_from_corpus('cisi/cisi.qry',"query")
    query_mod_dict = preprocess_query(query_dict)
    query_tf_df = create_tf_query_matrix(query_mod_dict, index_list)
    query_tf_idf_df = create_tf_idf_matrix(query_tf_df, idf_df)

    # Calculate the similarity score for query and document pair
    sim_df = calculate_similarity_score(query_tf_idf_df, doc_tf_idf_df)
    ranked_dict = retrieve_ranked_document(sim_df, doc_dict)




