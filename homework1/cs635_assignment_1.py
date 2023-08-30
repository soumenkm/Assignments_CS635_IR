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
def get_documents_from_corpus(corpus_src):

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
            "title": title,
            "author": author,
            "abstract": abstract,
            "ref": ref
        }

    return doc_dict

# Tokenize the documents
def tokenize_document(document, remove_stopword = True, remove_number = True):

    # Convert tokens to lowercase
    token_list = []
    abstract = document["abstract"].lower()
    title = document["title"].lower()
    author = document["author"].lower()

    # Remove punctuations from tokens
    text = title + "\n" + author + "\n" + abstract
    punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')',
                         '*', '+', ',', '-', '.', '/', ':', ';', '<',
                         '=', '>', '?', '@', '[', '\\', ']', '^', '_',
                         '`', '{', '|', '}', '~']
    text_mod = ""
    for char in text:
        if char not in punctuation_chars:
            text_mod += char

    # Remove stopwords
    token_list += text_mod.split()
    stop_word_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                      'ourselves', 'you', 'your', 'yours', 'yourself',
                      'yourselves', 'he', 'him', 'his', 'himself', 'she',
                      'her', 'hers', 'herself', 'it', 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves',
                      'what', 'which', 'who', 'whom', 'this', 'that',
                      'these', 'those', 'am', 'is', 'are', 'was', 'were',
                      'be', 'been', 'being', 'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                      'but', 'if', 'or', 'because', 'as', 'until', 'while',
                      'of', 'at', 'by', 'for', 'with', 'about', 'against',
                      'between', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'to', 'from', 'up', 'down',
                      'in', 'out', 'on', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no',
                      'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                      'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
                      'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
                      'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won',
                      'wouldn', 'also']

    if remove_stopword:
        token_list = [i for i in token_list if i not in stop_word_list]

    # Remove numbers
    digits_str = ["0","1","2","3","4","5","6","7","8","9"]
    if remove_number:
        token_list = [i for i in token_list if not i[0] in digits_str]

    # Remove single character tokens
    token_list = [i for i in token_list if len(i) > 1]

    # Get the unique tokens and sorted order
    token_list = sorted(list(set(token_list)))
    token_dict = {}
    for i in token_list:
        token_dict[i] = text_mod.count(i)

    return token_dict

# Create the term - tf array for all documents
def create_tf_matrix(doc_dict):

    # Tokenize the documents
    all_token_dict = {}
    index_list = []
    for docid, doc in doc_dict.items():
        all_token_dict[docid] = tokenize_document(doc)
        index_list += list(all_token_dict[docid].keys())

    # Get sorted token dict for whole corpus
    index_list = sorted(list(set(index_list)))

    # Get the term occurrence matrix
    toc_df = pd.DataFrame(columns=list(all_token_dict.keys()), index=index_list)

    for docid, token_dict in all_token_dict.items():
         for token, toc in token_dict.items():
             toc_df.loc[token, docid] = toc

    # Fill the NaN by 0 which means the toc is 0 for the (term,docid) pair
    toc_df.fillna(0, inplace=True)

    # Get the term frequency matrix
    tf_df = toc_df.div(toc_df.sum(axis=0, numeric_only=True), axis=1)

    return tf_df

# Create the term - tf-idf vector for all terms
def create_tf_idf_vector(tf_df):

    # Create the doc_freq dataframe
    docfreq_df = pd.DataFrame(index=tf_df.index, columns=["doc_freq","idf","tf_idf"])
    docfreq_series = (tf_df != 0).sum(axis=1)
    docfreq_df["doc_freq"] = docfreq_series
    docfreq_df["idf"] = -np.log10(docfreq_series.to_numpy()) + np.log10(len(doc_dict))
    docfreq_df["tf_idf"] = docfreq_df["doc_freq"] * docfreq_df["idf"]

    return docfreq_df.iloc[:,-1].to_frame()

# Main code
if __name__ == "__main__":
    doc_dict = get_documents_from_corpus('cisi/cisi.all')
    tf_df = create_tf_matrix(doc_dict)
    tf_idf_df = create_tf_idf_vector(tf_df)

