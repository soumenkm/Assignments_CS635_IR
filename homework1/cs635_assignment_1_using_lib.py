#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:52:57 2023

@author: soumensmacbookair
"""

# Imports
import os
import numpy as np
import pandas as pd
from whoosh.index import create_in, open_dir
from whoosh.fields import TEXT, ID, Schema
from whoosh.qparser import QueryParser, MultifieldParser

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

# Function to create an index using Whoosh
def create_index(doc_dict, index_dir):
    schema = Schema(title=TEXT(stored=True),
                    author=TEXT(stored=True),
                    abstract=TEXT(stored=True),
                    docid=ID(unique=True, stored=True))
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for docid, doc in doc_dict.items():
        title = doc['title']
        author = doc['author']
        abstract = doc['abstract']
        writer.add_document(title=title, author=author, abstract=abstract, docid=docid)
    writer.commit()

# Function to perform TF-IDF retrieval using Whoosh
def tf_idf_retrieval(query, index_dir):
    ix = open_dir(index_dir)
    searcher = ix.searcher()

    # Define the fields you want to search across
    fields_to_search = ["title", "abstract", "author"]

    # Create a MultiFieldParser for the specified fields
    parser = MultifieldParser(fields_to_search, schema=ix.schema)

    # Parse the user's query to generate a query object
    query = parser.parse(query)

    # Execute the search query using the searcher
    results = searcher.search(query, limit=10)

    # Return the search results
    return results

if __name__ == "__main__":
    doc_dict = get_documents_from_corpus('cisi/cisi.all')
    index_dir = 'cisi_index'

    # Create the Whoosh index
    create_index(doc_dict, index_dir)

    # Perform TF-IDF retrieval for a query
    query = "information retrieval techniques"
    results = tf_idf_retrieval(query, index_dir)

    # Display the results (e.g., titles of relevant documents)
    for result in results:
        print(f"Doc ID: {result['docid']} - Title: {result['title']}")




