#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:52:57 2023

@author: soumensmacbookair
"""

# Imports
import os
import shutil as st
import numpy as np
import pandas as pd
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh import index, fields, writing
import cs635_assignment_1 as util

# Define a schema for the index
schema = fields.Schema(
    docid=fields.ID(stored=True),
    title=fields.TEXT(stored=True),
    abstract=fields.TEXT(stored=True),
    author=fields.TEXT(stored=True)
)

# Create an index directory
index_dir = "cisi_index"
st.rmtree(index_dir)
if not index.exists_in(index_dir):
    os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)

# Open the index for writing
writer = writing.BufferedWriter(ix)

# Index the documents
doc_dict = util.get_documents_from_corpus('cisi/cisi.all')
for docid, doc in doc_dict.items():
    writer.add_document(
        docid=docid,
        title=doc["title"],
        abstract=doc["abstract"],
        author=doc["author"]
    )
writer.commit()

# Create a searcher for the index
searcher = ix.searcher()
query_parser = MultifieldParser(["title", "abstract", "author"],
                                schema=ix.schema,
                                group=OrGroup)
retrieval_results = {}

# Perform retrieval for each query
query_dict = util.get_documents_from_corpus('cisi/cisi.qry',"query")
query_mod_dict = util.preprocess_query(query_dict)
for qid, query in query_mod_dict.items():
    parsed_query = query_parser.parse(query)
    results = searcher.search(parsed_query, limit=20, scored=True)
    retrieval_results[qid] = [result["docid"] for result in results]

top = len(retrieval_results["queryid_1"])
ranked_df = pd.DataFrame(0,
                         columns=[f"rank_{j+1}" for j in range(top)],
                         index=list(retrieval_results.keys()))

for i,j in retrieval_results.items():
    ranked_df.loc[i,:] = j

rel_df = util.get_relevance("cisi/cisi.rel")

#%%
ranked_rel_df = pd.DataFrame(0,
                             index=list(ranked_df.index),
                             columns=[f"rank_{i+1}" for i in range(top)])

for qid in ranked_df.index:
    for rid in ranked_df.columns:
        docid = ranked_df.loc[qid,rid].replace("docid_","")
        if (rel_df[(rel_df["qryid"] == qid.replace("queryid_","")) &
                   (rel_df["docid"] == docid)]).shape[0] != 0:
            ranked_rel_df.loc[qid,rid] = 1
#%%
mean_avg_prec = util.calculate_MAP(ranked_rel_df, ranked_df)
avg_ndcg = util.calculate_NDCG(ranked_rel_df, ranked_df)
