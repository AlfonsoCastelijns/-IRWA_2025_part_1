import random
import numpy as np

from collections import Counter
import math
from myapp.search.objects import Document


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res

import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Delete no ASCII character
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces
    tokens = text.split() # Tokenize
    tokens = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in tokens] # Apply stemming

    return ' '.join(stemmed)


def query(query_terms, inverted_index):
    # We preprocess the query terms using the same pipeline as the corpus
    query_string = ' '.join(query_terms)
    cleaned_query = clean_text(query_string).split()

    if not cleaned_query:
        return []

    # We start with the set of documents that contain the first term
    first_term = cleaned_query[0]
    if first_term not in inverted_index:
        return []

    result_docs = set(inverted_index[first_term])

    # We intersect with the sets of documents for the remaining terms
    for term in cleaned_query[1:]:
        if term not in inverted_index:
            return []  # If any term is missing, no document can satisfy the full query
        result_docs.intersection_update(inverted_index[term])

    return list(result_docs)



# Function to get idf




def bm25_rank_classic_idf(query_terms, inverted_index, tf_scores, K1=1.6, B=0.75):


    df = {}
    for term in set(t for d in tf_scores.values() for t in d.keys()):
        df[term] = sum(1 for doc_id, counts in tf_scores.items() if term in counts)
    N=len(tf_scores)
    def get_idf(term):
        n_q = df.get(term, 0)
        return math.log(N / n_q) if n_q > 0 else 0.0

    # We clean query terms consistently with the rest of the pipeline
    cleaned = clean_text(' '.join(query_terms)).split()
    if not cleaned:
        return [], {}

    # Candidate documents obtained with query function already defined
    candidate_doc_ids = query(query_terms, inverted_index)
    if not candidate_doc_ids:
        return [], {}

    scores = {}
    # Parameters
    doc_lengths = {doc_id: sum(tf_scores.values()) for doc_id, tf_scores in tf_scores.items()}
    avgdl = np.mean(list(doc_lengths.values())) if doc_lengths else 0.0
    for doc_id in candidate_doc_ids:
        dl = doc_lengths.get(doc_id, 0)
        denom_norm = K1 * ((1 - B) + B * (dl / avgdl)) if avgdl > 0 else K1
        s = 0.0
        for q in cleaned:
            f = tf_scores[doc_id].get(q, 0)
            if f == 0:
                continue
            idf = get_idf(q)
            # BM25 contribution of a single term
            s += idf * ((f * (K1 + 1)) / (f + denom_norm))
        scores[doc_id] = s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked], scores




class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus,inverted_index, tf_scores):
        print("Search query:", search_query)

        query_terms = search_query.split()


        ranked_doc_ids, scores = bm25_rank_classic_idf(
            query_terms,
            inverted_index,   # must exist globally
            tf_scores         # must exist globally
        )
        ranked_doc_ids = sorted(
            scores.keys(),
            key=lambda pid: scores[pid],
            reverse=True
        )

        results = []
        for pid in ranked_doc_ids:
            if pid not in corpus:
                continue

            doc = corpus[pid]

            # Construct a Document() with a BM25 ranking score
            results.append(
                Document(
                    pid=doc.pid,
                    title=doc.title,
                    description=doc.description,
                    url=f"doc_details?pid={doc.pid}&search_id={search_id}&param2=2",
                    ranking=scores.get(pid, 0.0)
                )
            )

        # Return *all* ranked results or limit to top 20
        return results[:20]
