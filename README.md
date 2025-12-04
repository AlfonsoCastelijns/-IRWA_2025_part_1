# -IRWA_2025_part_1

**Part 1**

This part of the IRWA project involves text cleaning, data normalization, and preprocessing of a fashion product dataset, to prepare it for future search.

**Requirements:** The following python packages are used: json, re, nltk, pandas, numpy, matplotlib, seaborn, collections


In the file IRWA_part1.ipynb, we created multiple functions and code to do the tasks asked:


**Function:** `clean_text(text)`

-   Converts text to lowercase
-   Removes non-ASCII characters and punctuation
-   Collapses multiple spaces
-   Removes stopwords (using NLTK English stopwords)
-   Applies stemming (Porter Stemmer)


**Function:** `ensure_fields(doc)`

Ensures every document has the required set of fields:

``` python
REQUIRED_FIELDS = [
  'pid', 'title', 'description', 'brand', 'category', 'sub_category',
  'product_details', 'seller', 'out_of_stock', 'selling_price',
  'discount', 'actual_price', 'average_rating', 'url'
]
```
If any are missing, they are filled with `None`.

**Function:** `build_metadata_text(doc)`

Creates a new field called metadata_text by combining several metadata attributes into a single string. This field is useful for general-purpose indexing and fallback matching when the query is vague or not tied to a specific field.

  -Combines brand, category, sub_category, product_details, and seller
  -Helps support flexible search and exploratory queries

**Function:** `normalize_numeric_fields(doc)`

Cleans and converts numeric and rating-related fields into proper data
types: - Converts `selling_price` and `actual_price` to `float` -
Extracts integer values from `discount` (e.g., "69% off" → `69`) -
Converts `average_rating` to `float`


For part 2 we don't define any more functions, we use functions that are already created from different libraries, mainly matplotlib and seaborn and the code is executed just by pressing run

# -IRWA_2025_part_2

This second deliverable of the project involves creating an inverted index for our dataset of fashion products, the ranking of results and its evaluation.

**Part 1**

**Requirements:** The following python packages are used: json, re, nltk

In the file IRWA_part2.ipynb, we include the functions created in the previous deliverable: `clean_text()`,`ensure_fields()`,`build_metadata_text()`, and `normalize_numeric_fields()`. We then created the code to generate the inverted index, iterating through the documents' title and description, and using the stemmed words (with stopwords removed) as terms in our index. We define the following functions in succession:

**Function:** `query(query_terms,inverted_index)`

Executes a search query over an inverted index to retrieve documents containing all the query terms.
Parameters: 
- `query_terms (list[str])`: A list of terms representing the search query.
- `inverted_index (dict[str, list[int]])`: A mapping of terms to lists (or sets) of document IDs in which those terms appear.
Returns: `list[int]`: A list of document IDs that contain all the query terms. Returns an empty list if no documents match.ç

**Function:** `print_matching_documents(features)`  

Prints all documents from the corpus that match the specified feature terms.
Parameters:  
- `features (list[str])`:  A list of feature terms to search for in the corpus.
Returns: None, function only prints results to terminal/console.

**Function:** `rank_documents_tfidf(query_terms, inverted_index, tfidf_scores)`  

This function retrieves all documents matching the query terms (using the conjunctive query method) and computes a relevance score for each by summing the TF-IDF values of the query terms within each document. It then sorts the documents in descending order of their relevance scores.
Parameters: 
- `query_terms (list[str])`:  The list of search terms to query.  
- `inverted_index (dict[str, list[int]])`:  A mapping of terms to lists (or sets) of document IDs where those terms appear.  
- `tfidf_scores (dict[int, dict[str, float]])`:  A mapping of document IDs to dictionaries containing TF-IDF scores for each term.
Returns:
- `tuple`:
  - `list[int]`: Document IDs ranked by descending relevance.  
  - `dict[int, float]`: A dictionary mapping document IDs to their computed relevance scores.
 
**Function:** `print_ranked_documents(features, inverted_index, tfidf_scores, corpus)`  

Prints documents ranked by TF-IDF relevance for the given feature terms.
Parameters: 
Same as previous, with one addition:  - `corpus` (`list[dict]`):  The full list of documents, where each document is represented as a dictionary containing metadata.
Returns: None, The function prints ranked results directly to the console.

**Part 2**

For the evaluation section of this deliverable, we create multiple helper functions to calculate certain values:

**Function:** `precision_at_k(retrieved, relevant, k)`

Calculates the precision of retrieved documents at a specified cutoff rank *k*.
Parameters:
- `retrieved` (`list[int]`):  A list of document IDs retrieved by the search or ranking system, ordered by relevance.  
- `relevant` (`set[int]` or `list[int]`):  A collection of document IDs that are considered relevant to the query.  
- `k` (`int`):  The cutoff rank — only the top-*k* retrieved documents are considered.

Returns:* `float`:  The precision at *k*, calculated as the number of relevant documents in the top-*k* results divided by *k*.

**Function:** `recall_at_k(retrieved, relevant, k)`

Calculates the recall of retrieved documents at a specified cutoff rank *k*.
Parameters: Same as previous
Returns: `float`:  The recall at *k*, calculated as the number of relevant documents retrieved in the top-*k* divided by the total number of relevant documents. Returns `0` if there are no relevant documents.

**Function:** `average_precision_at_k(retrieved, relevant, k)`  

Computes the Average Precision of retrieved documents at a specified cutoff rank *k*.
Parameters: Same as previous
Returns:`float`:  The average precision at *k*, computed as the mean of the precision values obtained at the ranks where relevant documents occur. Returns `0` if there are no relevant documents.

**Function:** `f1_at_k(retrieved, relevant, k)`  

Calculates the F1-score at a specified cutoff rank *k*.
Parameters: Same as previous
Returns: `float`:  The F1-score at *k*, computed as `2 * (precision * recall) / (precision + recall)`. Returns `0` if both precision and recall are zero.

**Function:** `dcg_at_k(retrieved, relevant, k)`  

Calculates the Discounted Cumulative Gain at a specified cutoff rank *k*.
Parameters: Same as previous
Returns: `float`: The DCG score at *k*, computed as the sum of `(relevance / log2(rank + 1))` over the top-*k* documents.

**Function:** `ndcg_at_k(retrieved, relevant, k)`  

Computes the Normalized Discounted Cumulative Gain at a specified cutoff rank *k*.
Parameters: Same as previous
Returns: `float`:  The nDCG score at *k*, computed as `DCG@k / IDCG@k`. Returns `0` if the ideal DCG is zero (i.e., no relevant documents).

**Function:** `reciprocal_rank(retrieved, relevant)`  

Computes the Reciprocal Rank for a single query.

Parameters: Same as previous without `k (int)`
Returns: `float`:  The reciprocal of the rank of the first relevant document (`1 / rank`). Returns `0` if no relevant documents are retrieved.


**Function:** `mean_average_precision_at_k(ranked_results, relevant_dict, k)`  

Calculates the Mean Average Precision across multiple queries at a specified cutoff rank *k*.
Parameters:
- `ranked_results` (`dict[str, list[int]]`):  A dictionary mapping each query ID to its list of retrieved document IDs, ordered by predicted relevance.  
- `relevant_dict` (`dict[str, set[int]]`):  A dictionary mapping each query ID to a set of document IDs that are considered relevant for that query.  
- `k` (`int`):  The cutoff rank — only the top-*k* retrieved documents per query are considered.
Returns: `float`: The mean average precision at *k*, computed as the average of all *AP@k* scores across queries. Returns `0` if no queries are available or if all *AP* scores are empty.


Finally, we create a function which calls the previous and evaluates a single query:

**Function:** `evaluate_query(retrieved, relevant, k=5)`

Evaluates a single query using multiple information retrieval metrics at a specified cutoff rank *k*.
Parameters:  
- `retrieved` (`list[int]`): List of document IDs retrieved by the system, ordered by predicted relevance.  
- `relevant` (`set[int]` or `list[int]`): Collection of document IDs that are known to be relevant to the query.  
- `k` (`int`, optional, default=`5`): The cutoff rank — only the top-*k* retrieved documents are considered for evaluation.
Returns:
- `dict[str, float]`:  
  A dictionary containing rounded values of key retrieval metrics:  
  - `"P@K"` – Precision at *k*  
  - `"R@K"` – Recall at *k*  
  - `"AP@K"` – Average Precision at *k*  
  - `"MAP@K"` – Mean Average Precision at *k* (same as `AP@K` for a single query)  
  - `"F1@K"` – F1-score at *k*  
  - `"MRR"` – Mean Reciprocal Rank  
  - `"NDCG@K"` – Normalized Discounted Cumulative Gain at *k*


To execute the code for the second deliverable, simply run each cell in file `IRWA_part2.ipynb` in order. 

# -IRWA_2025_part_3

This deliverable is on different methods of ranking.
To execute the code for the third deliverable, simply run each cell in file `IRWA_part3.ipynb` in order, having the fashion dataset in the same directory.

**Part 1**

**Function:** `build_query_vector(query_terms, idf_scores)`  
Builds a TF‑IDF weighted query vector from the given query terms.  

Parameters:
- `query_terms` (`list[str]`): Terms in the query.  
- `idf_scores` (`dict[str, float]`): IDF values for each term in the corpus.  

Returns:
- `dict[str, float]`: A dictionary mapping each query term to its TF‑IDF weight.  

**Function:** `cosine_similarity(query_vector, doc_vector, doc_norm)`  
Computes the cosine similarity between a query vector and a document vector.  

Parameters:
- `query_vector` (`dict[str, float]`): TF‑IDF weighted query vector.  
- `doc_vector` (`dict[str, float]`): TF‑IDF weighted document vector.  
- `doc_norm` (`float`): Precomputed norm of the document vector.  

Returns:
- `float`: Cosine similarity score between the query and the document.  

**Function:** `get_idf(term)`  
Computes the inverse document frequency (IDF) for a given term.  

Parameters:
- `term` (`str`): The term for which to compute IDF.  

Returns:
- `float`: The IDF value for the term. Returns `0.0` if the term does not appear in any document.  

**Function:** `rank_documents_tfidf_cosine(query_terms, inverted_index, tfidf_scores, idf_scores, doc_norms)`
Ranks documents using TF‑IDF weights combined with cosine similarity.  

Parameters:
- `query_terms` (`list[str]`): Terms in the query.  
- `inverted_index` (`dict[str, set[int]]`): Maps terms to document IDs containing them.  
- `tfidf_scores` (`dict[int, dict[str, float]]`): TF‑IDF weights per document.  
- `idf_scores` (`dict[str, float]`): IDF values for each term.  
- `doc_norms` (`dict[int, float]`): Precomputed vector norms for each document.  

Returns:
- `list[int]`: Ranked document IDs.  
- `dict[int, float]`: Cosine similarity scores per document.  

**Function:** `bm25_rank_classic_idf(query_terms, inverted_index, tf_counts, K1=1.2, B=0.75)`
Ranks documents using the BM25 probabilistic model.  

Parameters:
- `query_terms` (`list[str]`): Terms in the query.  
- `inverted_index` (`dict[str, set[int]]`): Maps terms to candidate documents.  
- `tf_counts` (`dict[int, dict[str, int]]`): Term frequencies per document.  
- `K1` (`float`): Controls term frequency saturation (default = 1.2).  
- `B` (`float`): Controls document length normalization (default = 0.75).  

Returns:
- `list[int]`: Ranked document IDs.  
- `dict[int, float]`: BM25 scores per document.

**Function**: `custom_score(query_terms, inverted_index, tf_counts, corpus, alpha=0.3, beta=0.2, gamma=0.8, title_weight=1.5, desc_weight=1.0)`
Ranks documents using a hybrid scoring function tailored for e‑commerce.  

Parameters:
- `query_terms` (`list[str]`): Terms in the query.  
- `inverted_index` (`dict[str, set[int]]`): Candidate documents.  
- `tf_counts` (`dict[int, dict[str, int]]`): Term frequencies.  
- `corpus` (`list[dict]`): Full product dataset with metadata.  
- `alpha, beta, gamma` (`float`): Weights for discount, rating, and stock penalty.  
- `title_weight, desc_weight` (`float`): Weights for textual matches.  

Returns:
- `list[int]`: Ranked document IDs.  
- `dict[int, float]`: Custom scores per document.

**Part 2**

**Function:** `text_to_w2v_vector(text, model)`  
Converts a text string into a Word2Vec vector representation by averaging word embeddings.  

Parameters:
- `text` (`str`): Input text to be converted into a vector.  
- `model` (`gensim.models.Word2Vec`): Pre‑trained Word2Vec model containing word embeddings.  

Returns:
- `numpy.ndarray`: A vector representation of the text. Returns a zero vector if none of the words are in the model vocabulary.  

**Function:** `rank_documents_word2vec(query_terms, inverted_index, doc_vectors, model)`  
Ranks documents using Word2Vec embeddings and cosine similarity.  

Parameters:
- `query_terms` (`list[str]`): Terms in the query.  
- `inverted_index` (`dict[str, set[int]]`): Maps terms to candidate document IDs.  
- `doc_vectors` (`dict[int, numpy.ndarray]`): Precomputed Word2Vec vector representations for each document.  
- `model` (`gensim.models.Word2Vec`): Pre‑trained Word2Vec model used to generate query vectors.  

Returns:
- `list[int]`: Ranked document IDs.  
- `dict[int, float]`: Cosine similarity scores between the query vector and each document vector.  

# -IRWA_2025_part_4

This is a guide on how to execute our code:

- Clone the github and proceed to the `\part4\irwa_search_engine` folder. Insert the `fashion_dataset.json` file into the `\data` folder.
- Create a virtual environment if desired, then execute the 'web_app.py' file from the terminal.
- The first execution of the code will compute all the necessary files for a fast searching process, so it could take around 30 seconds to complete. This will be saved into the `/data' folder and will be loaded in any other execution.
- Open any of the local links provided, and you will be presented with the search page. From there, you can input your query, set minimum and maximum prices, and search. In the results page, you will see the LLM summary of the best results, and can view and click on the results shown below it for further information.
- You can also access the `Stats` or `Dashboard` pages from the links at the top of the results page.

