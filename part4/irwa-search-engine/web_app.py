import os
import json
from json import JSONEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from collections import Counter


try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    #only run this code once, for the main process
    # load documents corpus into memory.
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    file_path = path + "/" + os.getenv("DATA_FILE_PATH")
    # Read cached processed data if available
    if os.path.exists("data/cleaned_corpus.json") and os.path.exists("data/inverted_index.json") and os.path.exists("data/tf_scores.json") and os.path.exists("data/df.json"):
        print("Loading cached processed data...")

        # Load corpus
        with open("data/cleaned_corpus.json", "r", encoding="utf-8") as f:
            corpus_json = json.load(f)
            # convert json → Document instances
            corpus = {pid: Document(**doc) for pid, doc in corpus_json.items()}   

        # Load inverted index
        with open("data/inverted_index.json", "r", encoding="utf-8") as f:
            inverted_index = json.load(f)

        # Load TF scores
        with open("data/tf_scores.json", "r", encoding="utf-8") as f:
            tf_scores = json.load(f)
        with open("data/df.json", "r", encoding="utf-8") as f:
            df = json.load(f)

    else:
        print("Loading and preprocessing original corpus...")
        #Otherwise load original corpus and process it
        corpus = load_corpus(file_path)



        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\x00-\x7F]+', '', text) # Delete no ASCII character
            text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
            text = re.sub(r'\s+', ' ', text) # Replace multiple spaces
            tokens = text.split() # Tokenize
            tokens = [word for word in tokens if word not in stop_words]
            stemmed = [stemmer.stem(word) for word in tokens] # Apply stemming

            return ' '.join(stemmed)

        for doc in corpus.values():  # or corpus if it's a list
            doc.title_clean = clean_text(doc.title or "")
            doc.description_clean = clean_text(doc.description or "")

        REQUIRED_FIELDS = [
            'pid', 'title', 'description', 'brand', 'category', 'sub_category',
            'product_details', 'seller', 'out_of_stock', 'selling_price',
            'discount', 'actual_price', 'average_rating', 'url'
        ]
        # We ensure all required fields are present in each document
        def ensure_fields(doc: Document):
            for field in REQUIRED_FIELDS:
                if not hasattr(doc, field):
                    setattr(doc, field, None)
            return doc

        for doc in corpus.values():
            ensure_fields(doc)

        def build_metadata_text(doc: Document):
            brand = doc.brand or ""
            category = doc.category or ""
            sub_category = doc.sub_category or ""
            seller = doc.seller or ""

            # product_details is list[dict]
            details_list = []
            if isinstance(doc.product_details, list):
                for d in doc.product_details:
                    for k, v in d.items():
                        details_list.append(f"{k} {v}")

            product_details = " ".join(details_list)

            return f"{brand} {category} {sub_category} {product_details} {seller}".lower()

        for doc in corpus.values():
            doc.metadata_text = build_metadata_text(doc)


        def normalize_numeric_fields(doc: Document):
            # selling_price
            try:
                if isinstance(doc.selling_price, str):
                    doc.selling_price = float(doc.selling_price.replace(",", "."))
                else:
                    doc.selling_price = float(doc.selling_price)
            except:
                doc.selling_price = None
            # actual_price
            try:
                if isinstance(doc.actual_price, str):
                    doc.actual_price = float(doc.actual_price.replace(",", "."))
                else:
                    doc.actual_price = float(doc.actual_price)
            except:
                doc.actual_price = None
            # discount
            try:
                if isinstance(doc.discount, str):
                    doc.discount = int(doc.discount.replace("% off", "").strip())
                else:
                    doc.discount = int(doc.discount)
            except:
                doc.discount = None
            # average_rating
            try:
                doc.average_rating = float(doc.average_rating)
            except:
                doc.average_rating = None

            return doc

        for doc in corpus.values():
            normalize_numeric_fields(doc)

        inverted_index = {}
        # Iterate through each document in the corpus
        for doc in corpus.values():
            doc_id = doc.pid

            cleaned_text = (doc.title_clean or "") + " " + (doc.description_clean or "")
            terms = cleaned_text.split()

            for term in terms:
                if term not in inverted_index:
                    inverted_index[term] = []
                if doc_id not in inverted_index[term]:
                    inverted_index[term].append(doc_id)

        tf_scores = {}

        for doc in corpus.values():
            doc_id = doc.pid
            cleaned_text = (doc.title_clean or "") + " " + (doc.description_clean or "")
            terms = cleaned_text.split()
            term_counts = Counter(terms)
            tf_scores[doc_id] = dict(term_counts)

        df = {}
        for term in set(t for d in tf_scores.values() for t in d.keys()):
            df[term] = sum(1 for doc_id, counts in tf_scores.items() if term in counts)
        N=len(tf_scores)

        # Save processed data to cache files
        with open("data/cleaned_corpus.json", "w", encoding="utf-8") as f:
            json.dump({pid: doc.to_json() for pid, doc in corpus.items()},f,indent=2)
        with open("data/inverted_index.json", "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, indent=2)
        with open("data/tf_scores.json", "w", encoding="utf-8") as f:
            json.dump(tf_scores, f, indent=2)
        with open("data/df.json", "w", encoding="utf-8") as f:
            json.dump(df, f, indent=2)
    # Log first element of corpus to verify it loaded correctly:
    print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])





# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    
    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus, inverted_index, tf_scores,df)

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, rag_response=rag_response)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    res = session["some_var"]
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["pid"]
    print("click in id={}".format(clicked_doc_id))

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
    return render_template('doc_details.html',doc=corpus[clicked_doc_id])


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
