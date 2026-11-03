"""
========================================================================================
Vector Space Model (VSM) Information Retrieval Search Engine
Comparing TF-IDF Variant Schemes and Cosine Similarity

Student Information:
  - Name: Devanarayan C S
  - Roll No: 23FE10CCE00110
  - Degree: B.Tech in Computer and Communication Engineering (CCE), 4th Year
  - University: Manipal University
  - Course: Information Retrieval Mini Project
========================================================================================
"""

import os
import math
from flask import Flask, request, render_template, jsonify, redirect, url_for
from vsm_engine import engine, SCHEME_DEFINITIONS

# Creating Flask web application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vsm-ir-project-devanarayan-23fe10cce00110'

# Global author & project metadata
PROJECT_METADATA = {
    "author": "Devanarayan C S",
    "roll_no": "23FE10CCE00110",
    "degree": "B.Tech in Computer and Communication Engineering",
    "year": "4th Year",
    "institution": "Manipal University",
    "project": "Information Retrieval Mini Project",
    "title": "Vector Space Model Search Engine: Comparing TF-IDF Variant Schemes and Cosine Similarity"
}


# ======================================================================================
# Flask Routes
# ======================================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main Search Portal and Results Handler.
    Supports GET (displays clean landing page) and POST (performs query ranking).
    """
    stats = engine.get_corpus_stats()

    query = ""
    scheme = "multi"

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        scheme = request.form.get('scheme', 'multi').strip().lower()
    elif request.method == 'GET' and request.args.get('query'):
        query = request.args.get('query', '').strip()
        scheme = request.args.get('scheme', 'multi').strip().lower()

    if not query:
        return render_template(
            'search.html',
            active_page='search',
            stats=stats,
            project=PROJECT_METADATA,
            scheme=scheme
        )

    # Execute search based on scheme selection
    if scheme == 'multi':
        multi_results = engine.search_all_schemes(query, top_n=10)
        return render_template(
            'results.html',
            active_page='search',
            query=query,
            scheme=scheme,
            multi_results=multi_results,
            stats=stats,
            project=PROJECT_METADATA
        )
    else:
        single_results = engine.search_single_scheme(query, scheme=scheme, top_n=10)
        return render_template(
            'results.html',
            active_page='search',
            query=query,
            scheme=scheme,
            single_results=single_results,
            stats=stats,
            project=PROJECT_METADATA
        )


@app.route('/compare', methods=['GET'])
def compare():
    """
    Dedicated Multi-Scheme Comparative Analysis View.
    Evaluates the query simultaneously across:
      1. Standard Raw TF-IDF (ntc)
      2. Logarithmic TF-IDF (ltc)
      3. Augmented TF-IDF (atc)
      4. Binary TF-IDF (btc)
    """
    query = request.args.get('query', 'explainable artificial intelligence').strip()
    stats = engine.get_corpus_stats()
    multi_results = engine.search_all_schemes(query, top_n=10)

    return render_template(
        'compare.html',
        active_page='compare',
        query=query,
        multi_results=multi_results,
        stats=stats,
        project=PROJECT_METADATA
    )


@app.route('/analytics', methods=['GET'])
def analytics():
    """
    Corpus Overview, Vocabulary Analytics, and SMART Weighting Scheme Reference.
    """
    stats = engine.get_corpus_stats()
    sorted_docs = dict(sorted(engine.documents.items()))

    return render_template(
        'analytics.html',
        active_page='analytics',
        stats=stats,
        documents=sorted_docs,
        project=PROJECT_METADATA
    )


# ======================================================================================
# REST API Endpoints
# ======================================================================================

@app.route('/api/search', methods=['GET'])
def api_search():
    """JSON API to perform search."""
    query = request.args.get('query', '').strip()
    scheme = request.args.get('scheme', 'multi').strip().lower()

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    if scheme == 'multi':
        results = engine.search_all_schemes(query)
    else:
        results = engine.search_single_scheme(query, scheme=scheme)

    return jsonify(results)


@app.route('/api/document/<int:doc_id>', methods=['GET'])
def api_document(doc_id):
    """JSON API to fetch full document text and metadata."""
    doc = engine.documents.get(doc_id)
    if not doc:
        return jsonify({'error': f'Document #{doc_id} not found'}), 404

    return jsonify({
        'doc_id': doc.doc_id,
        'title': doc.title,
        'authors': doc.authors,
        'topic': doc.topic,
        'year': doc.year,
        'source': doc.source,
        'word_count': doc.word_count,
        'abstract': doc.abstract,
        'content_excerpt': doc.raw_text[:4000] + ("\n\n... [Content truncated for preview] ..." if len(doc.raw_text) > 4000 else "")
    })


# ======================================================================================
# Backward Compatibility Functions
# ======================================================================================

def preprocessing(text):
    """Backward compatibility wrapper for tokenization and stemming."""
    return engine.preprocess(text)


def cosine_similarity(query_dict, doc_dict):
    """
    Calculates cosine similarity between two term weight dictionaries.
    Formula: dot_product / (||query|| * ||doc||)
    """
    dot_product = sum(query_dict.get(term, 0.0) * doc_dict.get(term, 0.0) for term in set(query_dict) & set(doc_dict))
    query_norm = math.sqrt(sum(val ** 2 for val in query_dict.values()))
    doc_norm = math.sqrt(sum(val ** 2 for val in doc_dict.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot_product / (query_norm * doc_norm)


def search_query(query, scheme='ntc', top_n=10):
    """CLI search helper function for interactive terminal testing."""
    res = engine.search_single_scheme(query, scheme=scheme, top_n=top_n)
    print(f"\nSearch results for query: '{query}' [Scheme: {scheme.upper()}]:")
    print("-" * 75)
    for r in res['results']:
        print(f"Rank #{r['rank']} | Doc ID: {r['doc_id']:2d} | Score: {r['score']:.5f} | Title: {r['title'][:45]}")
    return [(r['doc_id'], r['score']) for r in res['results']]


def main():
    """CLI test routine."""
    print("=" * 80)
    print(f" {PROJECT_METADATA['title']} ")
    print(f" Author: {PROJECT_METADATA['author']} | Roll No: {PROJECT_METADATA['roll_no']} ")
    print(f" {PROJECT_METADATA['degree']} | {PROJECT_METADATA['institution']} ")
    print("=" * 80)
    print(f"Corpus Loaded: {engine.N} Research Papers | Vocabulary: {len(engine.vocabulary)} Unique Terms\n")

    # Sample query execution
    sample_q = "explainable artificial intelligence"
    search_query(sample_q, scheme='ltc', top_n=5)


if __name__ == '__main__':
    main()
    print("\nStarting Flask web server on http://localhost:5000 ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
