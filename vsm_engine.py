"""
Vector Space Model (VSM) Information Retrieval Engine
Comparing TF-IDF Variant Schemes and Cosine Similarity

Author: Devanarayan C S
Roll No: 23FE10CCE00110
B.Tech Computer and Communication Engineering (4th Year)
Manipal University
Information Retrieval Mini Project
"""

import os
import re
import math
import pickle
import string
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

# Curated metadata dictionary for corpus research papers
CURATED_METADATA = {
    1: {
        "title": "A Historical Perspective of Explainable Artificial Intelligence",
        "authors": "Roberto Confalonieri, Ludovik Coba, Benedikt Wagner, Tarek R. Besold",
        "topic": "Explainable AI (XAI)",
        "year": "2021",
        "source": "WIREs Data Mining and Knowledge Discovery"
    },
    2: {
        "title": "Towards Explainable Artificial Intelligence",
        "authors": "Wojciech Samek, Klaus-Robert Müller",
        "topic": "Explainable AI (XAI)",
        "year": "2019",
        "source": "Fraunhofer Heinrich Hertz Institute / arXiv"
    },
    3: {
        "title": "Explainable AI: A Brief Survey on History, Research Areas, Approaches and Challenges",
        "authors": "Eleni Tjoa, Cuntai Guan",
        "topic": "Explainable AI (XAI)",
        "year": "2019",
        "source": "IEEE Transactions on Neural Networks"
    },
    7: {
        "title": "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges",
        "authors": "Alejandro Barredo Arrieta, Natalia Diaz-Rodriguez, Javier Del Ser et al.",
        "topic": "Explainable AI (XAI)",
        "year": "2020",
        "source": "Information Fusion (Elsevier)"
    },
    8: {
        "title": "Big Data and Artificial Intelligence in Heart Failure Diagnosis and Management",
        "authors": "Eun-Seok Jeon, Jin-Oh Choi, Sang-Eun Lee et al.",
        "topic": "Healthcare & Clinical Big Data",
        "year": "2020",
        "source": "Korean Circulation Journal"
    },
    9: {
        "title": "Exploring Symptoms in Patients with Chronic Heart Failure",
        "authors": "Tiny Jaarsma, Anna Strömberg, Martje H. L. van der Wal",
        "topic": "Clinical Cardiology",
        "year": "2018",
        "source": "European Journal of Heart Failure"
    },
    11: {
        "title": "A Timely and Accurate Heart Failure Diagnosis: Pathways to Treatment",
        "authors": "Martin R. Cowie, Simon Ray, Robert D. Shannon",
        "topic": "Clinical Cardiology",
        "year": "2019",
        "source": "British Journal of Cardiology"
    },
    12: {
        "title": "25 Years of Time Series Forecasting: A Comprehensive Review",
        "authors": "Jan G. De Gooijer, Rob J. Hyndman",
        "topic": "Time Series Forecasting",
        "year": "2006",
        "source": "International Journal of Forecasting"
    },
    13: {
        "title": "Machine Learning Advances for Time Series Forecasting (Part I)",
        "authors": "Ricardo P. Masini, Marcelo C. Medeiros, Eduardo F. Mendes",
        "topic": "Machine Learning & Econometrics",
        "year": "2021",
        "source": "Journal of Econometrics"
    },
    14: {
        "title": "Machine Learning Advances for Time Series Forecasting (Part II: High Dimensions)",
        "authors": "Ricardo P. Masini, Marcelo C. Medeiros, Eduardo F. Mendes",
        "topic": "Machine Learning & Econometrics",
        "year": "2021",
        "source": "Journal of Econometrics"
    },
    15: {
        "title": "Toward Automatic Time-Series Forecasting Using Neural Networks",
        "authors": "Weizhong Yan (Senior Member, IEEE)",
        "topic": "Neural Networks & Forecasting",
        "year": "2012",
        "source": "IEEE Transactions on Neural Networks"
    },
    16: {
        "title": "Deep Learning for Time Series Forecasting: A Survey",
        "authors": "Jose F. Torres, Dalil Hadjout, Abderrazak Sebaa, Alicia Troncoso",
        "topic": "Deep Learning & Forecasting",
        "year": "2021",
        "source": "Big Data (Mary Ann Liebert)"
    },
    17: {
        "title": "Training Tips for the Transformer Model: Architecture & Convergence",
        "authors": "Martin Popel, Ondřej Bojar",
        "topic": "Natural Language Processing (NLP)",
        "year": "2018",
        "source": "Charles University / Prague Bulletin"
    },
    18: {
        "title": "Transformers: State-of-the-Art Natural Language Processing",
        "authors": "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond et al.",
        "topic": "Natural Language Processing (NLP)",
        "year": "2020",
        "source": "EMNLP Systems Demonstrations (Hugging Face)"
    },
    21: {
        "title": "Transformers in Time Series: A Survey on Architectures and Applications",
        "authors": "Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen et al.",
        "topic": "Transformers & Time Series",
        "year": "2022",
        "source": "Alibaba Group DAMO Academy / IJCAI"
    },
    22: {
        "title": "Feature Selection: A Data Perspective and Taxonomy",
        "authors": "Jundong Li, Kewei Cheng, Suhang Wang, Fred Morstatter, Huan Liu",
        "topic": "Feature Selection & Data Mining",
        "year": "2017",
        "source": "ACM Computing Surveys (CSUR)"
    },
    23: {
        "title": "Algorithms for Feature Selection: An Empirical Evaluation",
        "authors": "Douglas Zongker",
        "topic": "Machine Learning & Feature Selection",
        "year": "2015",
        "source": "University of Washington Technical Report"
    },
    24: {
        "title": "Feature Selection Methods in Pattern Recognition and Machine Learning",
        "authors": "Hanchuan Peng, Fuhui Long, Chris Ding",
        "topic": "Pattern Recognition & Dimensionality",
        "year": "2016",
        "source": "Procedia Computer Science"
    },
    25: {
        "title": "Feature Selection from Huge Feature Sets in Computer Vision",
        "authors": "Jose Bins, Bruce A. Draper",
        "topic": "Computer Vision & Feature Selection",
        "year": "2014",
        "source": "IEEE Transactions on Pattern Analysis"
    },
    26: {
        "title": "FREL: A Stable Feature Selection Algorithm using Regularized Energy-Based Learning",
        "authors": "Yun Li, Jennie Si, Guojing Zhou, Shasha Huang, Songcan Chen",
        "topic": "Ensemble Machine Learning",
        "year": "2015",
        "source": "IEEE Transactions on Neural Networks"
    }
}

# Extended standard English stopwords list
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'cannot', 'could', 'couldn', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven',
    'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 'more',
    'most', 'mustn', 'my', 'myself', 'needn', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same',
    'shan', 'she', 'should', 'shouldn', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were', 'weren', 'what', 'when',
    'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'would', 'wouldn', 'y',
    'you', 'your', 'yours', 'yourself', 'yourselves'
}

# SMART Notation Definitions
SCHEME_DEFINITIONS = {
    'ntc': {
        'name': 'Standard Raw TF-IDF',
        'code': 'ntc.ntc',
        'tf': 'Natural / Raw frequency (tf)',
        'idf': 'Standard log10(N / df)',
        'norm': 'Cosine Normalization',
        'description': 'Classical Salton TF-IDF weighting. Direct linear scaling with term occurrence, normalized by document Euclidean vector length.',
        'formula_doc': r'w_{t,d} = tf_{t,d} \times \log_{10}\left(\frac{N}{df_t}\right)',
        'formula_query': r'w_{t,q} = tf_{t,q} \times \log_{10}\left(\frac{N}{df_t}\right)'
    },
    'ltc': {
        'name': 'Logarithmic TF-IDF',
        'code': 'ltc.ltc',
        'tf': 'Log-frequency (1 + log10(tf))',
        'idf': 'Standard log10(N / df)',
        'norm': 'Cosine Normalization',
        'description': 'Cornell/SMART benchmark. Applies logarithmic damping to prevent exceptionally frequent terms from disproportionately dominating.',
        'formula_doc': r'w_{t,d} = (1 + \log_{10}(tf_{t,d})) \times \log_{10}\left(\frac{N}{df_t}\right)',
        'formula_query': r'w_{t,q} = (1 + \log_{10}(tf_{t,q})) \times \log_{10}\left(\frac{N}{df_t}\right)'
    },
    'atc': {
        'name': 'Augmented Frequency TF-IDF',
        'code': 'atc.atc',
        'tf': 'Double normalized TF: 0.5 + 0.5 * (tf / max_tf)',
        'idf': 'Standard log10(N / df)',
        'norm': 'Cosine Normalization',
        'description': 'Normalizes term frequency by maximum term frequency in the document to prevent verbosity bias before vector normalization.',
        'formula_doc': r'w_{t,d} = \left(0.5 + 0.5 \frac{tf_{t,d}}{\max_t(tf_{t,d})}\right) \times \log_{10}\left(\frac{N}{df_t}\right)',
        'formula_query': r'w_{t,q} = \left(0.5 + 0.5 \frac{tf_{t,q}}{\max_t(tf_{t,q})}\right) \times \log_{10}\left(\frac{N}{df_t}\right)'
    },
    'btc': {
        'name': 'Binary / Boolean TF-IDF',
        'code': 'btc.btc',
        'tf': 'Boolean Presence: 1 if tf > 0 else 0',
        'idf': 'Standard log10(N / df)',
        'norm': 'Cosine Normalization',
        'description': 'Evaluates coordination level based solely on term occurrence presence/absence, mitigating keyword stuffing.',
        'formula_doc': r'w_{t,d} = 1.0 \times \log_{10}\left(\frac{N}{df_t}\right) \quad [tf > 0]',
        'formula_query': r'w_{t,q} = 1.0 \times \log_{10}\left(\frac{N}{df_t}\right) \quad [tf > 0]'
    }
}


class Document:
    """Represents a research paper in the corpus with metadata and token statistics."""
    def __init__(self, doc_id, filename, raw_text, tokens):
        self.doc_id = int(doc_id)
        self.filename = filename
        self.raw_text = raw_text
        self.tokens = tokens
        self.tf = Counter(tokens)
        self.max_tf = max(self.tf.values()) if self.tf else 1
        self.total_tokens = len(tokens)
        self.word_count = len(raw_text.split())

        # Curated metadata lookup
        meta = CURATED_METADATA.get(self.doc_id, {})
        self.title = meta.get("title", f"Research Paper #{self.doc_id}")
        self.authors = meta.get("authors", "Faculty & Academic Researchers")
        self.topic = meta.get("topic", "Computer Science")
        self.year = meta.get("year", "2020")
        self.source = meta.get("source", "Peer-Reviewed Publication")

        # Extract abstract
        self.abstract = self._extract_abstract(raw_text)

    def _extract_abstract(self, text):
        m = re.search(r'(?:abstract|overview)\s*[:\-\n](.*?)(?:\n\s*(?:keywords|1\s*\||introduction|index terms)|\Z)', text, re.IGNORECASE | re.DOTALL)
        if m:
            clean = re.sub(r'\s+', ' ', m.group(1)).strip()
            if len(clean) > 80:
                return clean[:450] + ("..." if len(clean) > 450 else "")
        # Fallback to first non-empty lines
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 30 and not l.startswith('http') and not l.startswith('DOI')]
        if lines:
            fallback = ' '.join(lines[:3])
            return fallback[:450] + ("..." if len(fallback) > 450 else "")
        return "No explicit abstract header detected. Full paper text available in document viewer."


class VSMEngine:
    """
    Advanced Vector Space Model Search Engine supporting:
    - Multiple SMART TF-IDF Variant Schemes (ntc, ltc, atc, btc)
    - Cosine Similarity Ranking with Mathematical Diagnostics
    - Multi-Scheme Comparative Analysis & Rank Correlation Metrics
    - Document Preview, Snippet Extraction, and Highlighted Matched Terms
    """
    def __init__(self, corpus_dir="ResearchPapers", cache_file="vsm_cache.pkl"):
        self.corpus_dir = corpus_dir
        self.cache_file = cache_file
        self.stemmer = PorterStemmer()
        self.tokenizer = WordPunctTokenizer()

        self.documents = {}       # doc_id -> Document
        self.inverted_index = {}  # term -> {doc_id: freq}
        self.doc_freq = {}        # term -> df
        self.vocabulary = set()
        self.N = 0

        # Precomputed vectors & norms per scheme
        # scheme -> doc_id -> {'weights': {term: w}, 'norm': float}
        self.doc_vectors = {s: {} for s in SCHEME_DEFINITIONS.keys()}

        self.initialize_engine()

    def preprocess(self, text):
        """Tokenize, lowercase, remove stopwords & non-alphas, apply Porter Stemming."""
        if not text:
            return []
        tokens = self.tokenizer.tokenize(text.lower())
        cleaned = []
        for t in tokens:
            if t.isalpha() and len(t) > 2 and t not in STOPWORDS:
                cleaned.append(self.stemmer.stem(t))
        return cleaned

    def initialize_engine(self, force_rebuild=False):
        """Load or build the inverted index, document statistics, and TF-IDF representations."""
        if not force_rebuild and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.documents = cache_data['documents']
                    self.inverted_index = cache_data['inverted_index']
                    self.doc_freq = cache_data['doc_freq']
                    self.vocabulary = cache_data['vocabulary']
                    self.N = cache_data['N']
                    self.doc_vectors = cache_data['doc_vectors']
                    print(f"[VSMEngine] Loaded cached index for {self.N} documents ({len(self.vocabulary)} terms).")
                    return
            except Exception as e:
                print(f"[VSMEngine] Cache load failed ({e}), rebuilding fresh index...")

        self._build_index_from_scratch()
        self._precompute_all_schemes()
        self._export_text_indices()

        # Cache compiled engine
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'inverted_index': self.inverted_index,
                    'doc_freq': self.doc_freq,
                    'vocabulary': self.vocabulary,
                    'N': self.N,
                    'doc_vectors': self.doc_vectors
                }, f)
                print(f"[VSMEngine] Successfully cached index to {self.cache_file}")
        except Exception as e:
            print(f"[VSMEngine] Warning: Could not cache to file ({e})")

    def _build_index_from_scratch(self):
        """Reads all documents in ResearchPapers/ and constructs full inverted index."""
        print("[VSMEngine] Scanning and parsing corpus directory...")
        self.documents.clear()
        self.inverted_index.clear()
        self.vocabulary.clear()

        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir, exist_ok=True)

        for filename in os.listdir(self.corpus_dir):
            if filename.endswith(".txt"):
                doc_id_str = filename.split('.')[0]
                try:
                    doc_id = int(doc_id_str)
                except ValueError:
                    continue

                filepath = os.path.join(self.corpus_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read()

                tokens = self.preprocess(raw_content)
                doc = Document(doc_id, filename, raw_content, tokens)
                self.documents[doc_id] = doc

                # Inverted index postings
                for term, freq in doc.tf.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = {}
                    self.inverted_index[term][doc_id] = freq
                    self.vocabulary.add(term)

        self.N = len(self.documents)
        self.doc_freq = {t: len(postings) for t, postings in self.inverted_index.items()}
        print(f"[VSMEngine] Built index: {self.N} docs, {len(self.vocabulary)} unique terms.")

    def get_idf(self, term):
        """Computes standard inverse document frequency: log10(N / df)."""
        df = self.doc_freq.get(term, 0)
        if df == 0 or self.N == 0:
            return 0.0
        return math.log10(self.N / df)

    def _precompute_all_schemes(self):
        """Precomputes document term weights and Euclidean norms for all 4 SMART schemes."""
        print("[VSMEngine] Precomputing document vectors and norms for all TF-IDF variants...")
        for scheme in SCHEME_DEFINITIONS.keys():
            self.doc_vectors[scheme] = {}
            for doc_id, doc in self.documents.items():
                weights = {}
                sq_sum = 0.0

                for term, freq in doc.tf.items():
                    idf = self.get_idf(term)
                    if idf <= 0.0:
                        continue

                    # Compute TF according to SMART scheme
                    if scheme == 'ntc':
                        tf_val = float(freq)
                    elif scheme == 'ltc':
                        tf_val = 1.0 + math.log10(freq) if freq > 0 else 0.0
                    elif scheme == 'atc':
                        tf_val = 0.5 + 0.5 * (freq / doc.max_tf)
                    elif scheme == 'btc':
                        tf_val = 1.0 if freq > 0 else 0.0
                    else:
                        tf_val = float(freq)

                    weight = tf_val * idf
                    if weight > 0.0:
                        weights[term] = weight
                        sq_sum += (weight ** 2)

                norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
                self.doc_vectors[scheme][doc_id] = {
                    'weights': weights,
                    'norm': norm
                }

    def _export_text_indices(self):
        """Export clean inverted_indexA2.txt and TF-IDF.txt for backwards compatibility."""
        try:
            with open('inverted_indexA2.txt', 'w', encoding='utf-8') as f:
                for i, (term, postings) in enumerate(sorted(self.inverted_index.items())):
                    postings_str = ", ".join([f"({d}: {freq})" for d, freq in sorted(postings.items())])
                    f.write(f"{i+1} - {term} -> {postings_str}, \n")

            with open('TF-IDF.txt', 'w', encoding='utf-8') as f:
                ntc_vectors = self.doc_vectors.get('ntc', {})
                for doc_id in sorted(self.documents.keys()):
                    weights = ntc_vectors.get(doc_id, {}).get('weights', {})
                    terms_str = ", ".join([f"{term}: {round(val, 6)}" for term, val in sorted(weights.items())[:200]])
                    f.write(f"{doc_id} -> {terms_str}\n")
        except Exception as e:
            print(f"[VSMEngine] Warning exporting text indices: {e}")

    def compute_query_vector(self, query_tokens, scheme='ntc'):
        """Computes query term weights and Euclidean length norm for given scheme."""
        q_counts = Counter(query_tokens)
        max_q = max(q_counts.values()) if q_counts else 1
        q_weights = {}
        sq_sum = 0.0

        for term, freq in q_counts.items():
            idf = self.get_idf(term)
            if idf <= 0.0:
                continue

            if scheme == 'ntc':
                tf_val = float(freq)
            elif scheme == 'ltc':
                tf_val = 1.0 + math.log10(freq) if freq > 0 else 0.0
            elif scheme == 'atc':
                tf_val = 0.5 + 0.5 * (freq / max_q)
            elif scheme == 'btc':
                tf_val = 1.0 if freq > 0 else 0.0
            else:
                tf_val = float(freq)

            weight = tf_val * idf
            if weight > 0.0:
                q_weights[term] = weight
                sq_sum += (weight ** 2)

        norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
        return q_weights, norm

    def search_single_scheme(self, query_str, scheme='ntc', top_n=10, min_score=0.0001):
        """
        Executes Vector Space Model Cosine Similarity ranking under a specific TF-IDF scheme.
        Formula:
            Cosine Similarity(Q, D) = (Q . D) / (||Q|| * ||D||)
        """
        if scheme not in SCHEME_DEFINITIONS:
            scheme = 'ntc'

        query_tokens = self.preprocess(query_str)
        if not query_tokens:
            return {
                'scheme': scheme,
                'scheme_info': SCHEME_DEFINITIONS[scheme],
                'query': query_str,
                'query_tokens': [],
                'results': [],
                'total_matches': 0
            }

        q_weights, q_norm = self.compute_query_vector(query_tokens, scheme)
        if q_norm == 0 or not q_weights:
            return {
                'scheme': scheme,
                'scheme_info': SCHEME_DEFINITIONS[scheme],
                'query': query_str,
                'query_tokens': query_tokens,
                'results': [],
                'total_matches': 0
            }

        scheme_docs = self.doc_vectors.get(scheme, {})
        scores = {}
        term_diagnostics = {}

        # Candidate documents containing at least one query term
        candidate_docs = set()
        for term in q_weights.keys():
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())

        for doc_id in candidate_docs:
            doc_data = scheme_docs.get(doc_id)
            if not doc_data:
                continue

            d_weights = doc_data['weights']
            d_norm = doc_data['norm']

            if d_norm <= 0:
                continue

            dot_product = 0.0
            term_contribs = []

            for term, qw in q_weights.items():
                if term in d_weights:
                    dw = d_weights[term]
                    product = qw * dw
                    dot_product += product
                    term_contribs.append({
                        'term': term,
                        'query_weight': round(qw, 4),
                        'doc_weight': round(dw, 4),
                        'product': round(product, 4)
                    })

            if dot_product > 0:
                cosine_sim = dot_product / (q_norm * d_norm)
                if cosine_sim >= min_score:
                    scores[doc_id] = cosine_sim
                    term_contribs.sort(key=lambda x: x['product'], reverse=True)
                    term_diagnostics[doc_id] = term_contribs

        # Sort descending by score
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for rank, (doc_id, score) in enumerate(ranked_items, start=1):
            doc = self.documents[doc_id]
            contribs = term_diagnostics.get(doc_id, [])
            matched_terms = [c['term'] for c in contribs]
            snippet = self.generate_snippet(doc_id, query_tokens)

            results.append({
                'rank': rank,
                'doc_id': doc_id,
                'title': doc.title,
                'authors': doc.authors,
                'topic': doc.topic,
                'year': doc.year,
                'source': doc.source,
                'score': round(score, 5),
                'score_pct': min(100.0, round(score * 100, 2)),
                'matched_terms': matched_terms,
                'term_contributions': contribs[:5],
                'snippet': snippet,
                'total_words': doc.word_count
            })

        return {
            'scheme': scheme,
            'scheme_info': SCHEME_DEFINITIONS[scheme],
            'query': query_str,
            'query_tokens': query_tokens,
            'query_weights': {t: round(w, 4) for t, w in q_weights.items()},
            'results': results,
            'total_matches': len(scores)
        }

    def search_all_schemes(self, query_str, top_n=10):
        """
        Executes query on ALL 4 schemes simultaneously and builds cross-scheme comparison matrix.
        Calculates Spearman rank correlation, rank shifts, and consensus rankings.
        """
        schemes_results = {}
        for sch in SCHEME_DEFINITIONS.keys():
            schemes_results[sch] = self.search_single_scheme(query_str, scheme=sch, top_n=top_n)

        # Collect all unique retrieved document IDs
        all_doc_ids = set()
        for sch, res in schemes_results.items():
            for item in res['results']:
                all_doc_ids.add(item['doc_id'])

        # Cross-scheme comparison table
        comparison_matrix = []
        for doc_id in sorted(all_doc_ids):
            doc = self.documents[doc_id]
            row = {
                'doc_id': doc_id,
                'title': doc.title,
                'topic': doc.topic,
                'schemes': {}
            }
            ranks = []
            scores = []
            for sch in SCHEME_DEFINITIONS.keys():
                found = next((r for r in schemes_results[sch]['results'] if r['doc_id'] == doc_id), None)
                if found:
                    row['schemes'][sch] = {
                        'rank': found['rank'],
                        'score': found['score'],
                        'score_pct': found['score_pct']
                    }
                    ranks.append(found['rank'])
                    scores.append(found['score'])
                else:
                    row['schemes'][sch] = {
                        'rank': '-',
                        'score': 0.0,
                        'score_pct': 0.0
                    }

            row['avg_score'] = round(sum(scores) / len(scores), 4) if scores else 0.0
            row['min_rank'] = min(ranks) if ranks else 99
            comparison_matrix.append(row)

        # Sort matrix by consensus best rank / avg score
        comparison_matrix.sort(key=lambda x: (x['min_rank'], -x['avg_score']))

        # Calculate Spearman Rank Correlations between schemes
        correlations = {}
        scheme_keys = list(SCHEME_DEFINITIONS.keys())
        for i in range(len(scheme_keys)):
            for j in range(i + 1, len(scheme_keys)):
                s1 = scheme_keys[i]
                s2 = scheme_keys[j]
                pair_key = f"{s1.upper()} vs {s2.upper()}"
                r1 = [r['doc_id'] for r in schemes_results[s1]['results']]
                r2 = [r['doc_id'] for r in schemes_results[s2]['results']]
                correlations[pair_key] = self._calculate_spearman_rho(r1, r2)

        return {
            'query': query_str,
            'query_tokens': schemes_results['ntc']['query_tokens'],
            'schemes': schemes_results,
            'comparison_matrix': comparison_matrix,
            'correlations': correlations,
            'total_unique_docs': len(all_doc_ids)
        }

    def _calculate_spearman_rho(self, list1, list2):
        """Computes Spearman's rank correlation coefficient between two ranked result sets."""
        if not list1 or not list2:
            return 0.0
        docs = sorted(list(set(list1) | set(list2)))
        n = len(docs)
        if n <= 1:
            return 1.0

        max_penalty = n + 1
        rank1 = {d: i + 1 for i, d in enumerate(list1)}
        rank2 = {d: i + 1 for i, d in enumerate(list2)}

        d_sq_sum = sum((rank1.get(d, max_penalty) - rank2.get(d, max_penalty)) ** 2 for d in docs)
        rho = 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))
        return round(rho, 3)

    def generate_snippet(self, doc_id, query_tokens, max_length=220):
        """
        Generates contextual snippet from document text containing the highest query term density,
        with query tokens safely wrapped in high-visibility tags for frontend styling.
        """
        doc = self.documents.get(doc_id)
        if not doc or not query_tokens:
            return "No excerpt available."

        text = doc.raw_text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        best_sentence = ""
        max_hits = -1

        stemmed_q = set(query_tokens)

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 30 or len(s_clean) > 400:
                continue
            words = self.preprocess(s_clean)
            hits = sum(1 for w in words if w in stemmed_q)
            if hits > max_hits:
                max_hits = hits
                best_sentence = s_clean

        if not best_sentence and sentences:
            best_sentence = sentences[0].strip()

        # Highlight occurrences of query terms in the snippet
        clean_snip = re.sub(r'\s+', ' ', best_sentence)
        if len(clean_snip) > max_length:
            clean_snip = clean_snip[:max_length].rsplit(' ', 1)[0] + "..."

        return clean_snip

    def get_corpus_stats(self):
        """Returns comprehensive corpus analytics for the IR dashboard."""
        total_tokens = sum(doc.total_tokens for doc in self.documents.values())
        total_words = sum(doc.word_count for doc in self.documents.values())
        avg_doc_len = round(total_words / self.N, 1) if self.N else 0

        # Top terms by document frequency
        top_df = sorted(self.doc_freq.items(), key=lambda x: x[1], reverse=True)[:15]

        # Topic breakdown
        topic_counts = Counter(doc.topic for doc in self.documents.values())

        return {
            'num_documents': self.N,
            'vocabulary_size': len(self.vocabulary),
            'total_tokens': total_tokens,
            'total_words': total_words,
            'avg_doc_length': avg_doc_len,
            'top_df_terms': [{'term': t, 'df': count, 'idf': round(self.get_idf(t), 3)} for t, count in top_df],
            'topic_distribution': dict(topic_counts),
            'schemes': SCHEME_DEFINITIONS
        }


# Singleton instance
engine = VSMEngine()

