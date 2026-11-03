# Vector Space Model (VSM) Search Engine
### Comparing TF-IDF Variant Schemes and Cosine Similarity

> **Information Retrieval Mini Project**  
> **Student Name:** Devanarayan C S  
> **Roll No:** 23FE10CCE00110  
> **Degree:** B.Tech in Computer and Communication Engineering (CCE), 4th Year  
> **Institution:** Manipal University  

---

## 📌 Executive Summary

This project implements an advanced Information Retrieval (IR) System based on the **Vector Space Model (VSM)**. It enables searching, scoring, and ranking across a curated corpus of peer-reviewed computer science research papers.

Unlike rudimentary search engines that rely on a single, fixed TF-IDF formula, this engine features a comprehensive **multi-scheme comparative framework** implementing the **SMART notation** (Salton & Buckley, 1988). The system empowers researchers and evaluators to observe, analyze, and quantify how distinct term weighting and length normalization strategies affect retrieval performance and document ranking.

---

## 🔬 Theoretical Foundations: SMART TF-IDF Variant Schemes

In Information Retrieval literature, weighting schemes are conventionally represented using the three-letter SMART notation: `ddd.qqq` (document weight `ddd`, query weight `qqq`).

Each three-letter code designates:
1. **Term Frequency (TF) Weighting** (`n`, `l`, `a`, `b`)
2. **Document Frequency (IDF) Weighting** (`t`, `n`, `p`)
3. **Vector Normalization** (`c`, `n`)

This project implements and empirically compares four standard IR schemes:

| Scheme | SMART Code | Document TF Formula ($tf_{t,d}$) | IDF Weight ($idf_t$) | Vector Normalization | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Raw TF-IDF** | `ntc.ntc` | $tf_{t,d}$ (raw frequency) | $\log_{10}\left(\frac{N}{df_t}\right)$ | Cosine ($1 / \|\vec{d}\|$) | Classical Salton model; linear frequency scaling. Vulnerable to document length bias. |
| **Logarithmic TF-IDF** | `ltc.ltc` | $1 + \log_{10}(tf_{t,d})$ if $tf > 0$ else $0$ | $\log_{10}\left(\frac{N}{df_t}\right)$ | Cosine ($1 / \|\vec{d}\|$) | SMART Cornell benchmark. Sub-linear damping prevents single term repetition surges. |
| **Augmented TF-IDF** | `atc.atc` | $0.5 + 0.5 \times \frac{tf_{t,d}}{\max_t(tf_{t,d})}$ | $\log_{10}\left(\frac{N}{df_t}\right)$ | Cosine ($1 / \|\vec{d}\|$) | Double normalization against maximum term frequency. Mitigates document verbosity bias. |
| **Binary / Boolean TF-IDF** | `btc.btc` | $1.0$ if $tf_{t,d} > 0$ else $0$ | $\log_{10}\left(\frac{N}{df_t}\right)$ | Cosine ($1 / \|\vec{d}\|$) | Pure concept coordination level; scores relevance solely on distinct keyword coverage. |

### Cosine Similarity Ranking Function

For any query vector $\vec{Q}$ and document vector $\vec{D}$:

$$\text{Cosine Similarity}(\vec{Q}, \vec{D}) = \frac{\vec{Q} \cdot \vec{D}}{\|\vec{Q}\|_2 \|\vec{D}\|_2} = \frac{\sum_{t \in Q \cap D} w_{t,Q} \cdot w_{t,D}}{\sqrt{\sum_{t \in Q} w_{t,Q}^2} \cdot \sqrt{\sum_{t \in D} w_{t,D}^2}}$$

---

## ⚙️ Core Enhancements & Engineering Highlights

1. **Multi-Scheme Comparative Analysis Mode (`/compare`)**:
   - Executes queries simultaneously across all four SMART schemes (`ntc`, `ltc`, `atc`, `btc`).
   - Automatically computes **Spearman's Rank Correlation Coefficient ($\rho$)** across scheme pairs to measure ranking concordance and rank shifts:
     $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
   - Displays a side-by-side consensus matrix highlighting rank inversions caused by document verbosity.

2. **Full Diagnostic Scoring & Term Attribution**:
   - For every retrieved result, the engine breaks down the exact term contribution ($w_{t,Q} \times w_{t,D}$) that led to the final cosine score.

3. **Dynamic Contextual Snippet Extraction**:
   - Analyzes sentences in candidate papers to identify the window with the highest concentration of query terms, rendering them with soft green highlights.

4. **Corpus Explorer & Interactive Document Viewer**:
   - Complete document metadata modal (`/api/document/<id>`) allowing instant reading of paper abstracts and excerpts directly in the UI.

5. **Soft Black-Red-Green Aesthetic**:
   - Modern, eye-friendly cyber-academic dark palette:
     - Matte obsidian base (`#08090c`)
     - Soft glowing ruby red (`#ff4365`) for query focus, scheme indicators, and identity accents
     - Soft glowing emerald green (`#00e599`) for cosine similarity scores, relevance meters, and matched term highlights
     - Clean typography (Inter, JetBrains Mono) with frosted glassmorphism surfaces.

---

## 📂 Project Directory Structure

```plaintext
Vector-Space-Model-VSM--main/
├── ResearchPapers/                 # Document Corpus (20 CS Research Papers)
│   ├── 1.txt, 2.txt, 3.txt, ...    # Explainable AI, Time Series, Transformers, Feature Selection
│   └── 26.txt
├── static/
│   ├── css/
│   │   └── style.css               # Black Red Green Soft Aesthetic stylesheet
│   └── js/
│       └── app.js                  # Dynamic client controller, modal viewer, shortcuts
├── templates/
│   ├── base.html                   # Master layout with student header and footer
│   ├── search.html                 # Search portal with scheme selector pills & metrics
│   ├── results.html                # Unified single-scheme & multi-scheme results view
│   ├── compare.html                # Side-by-side scheme comparison & correlation matrix
│   └── analytics.html              # Collection analytics, top DF terms, SMART guide
├── vsm_engine.py                   # High-performance IR engine, inverted indexer & math core
├── Vector_Space_Model.py            # Main Flask web application & REST controller
├── vsm_cache.pkl                   # Precomputed index & vector cache (sub-millisecond load)
├── inverted_indexA2.txt            # Formatted inverted index text export
├── TF-IDF.txt                      # Precomputed TF-IDF text export
└── README.md                       # Comprehensive project documentation
```

---

## 🚀 Installation & Usage Guide

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1. Install Dependencies
```bash
pip install flask nltk
```

### 2. Launch the Application
Run the main application script:
```bash
python Vector_Space_Model.py
```

The system will initialize the inverted index (or load `vsm_cache.pkl` instantaneously) and output:
```plaintext
================================================================================
 Vector Space Model Search Engine: Comparing TF-IDF Variant Schemes and Cosine Similarity
 Author: Devanarayan C S | Roll No: 23FE10CCE00110
 B.Tech in Computer and Communication Engineering | Manipal University
================================================================================
Corpus Loaded: 20 Research Papers | Vocabulary: 16,068 Unique Terms

Starting Flask web server on http://localhost:5000 ...
```

### 3. Access in Web Browser
Open your browser and navigate to:
```text
http://localhost:5000
```

---

## 🔍 Key Search Workflows

### 1. Single Scheme Search
Select `ntc`, `ltc`, `atc`, or `btc` on the search bar. View cosine scores, term weights, and relevant snippets.

### 2. Multi-Scheme Comparison
Select **⚡ Multi-Scheme (All 4)**. The engine evaluates all 4 variants side-by-side, reports Spearman's rank correlation ($\rho$), and reveals rank shifts between long and short papers.

### 3. Collection Analytics & SMART Guide
Click **📊 Corpus & SMART Guide** in the top navigation to view vocabulary coverage, top terms by Document Frequency, and the theoretical SMART reference table.

### 4. Interactive Document Reader
Click **"View Full Paper"** on any search result to open an instant modal displaying the paper's title, authors, venue, extracted abstract, and full text excerpt.

---

## 👨‍🎓 Project Credits & Attribution

- **Student Name:** Devanarayan C S
- **Roll Number:** 23FE10CCE00110
- **Program:** B.Tech in Computer and Communication Engineering (CCE), 4th Year
- **Department:** Department of Computer and Communication Engineering
- **Institution:** Manipal University
- **Course:** Information Retrieval Mini Project
