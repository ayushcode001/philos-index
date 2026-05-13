# The Philomath Index
## Linguistic Regression for Philosophical Density
### Complete Project Problem Statement & Technical Specification

---

> *"The limit of my language is the limit of my world."*
> — Ludwig Wittgenstein

---

## Abstract

Standard readability metrics — Flesch-Kincaid, Gunning Fog, SMOG — measure surface-level difficulty using syllable counts and sentence lengths. These instruments, designed for journalism and instructional design, are categorically blind to the phenomenon of **conceptual compression**: the tendency of philosophical, theological, and literary prose to encode extraordinary intellectual weight into syntactically simple structures.

A sentence by Nietzsche may contain fewer words than a sentence from a children's primer, yet carry exponentially greater cognitive load due to its dense web of abstract nouns, inverted presuppositions, and cross-referential argumentation. Standard readability scores would rank Nietzsche's *Thus Spoke Zarathustra* as "easy" — a category error that exposes the fundamental inadequacy of syllable-counting as a proxy for intellectual complexity.

**The Philomath Index** addresses this gap. It is a regression-based scoring engine — trained on a curated corpus of philosophical, literary, and general-audience texts — that predicts a **Complexity Score (0–100)** by modelling syntactic depth, lexical rarity, and abstractness ratios as first-class features, producing a score that is linguistically grounded, interpretable, and empirically validated.

---

## 1. Problem Statement

### 1.1 The Core Limitation of Existing Metrics

| Metric | Basis | Limitation |
|---|---|---|
| Flesch-Kincaid | Syllables + sentence length | Cannot detect abstract vocabulary |
| Gunning Fog | Complex word ratio | "Complex" = polysyllabic, not abstract |
| SMOG | Polysyllabic words | Fails on short philosophical sentences |
| Dale-Chall | Known word lists | Static list; no semantic understanding |

None of these capture **semantic density** — the quantity of abstract concepts packed per unit of text.

### 1.2 The Hypothesis

Linguistic complexity in philosophical texts is a multi-dimensional phenomenon governed by at least three orthogonal axes:

1. **Syntactic Depth** — How deeply nested are the grammatical structures?
2. **Lexical Rarity** — How infrequent are the words in general language use?
3. **Abstractness Ratio** — What proportion of nouns refer to intangible concepts vs. physical objects?

A regression model that encodes these three axes as engineered features — derived via dependency parsing and corpus-frequency statistics — can predict perceived intellectual complexity with significantly greater fidelity than any surface-level formula.

### 1.3 Formal Problem Definition

> **Given** a natural language text snippet $T$ of arbitrary length,
> **predict** a continuous scalar $\hat{y} \in [0, 100]$ representing its **Philomath Complexity Score**, where:
> - $0$ represents maximally simple text (children's literature, basic instructions), and
> - $100$ represents maximally dense philosophical or theoretical prose.

---

## 2. Dataset Design & Ground Truth Construction

### 2.1 Corpus Architecture

The training corpus spans five **difficulty strata**, each with multiple representative sources to prevent author-specific overfitting:

| Stratum | Score Band | Example Sources |
|---|---|---|
| **Stratum 1 – Foundational** | 0–20 | Project Gutenberg children's classics, Aesop's Fables, nursery rhyme collections |
| **Stratum 2 – General Prose** | 20–40 | 19th-century journalism, popular essays, Mark Twain, O. Henry |
| **Stratum 3 – Literary Fiction** | 40–60 | Dickens, Hardy, Tolstoy (translated), Dostoevsky's narrative passages |
| **Stratum 4 – Dense Fiction & Essays** | 60–80 | Kafka, Woolf, Borges, Nietzsche's aphorisms, Emerson |
| **Stratum 5 – Philosophical Treatises** | 80–100 | Hegel, Heidegger, Krishnamurti, Wittgenstein's *Tractatus*, Kant (translated) |

> **Minimum target:** 5,000 labelled snippets, 200–500 words each, stratified across all five strata.

### 2.2 Hybrid Labelling Strategy

Initial labels $y_i$ are generated using a **three-component hybrid formula** to avoid relying solely on any single heuristic:

$$y_i = \alpha \cdot \text{GunningFog}(T_i) + \beta \cdot \text{AuthorRank}(T_i) + \gamma \cdot \text{ZipfPenalty}(T_i)$$

Where:
- $\alpha = 0.35$ — Weight for Gunning Fog (normalised to 0–100 scale)
- $\beta = 0.40$ — Weight for hand-assigned author difficulty rank (the "prior")
- $\gamma = 0.25$ — Weight for Zipf-frequency penalty (rare vocabulary bonus)

**Author Difficulty Rank Table (sample):**

| Author / Source | Assigned Rank |
|---|---|
| Children's nursery texts | 5 |
| Mark Twain, O. Henry | 30 |
| Dickens, Hardy | 50 |
| Dostoevsky, Kafka | 72 |
| Nietzsche, Emerson | 82 |
| Hegel, Heidegger, Kant | 95 |

> **Human Validation:** A random 10% sample ($n \approx 500$) should be independently scored by annotators with a philosophy background. Inter-annotator agreement (Cohen's $\kappa$) is the quality gate for label reliability.

### 2.3 Data Acquisition Pipeline

```
gutenberg (Python library)
    └── Fetch raw .txt by author/title
    └── Sentence-segment with spaCy
    └── Chunk into 200–500 word windows (sliding, stride=100)
    └── Deduplicate (Jaccard similarity > 0.85 → discard)
    └── Apply hybrid labelling formula
    └── Export as: corpus.csv [snippet_id, text, score, source, stratum]
```

---

## 3. Feature Engineering (The ML Core)

This is the intellectual centrepiece of the project. Six feature families are extracted from each text snippet:

### Feature 1 — Syntactic Depth Index (SDI)

**What it captures:** How many levels of grammatical subordination exist per sentence.

**Method:** Use spaCy dependency parse trees. For each token, compute its depth from the sentence root via `.head` traversal. Average across all tokens, then average across sentences.

```python
def syntactic_depth(doc):
    depths = []
    for sent in doc.sents:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                current = current.head
                depth += 1
            depths.append(depth)
    return np.mean(depths)
```

### Feature 2 — Subclausal Density (SCD)

**What it captures:** Average number of subordinate clauses per sentence (relative clauses, adverbial clauses, complement clauses).

**Method:** Count tokens with `.dep_` in `{"relcl", "advcl", "ccomp", "xcomp", "acl"}` per sentence.

### Feature 3 — Abstractness Ratio (AR)

**What it captures:** The ratio of abstract nouns to total nouns.

**Method:**
- Use WordNet (`nltk`) to classify nouns: abstract nouns tend to lack hypernyms that eventually reach `physical_entity.n.01`.
- Supplement with a curated **Philosophical Abstract Noun Lexicon** — a hand-compiled list of ~500 philosophy-domain abstractions (e.g., *Being, Essence, Dasein, Will, Dialectic, Apriori, Noumenon*) that WordNet alone may misclassify.

$$\text{AR} = \frac{\text{Count(Abstract Nouns)}}{\text{Count(Total Nouns)} + \epsilon}$$

### Feature 4 — Lexical Density (LD)

**What it captures:** The ratio of content words (nouns, verbs, adjectives, adverbs) to total words. Higher density = more information per word.

```python
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
LD = sum(1 for t in doc if t.pos_ in CONTENT_POS) / len(doc)
```

### Feature 5 — Vocabulary Rareness Score (VRS)

**What it captures:** How infrequent the words are in standard English usage.

**Method:** Use the `wordfreq` library to get Zipf frequency scores for each content word. Zipf scale: 6+ = very common, below 3 = rare/academic.

$$\text{VRS} = \frac{1}{|C|} \sum_{w \in C} (7 - \text{zipf\_frequency}(w, 'en'))$$

Where $C$ is the set of content words. A higher VRS indicates a vocabulary further from common usage.

### Feature 6 — Type-Token Ratio (TTR) with Correction

**What it captures:** Lexical diversity — how many unique words are used relative to total words.

**Method:** Use MATTR (Moving Average Type-Token Ratio) with a sliding window of 50 tokens to control for text length bias:

$$\text{MATTR} = \frac{1}{N - w} \sum_{i=1}^{N-w} \frac{|\text{unique tokens in window}_i|}{w}$$

### Summary: Feature Vector

Each snippet $T_i$ is transformed into a feature vector $\mathbf{x}_i \in \mathbb{R}^6$:

$$\mathbf{x}_i = [\text{SDI}, \text{SCD}, \text{AR}, \text{LD}, \text{VRS}, \text{MATTR}]$$

> **Bonus Feature (optional):** Sentence length variance — philosophical texts often mix very long, recursive sentences with short, aphoristic declarations. High variance itself signals stylistic deliberateness.

---

## 4. Model Selection & Training

### 4.1 Candidate Models

| Model | Rationale |
|---|---|
| **Ridge Regression** | Fast baseline; handles collinear linguistic features via L2 regularisation |
| **Support Vector Regression (RBF kernel)** | Captures non-linear relationships in the feature space |
| **Gradient Boosting Regressor (XGBoost)** | Handles feature interactions; strong empirical performer on tabular data |
| **Elastic Net** | Useful if feature selection is desired (L1 + L2 combined) |

> **Recommendation:** Train all four, compare on held-out test set. Use SVR-RBF or XGBoost as the final model, with Ridge as the interpretable fallback.

### 4.2 Training Protocol

```
Dataset Split:
  ├── Train:      70%  (~3,500 snippets)
  ├── Validation: 15%  (~750 snippets) — hyperparameter tuning
  └── Test:       15%  (~750 snippets) — final evaluation only (touch once)

Cross-Validation: 5-fold on training set
Hyperparameter Tuning: GridSearchCV / RandomizedSearchCV
```

### 4.3 Evaluation Metrics

| Metric | Target Threshold | Rationale |
|---|---|---|
| MAE (Mean Absolute Error) | < 8.0 points | Practical tolerance for a 0–100 scale |
| $R^2$ Score | > 0.75 | Explains ≥75% of variance in complexity |
| Spearman $\rho$ | > 0.80 | Ordinal ranking of snippets should be preserved |
| Stratified MAE | < 10.0 per stratum | No single stratum should be catastrophically wrong |

### 4.4 Interpretability Layer

Integrate **SHAP (SHapley Additive exPlanations)** to decompose each prediction into per-feature contributions:

```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
```

This powers the "Linguistic Fingerprint" breakdown in the frontend — users see *why* a text scored what it scored, not just the score itself.

---

## 5. Backend API Specification

**Framework:** FastAPI (preferred over Flask for async support and auto-generated OpenAPI docs)

### Endpoint: `POST /api/predict`

**Request Body:**
```json
{
  "text": "string (min 50 words, max 2000 words)"
}
```

**Response Body:**
```json
{
  "complexity_score": 74.3,
  "confidence_interval": [68.1, 80.5],
  "stratum": "Dense Fiction & Essays",
  "feature_breakdown": {
    "syntactic_depth_index": 3.82,
    "subclausal_density": 1.74,
    "abstractness_ratio": 0.61,
    "lexical_density": 0.54,
    "vocabulary_rareness": 2.31,
    "mattr": 0.73
  },
  "shap_contributions": {
    "abstractness_ratio": +12.4,
    "vocabulary_rareness": +9.1,
    "syntactic_depth_index": +6.8,
    "lexical_density": +3.2,
    "subclausal_density": +1.9,
    "mattr": -0.7
  },
  "closest_author_profile": "Nietzsche / Emerson tier",
  "processing_time_ms": 143
}
```

**Additional Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check for Docker/orchestration |
| `/api/compare` | POST | Submit two texts; returns side-by-side feature breakdown |
| `/api/corpus/stats` | GET | Returns training corpus statistics and model metadata |

---

## 6. Frontend Experience

**Framework:** Vue.js 3 (Composition API) with Vite build tooling

### 6.1 UI Panels

**Panel 1 — The Editor**
- Distraction-free, monospace text area with live word count
- Real-time score update with debounce (700ms after typing stops)
- "Analyse" button for manual trigger

**Panel 2 — The Complexity Gauge**
- A large, animated arc gauge (0–100)
- Colour-coded bands: Green (0–30) → Amber (30–60) → Deep Indigo (60–85) → Black (85–100)
- Dynamic label showing the nearest "Author Tier" (e.g., *"Dostoevsky territory"*)

**Panel 3 — The Linguistic Fingerprint (Radar Chart)**
- Six axes: Syntactic Depth, Subclausal Density, Abstractness, Lexical Density, Vocabulary Rareness, Lexical Diversity
- Rendered with Chart.js or D3.js
- Animated fill on score reveal

**Panel 4 — Feature Explanation Cards**
- One card per feature, showing its value, its SHAP contribution (±), and a one-sentence plain-language explanation
- Example: *"Abstractness Ratio: 0.61 (+12.4 pts) — Over 60% of nouns refer to intangible concepts, a signature of philosophical prose."*

### 6.2 UX Extras (Differentiators)

- **Comparison Mode:** Two-pane editor, overlay radar charts, delta score display
- **Example Texts Button:** Loads canonical Nietzsche, Kafka, or Hemingway snippets to demonstrate range
- **Export Report:** Download a PDF summary of the analysis

---

## 7. Deployment & DevOps

### 7.1 Project Directory Structure

```
philomath-index/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── predictor.py         # Prediction pipeline
│   │   ├── features.py          # Feature extraction (spaCy)
│   │   ├── model/
│   │   │   ├── model.pkl        # Trained regressor
│   │   │   └── scaler.pkl       # Feature scaler
│   │   └── abstract_lexicon.txt # Custom philosophical noun list
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.vue
│   │   ├── components/
│   │   │   ├── EditorPane.vue
│   │   │   ├── ComplexityGauge.vue
│   │   │   ├── RadarChart.vue
│   │   │   └── FeatureCards.vue
│   │   └── api/client.js
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── data/
│   ├── raw/
│   └── corpus.csv
├── docker-compose.yml
└── README.md
```

### 7.2 Docker Compose Architecture

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/model/model.pkl

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [backend]
```

> **spaCy Model Note:** The `en_core_web_lg` model (~750MB) must be downloaded during the Docker build step. Include `RUN python -m spacy download en_core_web_lg` in the backend Dockerfile.

---

## 8. Additions & Enhancements to Original Spec

The following extensions significantly elevate the project beyond a standard portfolio piece:

### 8.1 Confidence Interval on Prediction

Rather than a point estimate, return a confidence interval derived from the standard deviation of predictions across the cross-validation folds (or bootstrapped predictions). This communicates honest uncertainty — short texts with ambiguous features will have wider intervals.

### 8.2 Author Tier Classification (Bonus Classifier)

Train a secondary **multi-class classifier** (alongside the regressor) that maps the feature vector to an "Author Tier" label (e.g., "General Prose", "Literary Fiction", "Philosophical Treatise"). This dual output makes the UI more intuitive and gives the model a richer story to tell.

### 8.3 The Philosophical Abstract Noun Lexicon

A hand-curated list of ~500 philosophical abstract nouns is a genuine research contribution. WordNet's taxonomy is inconsistent for domain-specific philosophical vocabulary (*Dasein, aporia, dialectic, noumenon*). Publishing this lexicon as an open dataset alongside the project adds academic credibility.

### 8.4 Corpus Bias Audit

Explicitly document the known biases of the training corpus:
- Western philosophical tradition dominates (Kant, Hegel, Nietzsche)
- Texts are primarily translated from German/French/Russian — translation artifacts may affect feature distributions
- No contemporary academic papers or scientific texts (by design, but worth stating)

### 8.5 Adversarial Testing

Include a section in the README with deliberately adversarial examples — texts that *look* complex but score low (e.g., bureaucratic jargon with no abstraction) and texts that *look* simple but score high (e.g., a Wittgenstein proposition). This demonstrates the model's discriminative power and its departure from naive surface metrics.

---

## 9. Success Criteria

The project is considered complete and portfolio-ready when:

- [ ] $R^2 > 0.75$ and $\text{MAE} < 8.0$ on the held-out test set
- [ ] All six features have non-trivial SHAP contributions (no dead features)
- [ ] A Nietzsche snippet scores ≥ 20 points higher than a Hemingway snippet of equal length
- [ ] The API responds in < 500ms for a 500-word input
- [ ] Docker Compose brings the full stack up with a single `docker compose up`
- [ ] The README clearly explains *why* each feature was chosen and what it measures

---

*The Philomath Index — because not all difficulty is in the syllables.*
