
import os
import re
import pandas as pd
import numpy as np
from wordfreq import zipf_frequency
from nltk.corpus import wordnet as wn


# WordNet lexnames that represent abstract concepts
ABSTRACT_LEXNAMES = {
    'noun.attribute', 'noun.cognition', 'noun.communication',
    'noun.feeling', 'noun.motive', 'noun.phenomenon',
    'noun.relation', 'noun.state', 'noun.time', 'noun.act'
}

# Load the philosophical lexicon
PHILOSOPHICAL_LEXICON = set()
lexicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abstract_lexicon.txt")
if os.path.exists(lexicon_path):
    with open(lexicon_path) as f:
        for line in f:
            term = line.strip().lower()
            if term and not term.startswith('#'):
                PHILOSOPHICAL_LEXICON.add(term)
    print(f"  Loaded {len(PHILOSOPHICAL_LEXICON)} terms from philosophical lexicon.")
else:
    print(f"  [Warning] abstract_lexicon.txt not found at {lexicon_path}. AR/VRS/PVD will be degraded.")

# Compound prefixes used in continental philosophy
# (Heidegger: Being-in-the-world, ready-to-hand; Sartre: for-itself, in-itself)
PHIL_COMPOUND_PREFIXES = {
    'being', 'in', 'with', 'towards', 'for', 'at', 'ready',
    'present', 'self', 'non', 'inter', 'over', 'under', 'pre',
    'meta', 'onto', 'tele', 'proto', 'post', 'anti', 'co',
    'trans', 'de', 'super', 'sub', 'pseudo'
}


# =============================================================================
# FEATURE FUNCTIONS
# =============================================================================

def get_sdi(doc):
    """
    Syntactic Depth Index: Average token distance from its sentence root.
    Measures how deeply embedded tokens are in the parse tree.
    Higher = more complex nested grammatical structures.
    """
    depths = []
    for sent in doc.sents:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                current = current.head
                depth += 1
            depths.append(depth)
    return np.mean(depths) if depths else 0.0


def get_scd(doc):
    """
    Subclausal Density: Number of subordinate clauses per sentence.
    Counts dependent clause types (relative, adverbial, complement, subject).
    Higher = more clause embedding per sentence.
    """
    clause_tags = {'relcl', 'advcl', 'ccomp', 'csubj', 'xcomp'}
    clauses = sum(1 for token in doc if token.dep_ in clause_tags)
    sentence_count = len(list(doc.sents))
    return clauses / sentence_count if sentence_count > 0 else 0.0


def get_ar(doc):
    """
    Abstractness Ratio: Fraction of nouns that are abstract.
    Priority: (1) Philosophical Lexicon, (2) WordNet lexname classification.
    Higher = more nouns refer to intangible concepts.
    """
    nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and token.is_alpha]
    if not nouns:
        return 0.0

    abstract_count = 0
    for noun in nouns:
        # Priority 1: check the philosophical lexicon (higher precision for domain terms)
        if noun in PHILOSOPHICAL_LEXICON:
            abstract_count += 1
            continue
        # Priority 2: WordNet lexname classification
        synsets = wn.synsets(noun, pos=wn.NOUN)
        if synsets and synsets[0].lexname() in ABSTRACT_LEXNAMES:
            abstract_count += 1

    return abstract_count / len(nouns)


def get_ld(doc):
    """
    Lexical Density: Ratio of content words (N/V/ADJ/ADV) to all words.
    Higher = more information per word (denser prose).
    """
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    tokens = [t for t in doc if t.is_alpha]
    if not tokens:
        return 0.0
    content_words = [t for t in doc if t.pos_ in content_pos]
    return len(content_words) / len(tokens)


def get_vrs(doc):
    """
    Vocabulary Rareness Score: Inverted Zipf average, with philosophical
    lexicon terms given a 2x weight and treated as maximally rare (score 6.5).
    Prevents dense philosophical vocabulary from being diluted by
    the common connector words philosophers use deliberately.
    """
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return 0.0

    phil_scores = []
    common_scores = []

    for w in set(tokens):
        if w in PHILOSOPHICAL_LEXICON:
            phil_scores.append(6.5)   # treat domain terms as extremely rare
        else:
            common_scores.append(7.0 - zipf_frequency(w, 'en'))

    if not phil_scores and not common_scores:
        return 0.0
    if not phil_scores:
        return float(np.mean(common_scores))
    if not common_scores:
        return float(np.mean(phil_scores))

    # Weighted blend: philosophical terms count double
    phil_weight = 2.0
    common_weight = 1.0
    numerator = (phil_weight * np.mean(phil_scores) * len(phil_scores)) + \
                (common_weight * np.mean(common_scores) * len(common_scores))
    denominator = (phil_weight * len(phil_scores)) + (common_weight * len(common_scores))
    return float(numerator / denominator)


def get_mattr(doc, window_size=50):
    """
    Moving Average Type-Token Ratio: Lexical diversity robust to text length.
    Uses a sliding window of 50 tokens to avoid length bias.
    """
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if len(tokens) < window_size:
        return (len(set(tokens)) / len(tokens)) if tokens else 0.0

    ttr_values = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttr_values.append(len(set(window)) / window_size)
    return float(np.mean(ttr_values))


def get_parse_tree_depth(doc):
    """
    Parse Tree Depth (PTD): Average maximum depth of each sentence's
    dependency parse tree, measured from root to deepest leaf.
    Complements SDI by capturing worst-case nesting per sentence.
    """
    def walk_tree(node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(walk_tree(child, depth + 1) for child in node.children)
        return depth

    depths = [walk_tree(sent.root, 0) for sent in doc.sents]
    return round(sum(depths) / len(depths), 2) if depths else 0.0


def get_philosophical_compound_density(doc):
    """
    Philosophical Compound Density (PCD): Hyphenated ontological compounds
    per sentence. Only counts compounds where a component matches a known
    philosophical prefix OR a term in the philosophical lexicon.

    Captures: Being-in-the-world, ready-to-hand, for-itself, in-itself.
    Ignores: sun-kissed, tide-water, wide-spreading (literary/physical hyphens).
    """
    text = doc.text
    compound_pattern = re.compile(r'\b[A-Za-z]+-[A-Za-z]+(?:-[A-Za-z]+)*\b')
    all_compounds = compound_pattern.findall(text)

    phil_compounds = []
    for compound in all_compounds:
        parts = compound.lower().split('-')
        # Accept if first component is a known philosophical prefix
        # OR any component is in the philosophical lexicon
        if (parts[0] in PHIL_COMPOUND_PREFIXES or
                any(p in PHILOSOPHICAL_LEXICON for p in parts)):
            phil_compounds.append(compound)

    sentence_count = len(list(doc.sents))
    return len(phil_compounds) / sentence_count if sentence_count > 0 else 0.0


def get_abstract_noun_repetition(doc):
    """
    Abstract Noun Repetition (ANR): Ratio of abstract noun tokens to unique
    abstract noun types. Captures Heidegger's signature rhetorical device —
    heavy reuse of a small set of ontological terms (Dasein × 7, Being × 12).
    High ANR + high AR together = strong philosophical argumentation signal.
    """
    abstract_nouns = []
    for token in doc:
        if token.pos_ == 'NOUN' and token.is_alpha:
            lemma = token.lemma_.lower()
            if lemma in PHILOSOPHICAL_LEXICON:
                abstract_nouns.append(lemma)
            else:
                synsets = wn.synsets(lemma, pos=wn.NOUN)
                if synsets and synsets[0].lexname() in ABSTRACT_LEXNAMES:
                    abstract_nouns.append(lemma)

    if not abstract_nouns:
        return 0.0
    return len(abstract_nouns) / len(set(abstract_nouns))


def get_pvd(doc):
    """
    Philosophical Vocabulary Density (PVD): Count of philosophical lexicon
    terms per 100 tokens. Directly measures ontological vocabulary load
    without being diluted by surrounding connector words.
    Near-zero for literary fiction; 2-6+ for philosophical treatises.
    """
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return 0.0
    phil_count = sum(1 for t in tokens if t in PHILOSOPHICAL_LEXICON)
    return (phil_count / len(tokens)) * 100


def get_slv(doc):
    """
    Sentence Length Variance (SLV): Standard deviation of sentence lengths.
    Philosophical writing deliberately mixes very long recursive sentences
    with short aphoristic declarations ('But the they is there.').
    Narrative prose is comparatively uniform.
    High SLV = hallmark of philosophical stylistic deliberateness.
    """
    lengths = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


# =============================================================================
# FEATURE EXTRACTION PIPELINE
# =============================================================================

def process_features():
    # Load spaCy only when running this script standalone (not when imported by main.py)
    import spacy
    nlp = spacy.load('en_core_web_lg')
    print("spaCy model loaded for batch processing.")

    csv_path = "../data/corpus.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Run build_corpus.py first.")
        return

    print("Loading corpus data...")
    df = pd.read_csv(csv_path)

    feature_rows = []
    total_rows = len(df)

    print(f"Extracting features for {total_rows} snippets...")

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  -> Processed {idx + 1} / {total_rows} snippets...")

        text = row['text']
        doc = nlp(text)

        feature_rows.append({
            "text":             text,
            "source":           row['source'],
            # Handle the historical typo 'stratrum' gracefully
            "stratum":          row.get('stratum', row.get('stratrum', 'Unknown')),
            "complexity_score": row['complexity_score'],
            "SDI":  round(get_sdi(doc), 3),
            "SCD":  round(get_scd(doc), 3),
            "AR":   round(get_ar(doc), 3),
            "LD":   round(get_ld(doc), 3),
            "VRS":  round(get_vrs(doc), 3),
            "MATTR":round(get_mattr(doc), 3),
            "PTD":  round(get_parse_tree_depth(doc), 3),
            "PCD":  round(get_philosophical_compound_density(doc), 3),
            "ANR":  round(get_abstract_noun_repetition(doc), 3),
            "PVD":  round(get_pvd(doc), 3),
            "SLV":  round(get_slv(doc), 3),
        })

    features_df = pd.DataFrame(feature_rows)
    output_path = "../data/features.csv"
    features_df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(f"SUCCESS! Feature engineering complete.")
    print(f"Data shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")
    print(f"Saved to: {os.path.abspath(output_path)}")
    print("=" * 50)


if __name__ == "__main__":
    process_features()