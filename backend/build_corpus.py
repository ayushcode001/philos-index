import os
import re
import pandas as pd
import spacy
import textstat
from wordfreq import zipf_frequency
import requests


# Intializing NLP
print("Loading spaCy model(en_core_web_lg)")
nlp = spacy.load('en_core_web_lg')

# Load the philosophical lexicon for the PVD labelling component
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
    print("  [Warning] abstract_lexicon.txt not found. PVD component will be 0.")

# Configuring book source
# Format --> {'Title': (URL, Author_rank, stratum)}

SOURCES = {
    # Stratum 1 (Target: 0-20)
    'Alice in Wonderland': ("https://www.gutenberg.org/cache/epub/11/pg11.txt", 10, "Foundational"),
    "Aesop's Fables": ("https://www.gutenberg.org/cache/epub/21/pg21.txt", 15, "Foundational"),
    "Grimms' Fairy Tales": ("https://www.gutenberg.org/cache/epub/2591/pg2591.txt", 12, "Foundational"),
    "Peter Pan": ("https://www.gutenberg.org/cache/epub/16/pg16.txt", 18, "Foundational"),

    # Stratum 2 (Target: 20-40)
    'Adventures of Tom Sawyer': ("https://www.gutenberg.org/cache/epub/74/pg74.txt", 30, "General Prose"),
    'The Gift of the Magi (O. Henry)': ("https://www.gutenberg.org/cache/epub/7256/pg7256.txt", 35, "General Prose"),
    'Huckleberry Finn (Mark Twain)': ("https://www.gutenberg.org/cache/epub/76/pg76.txt", 32, "General Prose"),
    'Treasure Island (Stevenson)': ("https://www.gutenberg.org/cache/epub/120/pg120.txt", 38, "General Prose"),

    # Stratum 3: Literary Fiction (Target: 40-60)
    "A Tale of Two Cities": ("https://www.gutenberg.org/cache/epub/98/pg98.txt", 50, "Literary Fiction"),
    "Crime and Punishment (Dostoevsky)": ("https://www.gutenberg.org/cache/epub/2554/pg2554.txt", 55, "Literary Fiction"),
    "Frankenstein": ("https://www.gutenberg.org/cache/epub/84/pg84.txt", 48, "Literary Fiction"),
    "Pride and Prejudice": ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", 45, "Literary Fiction"),

    # Stratum 4: Dense Fiction & Essays (Target: 60-80)
    "Metamorphosis (Kafka)": ("https://www.gutenberg.org/cache/epub/5200/pg5200.txt", 72, "Dense Fiction & Essays"),
    "Nature (Ralph Waldo Emerson)": ("https://www.gutenberg.org/cache/epub/29433/pg29433.txt", 78, "Dense Fiction & Essays"),
    "Walden (Thoreau)": ("https://www.gutenberg.org/cache/epub/205/pg205.txt", 68, "Dense Fiction & Essays"),
    "The Prince (Machiavelli)": ("https://www.gutenberg.org/cache/epub/1232/pg1232.txt", 65, "Dense Fiction & Essays"),

    # Stratum 5: Philosophical Treatises (Target: 80-100)
    "Beyond Good and Evil (Nietzsche)": ("https://www.gutenberg.org/cache/epub/4363/pg4363.txt", 90, "Philosophical Treatises"),
    "Critique of Pure Reason (Kant)": ("https://www.gutenberg.org/cache/epub/4280/pg4280.txt", 95, "Philosophical Treatises"),
    "Ethics (Spinoza)": ("https://www.gutenberg.org/cache/epub/3800/pg3800.txt", 92, "Philosophical Treatises"),
    "Philosophy of Mind (Hegel)": ("https://www.gutenberg.org/cache/epub/39064/pg39064.txt", 98, "Philosophical Treatises"),
    "Meditations (Descartes)": ("https://www.gutenberg.org/cache/epub/59/pg59.txt", 88, "Philosophical Treatises"),
    "Tractatus Logico-Philosophicus (Wittgenstein)": (
        "https://www.gutenberg.org/cache/epub/5740/pg5740.txt", 94, "Philosophical Treatises"),
    "The World as Will and Representation (Schopenhauer)": (
        "https://www.gutenberg.org/cache/epub/38427/pg38427.txt", 93, "Philosophical Treatises"),
    "Pragmatism (William James)": (
        "https://www.gutenberg.org/cache/epub/5116/pg5116.txt", 83, "Philosophical Treatises"),
    "Leviathan (Hobbes)": (
        "https://www.gutenberg.org/cache/epub/3207/pg3207.txt", 84, "Philosophical Treatises"),
    "An Essay Concerning Human Understanding (Locke)": (
        "https://www.gutenberg.org/cache/epub/10615/pg10615.txt", 86, "Philosophical Treatises"),
    "Utilitarianism (Mill)": (
        "https://www.gutenberg.org/cache/epub/11224/pg11224.txt", 82, "Philosophical Treatises"),
}

def clean_text(raw_text):
    '''Striping the Headers and footers'''

    start_point = re.search(r'\*\*\*\s*START OF THE PROJECT GUTENBERG.*?\*\*\*', raw_text, re.IGNORECASE)

    if start_point:
        # Slice the text after the start_point
        text = raw_text[start_point.end():]
    else:
        print(' [Warning] Standard Start marker missing. Applying fallback slice')
        text = raw_text[2000:]

    end_point = re.search(r'\*\*\*\s*END OF THE PROJECT GUTENBERG.*?\*\*\*', text, re.IGNORECASE)

    if end_point:
        text = text[:end_point.start()]
    else:
        fallback_end = re.search(r'End of the Project Gutenberg', text, re.IGNORECASE)
        if fallback_end:
            text = text[:fallback_end.start()]
        else:
            print(" [Warning] Standard End marker missing. Applying fallback slice")
            text = text[:-20000] if len(text) > 40000 else text
    
    return " ".join(text.split())[1000:]



def calculate_zipf_penalty(doc):
    '''Calculate vocab rareness'''

    CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    content_words = [token.text.lower() for token in doc if token.pos_ in CONTENT_POS and token.is_alpha]

    if not content_words:
        return 0
    
    # Inverting the zipf scale (7 --> common, 1 --> rare). Higher penalty for rare words
    penalty_scores = [(7 - zipf_frequency(w, 'en')) for w in content_words]

    return sum(penalty_scores) / len(penalty_scores)

def calculate_pvd_norm(doc):
    """
    Normalized Philosophical Vocabulary Density for the labelling formula.
    Returns a 0-100 score: 5%+ philosophical token density → score of 100.
    This component lifts Heidegger/Sartre-style texts even when GunningFog
    is moderate (because they use ordinary words philosophically).
    """
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return 0.0
    phil_count = sum(1 for t in tokens if t in PHILOSOPHICAL_LEXICON)
    pvd_raw = (phil_count / len(tokens)) * 100
    # Normalize: 5% philosophical token density = score of 100
    return min((pvd_raw / 5.0) * 100, 100.0)


def generate_hybrid_score(text, doc, author_rank):
    '''Implements the Philos Ground Truth Formula'''

    # Gunning Fog (Capped at 30, normalized to 100)
    gf_raw = textstat.gunning_fog(text)
    gf_norm = min((gf_raw / 30.0) * 100, 100)

    # zipf Penalty (Normalized assumption: max average penalty is around 4.5)
    zipf_raw = calculate_zipf_penalty(doc)
    zipf_norm = min((zipf_raw / 4.5) * 100, 100)

     # Philosophical Vocabulary Density
    pvd_norm = calculate_pvd_norm(doc)

    # Formulae: y = 0.25(GF) + 0.38(AR) + 0.17(ZP) + 0.20(PVD)
    final_score = (0.25 * gf_norm) + (0.38 * author_rank) + (0.17 * zipf_norm) + (0.20 * pvd_norm)
    return round(min(final_score, 100.0), 2)


def process_books():
    corpus_data = []

    os.makedirs('../data', exist_ok=True)

    TARGET_ROWS = 500 
    TARGET_PER_BOOK = 125 # books per stratum * 125 = 500 [For data balancing]

    stratum_counts = {
        "Foundational": 0,
        "General Prose": 0,
        "Literary Fiction": 0,
        "Dense Fiction & Essays": 0,
        "Philosophical Treatises": 0
    }

    for title, (url, author_rank, stratum) in SOURCES.items():
        print(f'\nFetching: {title}...')
        try:
            response = requests.get(url, timeout=15)
            response.encoding = 'utf-8'
            clean_content = clean_text(response.text)

        except Exception as e:
            print(f' Failed to fetch {title}: {e}')
            continue

        print(f'Segmenting {title} (This will take a moment)')

        # Processing a slice to prevent RAM overload while ensuring enough data 
        nlp.max_length = len(clean_content) + 100000
        doc = nlp(clean_content)

        sentences = list(doc.sents)

        # Sliding window setup (targer approx 300 words per snippet)
        current_snippet = []
        word_count = 0
        snippet_count = 0
        book_count = 0

        for sent in sentences:
            if stratum_counts[stratum] >= TARGET_ROWS or book_count >= TARGET_PER_BOOK:
                break

            current_snippet.append(sent.text)
            word_count += len(sent)

            if word_count >= 300:
                snippet_text = " ".join(current_snippet)
                snippet_doc = nlp(snippet_text)

                score = generate_hybrid_score(snippet_text, snippet_doc, author_rank)

                corpus_data.append({
                    'text': snippet_text,
                    'complexity_score': score,
                    'source': title,
                    'stratum': stratum
                })

                snippet_count += 1
                stratum_counts[stratum] += 1
                book_count += 1

                # stride of roughly 100 words (keep last few sentences)
                current_snippet = current_snippet[-3:]
                word_count = sum(len(nlp(s)) for s in current_snippet)


        print(f' Extracted {snippet_count} valid snippet')

    df = pd.DataFrame(corpus_data)
    csv_path = '../data/corpus.csv'
    df.to_csv(csv_path, index=False)

    print('\n' + '='*50)
    print("\n" + "="*50)
    print(f"SUCCESS! Total snippets generated: {len(df)}")
    print(f"Dataset saved to: {os.path.abspath(csv_path)}")
    print("="*50)

if __name__ == '__main__':
    process_books()