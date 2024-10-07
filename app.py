from flask import Flask, request, jsonify, render_template
import os
import nltk
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = nltk.corpus.stopwords.words('english')

# Load and clean documents
def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:  # Add encoding
                documents[filename] = file.read()
    return documents

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return tokens

def create_vocabulary(cleaned_documents):
    vocabulary = set()
    for tokens in cleaned_documents.values():
        vocabulary.update(tokens)
    return vocabulary

def compute_term_frequency(cleaned_documents):
    term_frequency = defaultdict(Counter)
    for filename, tokens in cleaned_documents.items():
        term_frequency[filename] = Counter(tokens)
    return term_frequency

def compute_inverse_document_frequency(cleaned_documents):
    num_documents = len(cleaned_documents)
    df = Counter()
    for tokens in cleaned_documents.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1
    idf = {token: math.log(num_documents / df[token]) for token in df}
    return idf

def compute_tf_idf(term_frequency, idf):
    tf_idf = defaultdict(dict)
    for filename, tf in term_frequency.items():
        for term, freq in tf.items():
            tf_idf[filename][term] = freq * idf[term]
    return tf_idf

def cosine_similarity(vec_a, vec_b):
    intersection = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[x] * vec_b[x] for x in intersection)
    sum_a = sum(vec_a[x] ** 2 for x in vec_a)
    sum_b = sum(vec_b[x] ** 2 for x in vec_b)
    denominator = math.sqrt(sum_a) * math.sqrt(sum_b)
    return 0.0 if not denominator else numerator / denominator

def compute_query_tf_idf(query, idf):
    tokens = clean_text(query)
    tf = Counter(tokens)
    query_tf_idf = {term: freq * idf.get(term, 0) for term, freq in tf.items()}
    return query_tf_idf

# Load and process the documents
documents = load_documents('documents')
cleaned_documents = {filename: clean_text(content) for filename, content in documents.items()}
vocabulary = create_vocabulary(cleaned_documents)
term_frequency = compute_term_frequency(cleaned_documents)
idf = compute_inverse_document_frequency(cleaned_documents)
tf_idf = compute_tf_idf(term_frequency, idf)

# Map each document to an image
image_paths = {
    'All Too Well - Taylor Swift.txt': '/static/photos/All Too Well - Taylor Swift.jpeg',
    'Baby - Justin Bieber & Ludacris.txt': '/static/photos/Baby - Justin Bieber & Ludacris.jpg',
    'Ceilings - Lizzy McApline.txt': '/static/photos/Ceilings - Lizzy McApline.jpg',
    'Die With A Smile - Lady Gaga feat. Bruno Mars.txt': '/static/photos/Die With A Smile - Lady Gaga feat. Bruno Mars.jpg',
    'Espresso - Sabrina Carpenter.txt': '/static/photos/Espresso - Sabrina Carpenter.jpg',
    'Fallin All in You - Shawn Mendes.txt': '/static/photos/Fallin All in You - Shawn Mendes.jpg',
    'Glimpse of Us - Joji.txt': '/static/photos/Glimpse of Us - Joji.jpeg',
    'Just the Way You Are - Bruno Mars.txt': '/static/photos/Just the Way You Are - Bruno Mars.jpg',
    'Lose You To Love Me - Selena Gomez.txt': '/static/photos/Lose You To Love Me - Selena Gomez.jpeg',
    'Natural - Imagine Dragon.txt': '/static/photos/Natural - Imagine Dragon.jpg',
    'Next To You - Chris Brown ft. Justin Bieber.txt': '/static/photos/Next To You - Chris Brown ft. Justin Bieber.jpg',
    'Night Changes - One Direction.txt': '/static/photos/Night Changes - One Direction.jpg',
    'November Rain - Guns N Roses.txt': '/static/photos/November Rain - Guns N Roses.jpg',
    'Supernatural - Ariana Grande.txt': '/static/photos/Supernatural - Ariana Grande.jpg',
    'The Spectre - Alan Walker.txt': '/static/photos/The Spectre - Alan Walker.jpg',
    'This Town - Niall Horan.txt': '/static/photos/This Town - Niall Horan.jpeg',
    'Treat You Better - Shawn Mendes.txt': '/static/photos/Treat You Better - Shawn Mendes.jpg',
    'Unnatural Selection - Muse.txt': '/static/photos/Unnatural Selection - Muse.jpg',
    'Watermelon Sugar - Harry Styles.txt': '/static/photos/Watermelon Sugar - Harry Styles.jpg',
    'We Dont Talk Anymore - Charlie Puth & Selena Gomez.txt': '/static/photos/We Dont Talk Anymore - Charlie Puth & Selena Gomez.jpg',
    'Yellow - Coldplay.txt': '/static/photos/Yellow - Coldplay.jpeg',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    data = request.json
    query = data.get('lyrics', '')

    if not query:
        return jsonify({"message": "No query provided."}), 404
    
    query_tf_idf = compute_query_tf_idf(query, idf)
    if not query_tf_idf:
        return jsonify({"message": "No matching songs found."}), 404
    
    # Compute cosine similarity
    similarities = []
    for filename, doc_tf_idf in tf_idf.items():
        similarity = cosine_similarity(query_tf_idf, doc_tf_idf)
        if similarity > 0:
            similarities.append((filename, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top results
    results = []
    for filename, similarity in similarities[:5]:
        image_path = image_paths.get(filename, '/static/photos/default.jpg')  # Use default image if not found
        results.append({
            'doc_name': filename,
            'similarity': similarity,
            'image_path': image_path,
        })

    if not results:
        return jsonify({"message": "No matching songs found."}), 404
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
