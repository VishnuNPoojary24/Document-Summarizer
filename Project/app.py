from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Text Preprocessing and Summarization Functions

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    return tokens

def calculate_word_frequencies(tokens):
    word_frequencies = Counter(tokens)
    max_frequency = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency
    return word_frequencies

def score_sentences(text, word_frequencies):
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
    return sentence_scores

def summarize_text(text, n=3):
    tokens = preprocess_text(text)
    word_frequencies = calculate_word_frequencies(tokens)
    sentence_scores = score_sentences(text, word_frequencies)
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
    summary = ' '.join(summarized_sentences)
    return summary

# Visualization Functions

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.tight_layout()
    plt.savefig('./static/wordcloud.png')  # Save word cloud image
    plt.close()

def generate_bar_chart(word_frequencies):
    most_common_words = word_frequencies.most_common(10)
    words = [word for word, freq in most_common_words]
    freqs = [freq for word, freq in most_common_words]

    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Words by Frequency')
    plt.tight_layout()
    plt.savefig('./static/barchart.png')  # Save bar chart image
    plt.close()

def generate_pie_chart(word_frequencies):
    most_common_words = word_frequencies.most_common(10)
    words = [word for word, freq in most_common_words]
    freqs = [freq for word, freq in most_common_words]

    plt.figure(figsize=(10, 5))
    plt.pie(freqs, labels=words, autopct='%1.1f%%', startangle=140)
    plt.title('Top 10 Words Frequency Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig('./static/piechart.png')  # Save pie chart image
    plt.close()

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']

        # Perform text summarization
        summary = summarize_text(text)

        # Generate visualizations
        summary_tokens = preprocess_text(summary)
        summary_word_frequencies = calculate_word_frequencies(summary_tokens)
        generate_word_cloud(summary)
        generate_bar_chart(summary_word_frequencies)
        generate_pie_chart(summary_word_frequencies)

        # Render template with results
        return render_template('result.html', text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
