import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from collections import Counter
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Step 1: Download Bengali stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Install system Bengali fonts
import subprocess
import os
import matplotlib.font_manager as fm

print("Installing Bengali fonts...")
try:
    subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
    subprocess.run(['apt-get', 'install', '-y', 'fonts-beng'], check=True, capture_output=True)
    print("Bengali fonts installed successfully.")

    # Rebuild font cache
    fm._rebuild()
except Exception as e:
    print(f"Error installing fonts: {e}")

# Find a Bengali font in the system
print("\nSearching for Bengali fonts...")
system_fonts = fm.findSystemFonts()
font_path = None
bengali_font_prop = None

for font in system_fonts:
    try:
        font_name = os.path.basename(font).lower()
        if any(keyword in font_name for keyword in ['beng', 'noto', 'kalpurush', 'solaiman', 'nikosh']):
            font_path = font
            print(f"Found Bengali font: {font_name}")
            # Create font properties for use in plots
            bengali_font_prop = fm.FontProperties(fname=font_path)
            print(f"Font loaded: {bengali_font_prop.get_name()}")
            break
    except:
        continue

if not font_path:
    print("No Bengali font found. Using default font.")
    bengali_font_prop = None

# Step 2: Load dataset
url = '/content/drive/MyDrive/Research Documents/Updated Work/Shohan Article for REBEC Corpus and my comments/Shafiq Code and Updated paper/Shohan_sentiment_analysis_dataset.csv'
df = pd.read_csv(url)
print("Raw data")
print(df.head())

# ==================== DATASET STATISTICS TABLE ====================
def print_dataset_statistics(df):
    # Calculate statistics
    total_samples = len(df)
    num_columns = len(df.columns)
    num_classes = df['classes'].nunique()

    # Class distribution
    class_counts = df['classes'].value_counts()
    class_distribution = "\n".join([f"{cls}: {count}" for cls, count in class_counts.items()])

    # Text length statistics (words)
    word_lengths = df['cleaned'].apply(lambda x: len(str(x).split()))
    avg_word_length = word_lengths.mean()
    max_word_length = word_lengths.max()
    min_word_length = word_lengths.min()

    # Text length statistics (characters)
    char_lengths = df['cleaned'].apply(lambda x: len(str(x)))
    avg_char_length = char_lengths.mean()
    max_char_length = char_lengths.max()
    min_char_length = char_lengths.min()

    # Vocabulary size
    all_words = ' '.join(df['cleaned'].astype(str)).split()
    vocabulary_size = len(set(all_words))

    # Print statistics table
    print("\n" + "="*50)
    print("DATASET STATISTICS TABLE OF RBEC")
    print("="*50)
    print(f"{'Statistic':<35} {'Value':<15}")
    print("-"*50)
    print(f"{'Total Samples':<35} {total_samples:<15}")
    print(f"{'Number of Columns':<35} {num_columns:<15} ({', '.join(df.columns)})")
    print(f"{'Number of Classes':<35} {num_classes:<15} ({', '.join(df['classes'].unique())})")
    print(f"{'Class Distribution':<35} {class_distribution}")
    print(f"{'Average Text Length (words)':<35} {avg_word_length:.2f}")
    print(f"{'Max Text Length (words)':<35} {max_word_length}")
    print(f"{'Min Text Length (words)':<35} {min_word_length}")
    print(f"{'Average Text Length (characters)':<35} {avg_char_length:.2f}")
    print(f"{'Max Text Length (characters)':<35} {max_char_length}")
    print(f"{'Min Text Length (characters)':<35} {min_char_length}")
    print(f"{'Vocabulary Size':<35} {vocabulary_size}")
    print("="*50)

# Print dataset statistics
print_dataset_statistics(df)

# Step 3: Define Bengali stopwords
bengali_stopwords = set([
    'আমি', 'তুমি', 'সে', 'এই', 'ওই', 'তারা', 'যা', 'কি', 'না', 'হয়', 'করেছে', 'করো', 'করছি', 'করবেন', 'গেছে'
])

# Step 4: Text preprocessing
def preprocess(text):
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in bengali_stopwords]
    return ' '.join(words)

df['cleaned'] = df['cleaned'].astype(str).apply(preprocess)

# ==================== WORD STATISTICS BY CLASS ====================
# Calculate word statistics for each emotion class
word_stats = df.groupby('classes').agg(
    Total_words=('cleaned', lambda x: sum(len(text.split()) for text in x)),
    Unique_words=('cleaned', lambda x: len(set(' '.join(x).split()))),
    Avg_words_per_text=('cleaned', lambda x: np.mean([len(text.split()) for text in x]))
).reset_index()

# Rename columns for better readability
word_stats.columns = ['Class', 'Total words', 'Unique words', 'Avg. words per text']

# Print the statistics table
print("\n=== WORD STATISTICS BY CLASS ===")
print(word_stats.to_string(index=False))

# Visualize the word statistics
plt.figure(figsize=(15, 12))
print("\n[Plot 1] Displaying Word Statistics by Class...")

# Total words per class
plt.subplot(3, 1, 1)
sns.barplot(data=word_stats, x='Class', y='Total words', palette='viridis')
plt.title('Total Words per Emotion Class', fontsize=16)
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Total Words', fontsize=12)
plt.xticks(rotation=45)
for i, val in enumerate(word_stats['Total words']):
    plt.text(i, val + 500, f'{val:.0f}', ha='center', va='bottom')

# Unique words per class
plt.subplot(3, 1, 2)
sns.barplot(data=word_stats, x='Class', y='Unique words', palette='plasma')
plt.title('Unique Words per Emotion Class', fontsize=16)
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Unique Words', fontsize=12)
plt.xticks(rotation=45)
for i, val in enumerate(word_stats['Unique words']):
    plt.text(i, val + 200, f'{val:.0f}', ha='center', va='bottom')

# Average words per text per class
plt.subplot(3, 1, 3)
sns.barplot(data=word_stats, x='Class', y='Avg. words per text', palette='magma')
plt.title('Average Words per Text per Emotion Class', fontsize=16)
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Average Words per Text', fontsize=12)
plt.xticks(rotation=45)
for i, val in enumerate(word_stats['Avg. words per text']):
    plt.text(i, val + 0.2, f'{val:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Additional: Lexical Diversity Analysis (Unique/Total ratio)
word_stats['Lexical Diversity'] = word_stats['Unique words'] / word_stats['Total words']

plt.figure(figsize=(12, 6))
print("\n[Plot 2] Displaying Lexical Diversity by Emotion Class...")
sns.barplot(data=word_stats, x='Class', y='Lexical Diversity', palette='coolwarm')
plt.title('Lexical Diversity by Emotion Class', fontsize=16)
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Lexical Diversity (Unique/Total)', fontsize=12)
plt.xticks(rotation=45)
for i, val in enumerate(word_stats['Lexical Diversity']):
    plt.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==================== COMPREHENSIVE VISUALIZATIONS & ANALYSES ====================
# Install required packages
try:
    from wordcloud import WordCloud
except ImportError:
    !pip install wordcloud
    from wordcloud import WordCloud

# Create figure for multiple plots
plt.figure(figsize=(40, 40))
plt.suptitle('Comprehensive Bengali Sentiment Analysis', fontsize=24, y=1.02)
print("\n[Plot 3] Displaying Comprehensive Analysis (15 subplots)...")

# 1. Word Cloud Visualization
plt.subplot(5, 3, 1)
text = ' '.join(df['cleaned'])
try:
    if font_path and os.path.exists(font_path):
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              font_path=font_path).generate(text)
        print(f"Using font: {os.path.basename(font_path)}")
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        print("Using default font")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Corpus', fontsize=24)
except Exception as e:
    print(f"Error generating word cloud: {e}")
    plt.text(0.5, 0.5, "Word Cloud Error\n(Bengali font issue)",
             ha='center', va='center', fontsize=24)
    plt.axis('off')
    plt.title('Word Cloud of Corpus (Error)', fontsize=24)

# 2. Target Class Pie Chart
plt.subplot(5, 3, 2)
class_counts = df['classes'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('pastel'))
plt.title('Target Class Distribution', fontsize=24)

# 3. Corpus Distribution by Emotions
plt.subplot(5, 3, 3)
sns.countplot(data=df, x='classes', palette='viridis')
plt.title('Corpus Distribution by Emotions', fontsize=24)
plt.xlabel('Emotion Class', fontsize=24)
plt.ylabel('Count', fontsize=24)
plt.xticks(rotation=45)

# 4. Text Length Distribution
df['text_length'] = df['cleaned'].apply(lambda x: len(x.split()))
plt.subplot(5, 3, 4)
sns.histplot(df['text_length'], bins=30, kde=True, color='skyblue')
plt.title('Text Length Distribution', fontsize=24)
plt.xlabel('Number of Words', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# 5. Emotion-based Text Length Distribution (Boxplot)
plt.subplot(5, 3, 5)
sns.boxplot(data=df, x='classes', y='text_length', palette='Set3')
plt.title('Text Length by Emotion Class', fontsize=24)
plt.xlabel('Emotion Class', fontsize=24)
plt.ylabel('Text Length (Words)', fontsize=24)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 6. Lexical Diversity Analysis
df['lexical_diversity'] = df['cleaned'].apply(
    lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
plt.subplot(5, 3, 6)
sns.boxplot(data=df, x='classes', y='lexical_diversity', palette='coolwarm')
plt.title('Lexical Diversity by Emotion', fontsize=24)
plt.xlabel('Emotion Class', fontsize=24)
plt.ylabel('Lexical Diversity', fontsize=24)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 7. Average Word Length Analysis
df['avg_word_length'] = df['cleaned'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
plt.subplot(5, 3, 7)
sns.violinplot(data=df, x='classes', y='avg_word_length', palette='magma')
plt.title('Avg Word Length by Emotion', fontsize=24)
plt.xlabel('Emotion Class', fontsize=24)
plt.ylabel('Avg Word Length', fontsize=24)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 8. Sentence Complexity Analysis
df['sentence_count'] = df['cleaned'].apply(lambda x: len(x.split('।')) if '।' in x else 1)
df['avg_sentence_length'] = df['text_length'] / df['sentence_count']
plt.subplot(5, 3, 8)
sns.scatterplot(data=df, x='sentence_count', y='avg_sentence_length',
                hue='classes', palette='Set1', alpha=0.7, s=50)
plt.title('Sentence Complexity by Emotion', fontsize=24)
plt.xlabel('Number of Sentences', fontsize=24)
plt.ylabel('Avg Sentence Length (Words)', fontsize=24)
plt.legend(title='Emotion Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# 9. Word Frequency Distribution (Zipf's Law)
plt.subplot(5, 3, 9)
word_counts = Counter(' '.join(df['cleaned']).split())
word_freq = sorted(word_counts.values(), reverse=True)
plt.loglog(range(1, len(word_freq)+1), word_freq, 'o')
plt.title("Word Frequency (Zipf's Law)", fontsize=16)
plt.xlabel('Rank', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.grid(True, which="both", ls="--", alpha=0.5)

# 10. Bigram Analysis
def plot_top_ngrams(corpus, n=None, ngram_range=(2,2)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=list(bengali_stopwords)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

top_bigrams = plot_top_ngrams(df['cleaned'], n=10)
top_bigrams_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Frequency'])

plt.subplot(5, 3, 10)
ax = sns.barplot(data=top_bigrams_df, x='Frequency', y='Bigram', palette='viridis')
plt.title('Top 10 Bigrams', fontsize=24)
plt.xlabel('Frequency', fontsize=24)
plt.ylabel('Bigram', fontsize=24)

# Apply Bengali font to y-axis labels
if bengali_font_prop:
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=bengali_font_prop)

# 11. Trigram Analysis
top_trigrams = plot_top_ngrams(df['cleaned'], n=10, ngram_range=(3,3))
top_trigrams_df = pd.DataFrame(top_trigrams, columns=['Trigram', 'Frequency'])

plt.subplot(5, 3, 11)
ax = sns.barplot(data=top_trigrams_df, x='Frequency', y='Trigram', palette='plasma')
plt.title('Top 10 Trigrams', fontsize=24)
plt.xlabel('Frequency', fontsize=24)
plt.ylabel('Trigram', fontsize=24)

# Apply Bengali font to y-axis labels
if bengali_font_prop:
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=bengali_font_prop)

# 12. Correlation Matrix
plt.subplot(5, 3, 12)
corr_matrix = df[['text_length', 'lexical_diversity', 'avg_word_length', 'sentence_count']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=24)

# 13. PCA Visualization
plt.subplot(5, 3, 13)
tfidf = TfidfVectorizer(max_features=50)
X_tfidf = tfidf.fit_transform(df['cleaned']).toarray()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tfidf)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['class'] = df['classes'].values
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', palette='Set2', s=50, alpha=0.8)
plt.title('PCA Visualization', fontsize=24)
plt.xlabel('Principal Component 1', fontsize=24)
plt.ylabel('Principal Component 2', fontsize=24)
plt.legend(title='Emotion Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# 14. t-SNE Visualization
plt.subplot(5, 3, 14)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['class'] = df['classes'].values
sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='class', palette='Set1', s=50, alpha=0.8)
plt.title('t-SNE Visualization', fontsize=24)
plt.xlabel('Dimension 1', fontsize=24)
plt.ylabel('Dimension 2', fontsize=24)
plt.legend(title='Emotion Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# 15. Detailed Emotion-based Length Distribution (Violin Plot)
plt.subplot(5, 3, 15)
sns.violinplot(data=df, x='classes', y='text_length', palette='Set2', inner='quartile')
plt.title('Detailed Text Length by Emotion', fontsize=24)
plt.xlabel('Emotion Class', fontsize=24)
plt.ylabel('Text Length (Words)', fontsize=24)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ==================== EMOTION-SPECIFIC WORD CLOUDS ====================
# Create a separate figure for emotion-specific word clouds
emotion_classes = df['classes'].unique()
num_emotions = len(emotion_classes)
cols = 3
rows = (num_emotions + cols - 1) // cols  # Calculate rows needed

plt.figure(figsize=(24, 5 * rows))
plt.suptitle('Emotion-Specific Word Clouds', fontsize=20, y=1.02)
print("\n[Plot 4] Displaying Emotion-Specific Word Clouds...")

for i, emotion in enumerate(emotion_classes):
    plt.subplot(rows, cols, i+1)
    text = ' '.join(df[df['classes'] == emotion]['cleaned'])
    try:
        if font_path and os.path.exists(font_path):
            wordcloud = WordCloud(width=400, height=200, background_color='white',
                                 font_path=font_path).generate(text)
            print(f"Using font: {os.path.basename(font_path)} for {emotion}")
        else:
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
            print(f"Using default font for {emotion}")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud: {emotion}', fontsize=24)
    except Exception as e:
        print(f"Error generating word cloud for {emotion}: {e}")
        plt.text(0.5, 0.5, f"Word Cloud Error\nfor {emotion}\n(Bengali font issue)",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.title(f'Word Cloud: {emotion} (Error)', fontsize=24)

plt.tight_layout()
plt.show()

# ==================== STATISTICAL ANALYSES ====================
# 1. Descriptive Statistics Report
stats_df = df.groupby('classes').agg({
    'text_length': ['mean', 'median', 'std', 'min', 'max'],
    'lexical_diversity': ['mean', 'median', 'std'],
    'avg_word_length': ['mean', 'median', 'std'],
    'sentence_count': ['mean', 'median', 'std']
}).round(2)

print("\n=== DESCRIPTIVE STATISTICS BY EMOTION CLASS ===")
print(stats_df)

# 2. Statistical Significance Testing
# ANOVA test for text length differences between emotions
text_lengths = [df[df['classes'] == emotion]['text_length'].values for emotion in df['classes'].unique()]
f_stat, p_value = stats.f_oneway(*text_lengths)
print(f"\nANOVA test for text length differences: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")

# Chi-square test for word-emotion association
cv = CountVectorizer(max_features=50)
X_cv = cv.fit_transform(df['cleaned'])
chi2_stat, p_values, dof, expected = stats.chi2_contingency(
    pd.crosstab(df['classes'], np.argmax(X_cv.toarray(), axis=1)))
print(f"Chi-square test for word-emotion association: χ² = {chi2_stat:.2f}, p-value = {p_values:.4f}")

# 3. TF-IDF Feature Importance
plt.figure(figsize=(14, 10))
print("\n[Plot 5] Displaying TF-IDF Feature Importance...")
tfidf = TfidfVectorizer(max_features=20)
X_tfidf = tfidf.fit_transform(df['cleaned'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df['class'] = df['classes'].values
class_tfidf = tfidf_df.groupby('class').mean()

# Create the heatmap with Bengali font support
sns.heatmap(class_tfidf.T, cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('TF-IDF Feature Importance by Emotion Class', fontsize=16)
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Terms', fontsize=12)

# Apply Bengali font to the y-axis labels (terms)
if bengali_font_prop:
    plt.yticks(fontproperties=bengali_font_prop)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
