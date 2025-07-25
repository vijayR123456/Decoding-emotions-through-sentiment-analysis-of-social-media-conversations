# -*- coding: utf-8 -*-
"""2.Decoding emotions through sentiment analysis of social media conversations.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s3LHuEqFI1fJQrJWQPzAcHTP6XgdVO6g
"""

pip install twython

pip install vaderSentiment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')

"""Load Data"""

df = pd.read_csv("/content/sentimentdataset.csv")  # Remove the extra dot (.) at the end

df.head()

def null_count():
    return pd.DataFrame({'features': df.columns,
                'dtypes': df.dtypes.values,
                'NaN count': df.isnull().sum().values,
                'NaN percentage': df.isnull().sum().values/df.shape[0]}).style.background_gradient(cmap='Set3',low=0.1,high=0.01)
null_count()

df.duplicated().sum()

df.columns

for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

df['Platform'].value_counts()

df['Platform'] = df['Platform'].str.strip()

df['Country'].value_counts()

df['Country'] = df['Country'].str.strip()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Day_of_Week'] = df['Timestamp'].dt.day_name()

month_mapping = {
    1: 'Januari',
    2: 'Februari',
    3: 'Maret',
    4: 'April',
    5: 'Mei',
    6: 'Juni',
    7: 'Juli',
    8: 'Agustus',
    9: 'September',
    10: 'Oktober',
    11: 'November',
    12: 'Desember'
}

df['Month'] = df['Month'].map(month_mapping)

df['Month'] = df['Month'].astype('object')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('.∗?.*?', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = " ".join(text.split())

df["Clean_Text"] = df["Text"].apply(clean)

!pip install colorama
from colorama import Fore

specified_columns = ['Platform','Country', 'Year','Month','Day_of_Week']

for col in specified_columns:
    total_unique_values = df[col].nunique()
    print(f'Total unique values for {col}: {total_unique_values}')

    top_values = df[col].value_counts()

    # Define colors using Fore from colorama
    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX]

    for i, (value, count) in enumerate(top_values.items()):
        color = colors[i % len(colors)]
        print(f'{color}{value}: {count}{Fore.RESET}')

    print('\n' + '=' * 30 + '\n')

"""Exploratory Data Analysis (EDA)"""

df1 = df.copy()

"""Sentiment Analysis"""

# 1. Import the necessary tools
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# 2. Download the lexicon (only needed once)
nltk.download('vader_lexicon')

# 3. Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 4. Sample data
data = {
    'Clean_Text': [
        "I love the service!",
        "The doctor was rude.",
        "It was okay, not great.",
        "Absolutely fantastic experience.",
        "Terrible. Will not return."
    ]
}

df = pd.DataFrame(data)

# 5. Get sentiment scores
df['Vader_Score'] = df['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

# 6. Classify the sentiment
df['Sentiment'] = df['Vader_Score'].apply(
    lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
)

# 7. Show the result
print(df)

import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame
data = {'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']}
df1 = pd.DataFrame(data)

# Pie chart parameters
colors = ['#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0)  # Adjust length based on unique Sentiments

# Grouping data
sentiment_counts = df1['Sentiment'].value_counts()

# Plot
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    x=sentiment_counts.values,  # Pass values
    labels=sentiment_counts.index,  # Pass labels
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})',
    wedgeprops=dict(width=0.7),
    textprops=dict(size=10, color="r"),
    pctdistance=0.7,
    colors=colors[:len(sentiment_counts)],  # Ensure matching colors
    explode=explode[:len(sentiment_counts)],  # Ensure matching explode
    shadow=True
)

# Add center circle
center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
fig.gca().add_artist(center_circle)

# Add central text
ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

# Add legend
ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax.axis('equal')  # Equal aspect ratio
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example DataFrame (replace with your actual data)
data = {
    'Year': [2020, 2020, 2021, 2021, 2022],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
}
df1 = pd.DataFrame(data)

# Ensure Year and Sentiment columns are correctly formatted
df1['Year'] = df1['Year'].astype(str)  # Convert Year to string
df1['Sentiment'] = df1['Sentiment'].astype(str)  # Convert Sentiment to string

# Plotting
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Years and Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example DataFrame (replace with your actual data)
data = {
    'Date': ['2023-01-15', '2023-02-20', '2023-03-05', '2023-01-10', '2023-03-25'],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
}
df1 = pd.DataFrame(data)

# Ensure the Month column exists
df1['Month'] = pd.to_datetime(df1['Date']).dt.month  # Extract month number
df1['Month'] = df1['Month'].map({
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
})  # Map to month names

# Plotting
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Sentiment', data=df1, palette='Paired', order=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
plt.title('Relationship between Month and Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example DataFrame (replace with your actual data)
data = {
    'Platform': ['Twitter', 'Facebook', 'Instagram', 'Twitter', 'Facebook'],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
}
df1 = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 6))
sns.countplot(
    x='Platform',
    hue='Sentiment',
    data=df1,
    palette='Paired',
    order=df1['Platform'].unique()  # Adjust if you want a specific order
)
plt.title('Relationship between Platform and Sentiment')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example DataFrame (replace with your actual data)
data = {
    'Country': ['USA', 'India', 'UK', 'USA', 'India', 'France', 'Canada', 'Germany', 'Italy', 'Japan'],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive']
}
df1 = pd.DataFrame(data)

# Extract the top 10 countries
top_10_countries = df1['Country'].value_counts().head(10).index

# Filter data for only the top 10 countries
df_top_10_countries = df1[df1['Country'].isin(top_10_countries)]

# Plotting
plt.figure(figsize=(12, 6))
sns.countplot(
    x='Country',
    hue='Sentiment',
    data=df_top_10_countries,
    palette='Paired',
    order=top_10_countries  # Ensure correct order
)
plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
from collections import Counter

# Example DataFrame (replace with your actual data)
data = {
    'Clean_Text': [
        'This is a sample text for testing',
        'Testing word frequency with sample data',
        'Data analysis and text mining are fun',
        'Sample text data for word count testing'
    ]
}
df1 = pd.DataFrame(data)

# Ensure no missing values in Clean_Text
df1['Clean_Text'] = df1['Clean_Text'].fillna('')

# Split text into words and count word frequency
df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])

# Convert to DataFrame
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

# Display with gradient (for Jupyter Notebook)
display(top_words_df.style.background_gradient(cmap='Blues'))

# For non-notebook environments, simply print
print(top_words_df)

df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

import plotly.express as px # Make sure to import plotly.express

fig = px.bar(
  top_words_df,
  x='count',
  y='Common_words',
  title='Common Words in Text Data',
  orientation='h',
  width=700,
  height=700,
  color='Common_words'
)

fig.show()

import pandas as pd

# Example DataFrame (replace with your actual data)
data = {
    'Text': ['I love this!', 'This is bad.', 'It is okay.', 'Absolutely fantastic!', 'Not good at all.'],
    'Sentiment': ['positive', 'negative', 'neutral', 'Positive', 'Negative']
}
df1 = pd.DataFrame(data)

# Ensure consistency in the Sentiment column
df1['Sentiment'] = df1['Sentiment'].str.lower()

# Filter data based on Sentiment
Positive_sent = df1[df1['Sentiment'] == 'positive']
Negative_sent = df1[df1['Sentiment'] == 'negative']
Neutral_sent = df1[df1['Sentiment'] == 'neutral']

# Validation
print("Positive Sentiment:")
print(Positive_sent)

print("\nNegative Sentiment:")
print(Negative_sent)

print("\nNeutral Sentiment:")
print(Neutral_sent)

"""Positive Common Words"""

import pandas as pd
from collections import Counter

# Example DataFrame (replace with your actual data)
data = {
    'Clean_Text': [
        'I love this product', 'Absolutely amazing experience',
        'Great service and support', 'Wonderful and fantastic!',
        'I would recommend this to everyone'
    ],
    'Sentiment': ['positive', 'positive', 'positive', 'positive', 'positive']
}
df1 = pd.DataFrame(data)

# Ensure consistency in Sentiment column
df1['Sentiment'] = df1['Sentiment'].str.lower()

# Create temp_list column if not already present
df1['temp_list'] = df1['Clean_Text'].fillna('').apply(lambda x: str(x).split())

# Count top 10 words for positive sentiment
top = Counter([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])

# Display with gradient (for Jupyter Notebook)
display(temp_positive.style.background_gradient(cmap='Greens'))

# For non-notebook environments, print the DataFrame
print(temp_positive)

import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Example DataFrame (replace with your actual data)
data = {
    'Clean_Text': [
        'I love this product', 'Absolutely amazing experience',
        'Great service and support', 'Wonderful and fantastic!',
        'I would recommend this to everyone', 'This is bad.', 'It is okay.'
    ],
    'Sentiment': ['positive', 'positive', 'positive', 'positive', 'positive', 'negative', 'neutral']
}
df1 = pd.DataFrame(data)

# Ensure consistency in Sentiment column
df1['Sentiment'] = df1['Sentiment'].str.lower()

# Create temp_list column if not already present
df1['temp_list'] = df1['Clean_Text'].fillna('').apply(lambda x: str(x).split())

# Filter for positive sentiments and join words
positive_words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import pandas as pd
from collections import Counter

# Example DataFrame (replace with your actual data)
data = {
    'Clean_Text': [
        'I feel neutral about this', 'Not good, not bad', 'Just an average experience',
        'Nothing special', 'This is okay, neither good nor bad'
    ],
    'Sentiment': ['neutral', 'neutral', 'neutral', 'neutral', 'neutral']
}
df1 = pd.DataFrame(data)

# Ensure consistency in Sentiment column
df1['Sentiment'] = df1['Sentiment'].str.lower()

# Create temp_list column if not already present
df1['temp_list'] = df1['Clean_Text'].fillna('').apply(lambda x: str(x).split())

# Count top 10 words for neutral sentiment
top = Counter([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])

# Display with gradient (for Jupyter Notebook)
display(temp_positive.style.background_gradient(cmap='Blues'))

# For non-notebook environments, print the DataFrame
print(temp_positive)

words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import pandas as pd
from collections import Counter

# Example DataFrame (replace with your actual data)
data = {
    'Clean_Text': [
        'I hate this product', 'Worst experience ever', 'I would not recommend this',
        'It was a terrible decision', 'This is awful'
    ],
    'Sentiment': ['negative', 'negative', 'negative', 'negative', 'negative']
}
df1 = pd.DataFrame(data)

# Ensure consistency in Sentiment column (e.g., lowercase)
df1['Sentiment'] = df1['Sentiment'].str.lower()

# Create temp_list column if not already present
df1['temp_list'] = df1['Clean_Text'].fillna('').apply(lambda x: str(x).split())

# Count top 10 words for negative sentiment
top = Counter([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])

# Display with gradient (for Jupyter Notebook)
display(temp_positive.style.background_gradient(cmap='Reds'))

# For non-notebook environments, print the DataFrame
print(temp_positive)

words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

"""Data Preparation"""

df2 = df1.copy()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

"""Split Data"""

X = df2['Clean_Text'].values
y = df2['Sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Modeling"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample text data (ensure it's properly labeled)
X = ['I love machine learning', 'Python is great', 'Text classification is fun', 'I enjoy learning new things']
y = [1, 0, 1, 0]  # Example binary sentiment labels (1 = positive, 0 = negative)

# Train-test split (with a very small dataset, you may want to ensure each class is represented in both train and test sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the vectorizer with max_features=5000
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data (using the same vocabulary as X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Example classifier (Naive Bayes)
clf = MultinomialNB()

# Fit the model to the training data
clf.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

"""Passive Aggressive Classifier"""

pac_classifier = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac_classifier.fit(X_train_tfidf, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Sample text data (replace with your actual data)
X = ['I love machine learning', 'Python is great', 'Text classification is fun', 'I enjoy learning new things']
y = [1, 0, 1, 0]  # Example binary sentiment labels (1 = positive, 0 = negative)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the vectorizer with max_features=5000
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data (using the same vocabulary as X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the RandomForestClassifier
random_forest_classifier = RandomForestClassifier()

# Train the model
random_forest_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_rf = random_forest_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Generate the classification report
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Print results
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print(f"Classification Report:\n{classification_rep_rf}")

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:\n", classification_rep_rf)

"""SVM Classifier"""

svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Generate the classification report
classification_rep_svm = classification_report(y_test, y_pred_svm)

print("Support Vector Machine Results:")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:\n", classification_rep_svm)

"""Multinomial NB"""

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Generate the classification report
classification_rep_nb = classification_report(y_test, y_pred_nb)

print("\nMultinomial Naive Bayes Results:")
print(f"Accuracy: {accuracy_nb}")
print("Classification Report:\n", classification_rep_nb)

"""Best Modeling : Passive Aggressive Classifier"""

param_dist = {
    'C': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'verbose': [0, 1],
}

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier

# Load dataset
data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'], remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# Text feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize classifier
pac_classifier = PassiveAggressiveClassifier(random_state=42)

# Define parameter distribution
param_dist = {
    'C': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'verbose': [0, 1],
}

# Randomized search
randomized_search = RandomizedSearchCV(
    pac_classifier,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)

# Fit model
randomized_search.fit(X_train_tfidf, y_train)

# Best parameters
print("Best Parameters:", randomized_search.best_params_)

# Best score
print("Best CV Score:", randomized_search.best_score_)

# Get the best parameters from the RandomizedSearchCV
best_params_randomized = randomized_search.best_params_

# Initialize the PassiveAggressiveClassifier with the best parameters
best_pac_classifier_randomized = PassiveAggressiveClassifier(random_state=42, **best_params_randomized)

# Fit the classifier with the training data
best_pac_classifier_randomized.fit(X_train_tfidf, y_train)

# Optionally, you can evaluate the classifier on the test set
y_pred_rf = best_pac_classifier_randomized.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

print(f"Accuracy: {accuracy_rf}")
print(f"Classification Report: \n{classification_rep_rf}")

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'], remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# Text feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize classifier
pac_classifier = PassiveAggressiveClassifier(random_state=42)

# Define parameter distribution
param_dist = {
    'C': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'verbose': [0, 1],
}

# Randomized search
randomized_search = RandomizedSearchCV(
    pac_classifier,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)

# Fit model
randomized_search.fit(X_train_tfidf, y_train)

# Best parameters
print("Best Parameters:", randomized_search.best_params_)

# Best score
print("Best CV Score:", randomized_search.best_score_)


# Get the best parameters from the RandomizedSearchCV
best_params_randomized = randomized_search.best_params_

# Initialize the PassiveAggressiveClassifier with the best parameters
best_pac_classifier_randomized = PassiveAggressiveClassifier(random_state=42, **best_params_randomized)

# Fit the classifier with the training data
best_pac_classifier_randomized.fit(X_train_tfidf, y_train)

# Optionally, you can evaluate the classifier on the test set
y_pred_rf = best_pac_classifier_randomized.predict(X_test_tfidf)
accuracy_best_pac_randomized = accuracy_score(y_test, y_pred_rf) # assign to accuracy_best_pac_randomized
classification_rep_best_pac_randomized = classification_report(y_test, y_pred_rf) # assign to classification_rep_best_pac_randomized

print(f"Accuracy: {accuracy_best_pac_randomized}") # use accuracy_best_pac_randomized
print(f"Classification Report: \n{classification_rep_best_pac_randomized}") # use classification_rep_best_pac_randomized

print("Best PassiveAggressiveClassifier Model (RandomizedSearchCV):")
print(f"Best Hyperparameters: {best_params_randomized}")
print(f"Accuracy: {accuracy_best_pac_randomized}") # use accuracy_best_pac_randomized
print("Classification Report:\n", classification_rep_best_pac_randomized) # use classification_rep_best_pac_randomized

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'], remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# Text feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize classifier
pac_classifier = PassiveAggressiveClassifier(random_state=42)

# Define parameter distribution
param_dist = {
    'C': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'verbose': [0, 1],
}

# Randomized search
randomized_search = RandomizedSearchCV(
    pac_classifier,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)

# Fit model
randomized_search.fit(X_train_tfidf, y_train)

# Best parameters
print("Best Parameters:", randomized_search.best_params_)

# Best score
print("Best CV Score:", randomized_search.best_score_)


# Get the best parameters from the RandomizedSearchCV
best_params_randomized = randomized_search.best_params_

# Initialize the PassiveAggressiveClassifier with the best parameters
best_pac_classifier_randomized = PassiveAggressiveClassifier(random_state=42, **best_params_randomized)

# Fit the classifier with the training data
best_pac_classifier_randomized.fit(X_train_tfidf, y_train)

# Optionally, you can evaluate the classifier on the test set
y_pred_best_pac_randomized = best_pac_classifier_randomized.predict(X_test_tfidf) # Assign the predictions to y_pred_best_pac_randomized
accuracy_best_pac_randomized = accuracy_score(y_test, y_pred_best_pac_randomized)
classification_rep_best_pac_randomized = classification_report(y_test, y_pred_best_pac_randomized)

print(f"Accuracy: {accuracy_best_pac_randomized}")
print(f"Classification Report: \n{classification_rep_best_pac_randomized}")

print("Best PassiveAggressiveClassifier Model (RandomizedSearchCV):")
print(f"Best Hyperparameters: {best_params_randomized}")
print(f"Accuracy: {accuracy_best_pac_randomized}")
print("Classification Report:\n", classification_rep_best_pac_randomized)

# Generate the confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_pred_best_pac_randomized)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Greys', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix - Hyperparameters')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()