import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import smtplib


class Article:
    def __init__(self, headline, author, date, content, sentiment=None, topic=None, cluster=None):
        self.headline = headline
        self.author = author
        self.date = date
        self.content = content
        self.sentiment = sentiment
        self.topic = topic
        self.cluster = cluster


class NewsAggregator:

    def __init__(self):
        self.articles = []

    def scrape_articles(self, sources):
        for source in sources:
            response = requests.get(source)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_elements = soup.find_all('article')
            for article_element in article_elements:
                headline = article_element.find('h2').text.strip()
                author = article_element.find('span', class_='author').text.strip()
                date = article_element.find('time')['datetime']
                content = article_element.find('div', class_='entry-content').text.strip()
                article = Article(headline, author, date, content)
                self.articles.append(article)

    def analyze_articles(self):
        sid = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        cleaned_articles = []
        for article in self.articles:
            words = word_tokenize(article.content)
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
            cleaned_articles.append(' '.join(words))
            article.sentiment = sid.polarity_scores(article.content)['compound']

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(cleaned_articles)
        lda = LatentDirichletAllocation(n_components=10)
        document_topics = lda.fit_transform(X)
        for i, article in enumerate(self.articles):
            article.topic = np.argmax(document_topics[i])
            
        self.clustering()

    def clustering(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([article.content for article in self.articles])
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(X)
        for i, article in enumerate(self.articles):
            article.cluster = kmeans.labels_[i]

    def trending_topics(self):
        topics = [article.topic for article in self.articles]
        unique_topics, counts = np.unique(topics, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        trending_topics = [{'topic': unique_topics[i], 'count': counts[i]} for i in sorted_indices]
        return trending_topics

    def recommend_courses(self):
        skills = ['python', 'data analysis', 'machine learning', 'web development']
        courses = ['Python for Data Analysis', 'Machine Learning 101', 'Web Development Fundamentals']
        recommended_courses = []
        for skill in skills:
            if any(article.topic == skill for article in self.articles):
                recommended_courses.append({
                    'skill': skill,
                    'courses': courses
                })
        return recommended_courses

    def generate_report(self):
        df = pd.DataFrame.from_records([article.__dict__ for article in self.articles])
        df.to_csv('news_articles.csv', index=False)

        sentiment_chart = df['sentiment'].value_counts().plot(kind='bar')
        plt.title('Sentiment Analysis')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig('sentiment_analysis.png')
        plt.close()

        cluster_chart = df['cluster'].value_counts().plot(kind='bar')
        plt.title('Article Clustering')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.savefig('article_clustering.png')
        plt.close()

        word_cloud = WordCloud().generate(' '.join(df['content']))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('word_cloud.png')
        plt.close()

    def find_influencers(self):
        contact_info = []
        for article in self.articles:
            # Extract contact information from article's content
            # ...
            # For example, extract name, email, and phone from article content
            name = 'John Doe'
            email = 'johndoe@example.com'
            phone = '123-456-7890'
            contact_info.append({
                'name': name,
                'email': email,
                'phone': phone
            })
        return contact_info

    def send_connection_requests(self):
        smtp_server = 'smtp.example.com'
        smtp_port = 587
        sender_email = 'your_email@example.com'
        sender_password = 'your_password'
        message = '''
        Subject: New Connection Request

        Hi {name},

        I came across your article on {topic} and found it very insightful. I would like to connect with you on LinkedIn.

        Best,
        Your Name
        '''
        for influencer in self.find_influencers():
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, influencer['email'], message.format(name=influencer['name'], topic='Topic'))

    def recommend_events(self):
        events = [{'name': 'Conference A', 'location': 'New York', 'industry': 'Technology'},
                  {'name': 'Conference B', 'location': 'San Francisco', 'industry': 'Finance'},
                  {'name': 'Webinar A', 'location': 'Online', 'industry': 'Marketing'}]
        return events

    def send_email_notifications(self, user_email):
        smtp_server = 'smtp.example.com'
        smtp_port = 587
        sender_email = 'your_email@example.com'
        sender_password = 'your_password'
        message = '''
        Subject: New Articles of Interest

        Hi,

        Here are some new articles that might be of interest to you:

        {article1_headline} - {article1_source}
        {article2_headline} - {article2_source}
        {article3_headline} - {article3_source}

        Best,
        News Aggregator
        '''
        articles_of_interest = [article for article in self.articles if article.sentiment > 0.5][:3]
        if len(articles_of_interest) > 0:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, user_email, message.format(
                    article1_headline=articles_of_interest[0].headline,
                    article1_source=articles_of_interest[0].source,
                    article2_headline=articles_of_interest[1].headline,
                    article2_source=articles_of_interest[1].source,
                    article3_headline=articles_of_interest[2].headline,
                    article3_source=articles_of_interest[2].source
                ))


if __name__ == '__main__':
    aggregator = NewsAggregator()
    sources = ['https://news_source1.com', 'https://news_source2.com']  # Replace with actual news sources
    aggregator.scrape_articles(sources)
    aggregator.analyze_articles()
    trending_topics = aggregator.trending_topics()
    recommended_courses = aggregator.recommend_courses()
    aggregator.generate_report()
    influencers = aggregator.find_influencers()
    aggregator.send_connection_requests()
    recommended_events = aggregator.recommend_events()
    user_email = 'user@example.com'  # Replace with actual user email address
    aggregator.send_email_notifications(user_email)