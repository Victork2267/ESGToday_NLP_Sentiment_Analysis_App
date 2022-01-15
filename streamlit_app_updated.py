## Datetime
import re
from datetime import datetime, timedelta
from time import strptime
import pandas as pd
import sqlite3

## Data Cleaning
import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer

## NLTK
from nltk.sentiment import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob

## Visualisation
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

## Streamlit
import streamlit as st

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('vader_lexicon')

# Create connection to db_file (Not required now)
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    
    return conn

## Remove stopwords, tags and lemmtize information
def text_cleaning(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(text.split('.'))
    text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', text.strip().lower())).strip()
    text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
    text = [word for word in text.split()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

## Tag news sentiment scores based on sentiment intensity analyser
def tag_sentiment_SIA(df):
    sia = SIA()
    
    #Empty lists to store positive, neutral and negative scores
    negative_scores = []
    neutral_scores = []
    positive_scores = []
    compound_scores = []
    final_tag = []
    
    texts = df["Content_Cleaned"].tolist()
    
    #loop through all news text
    for text in texts:
        score_dictionary = sia.polarity_scores(text)
        negative_scores.append(score_dictionary["neg"])
        positive_scores.append(score_dictionary["pos"])
        neutral_scores.append(score_dictionary["neu"])
        compound_scores.append(score_dictionary["compound"])
        
        if score_dictionary["compound"]>0:
            final_tag.append("positive")
        elif score_dictionary["compound"]<0:
            final_tag.append("negative")
        else:
            final_tag.append("neutral")
            
    df["Negative_Score"] = negative_scores
    df["Positive_Score"] = positive_scores
    df["Neutral_Score"] = neutral_scores
    df["Compound_Score"] = compound_scores
    df["Final_Score"] = final_tag
    
    print("done")

## Tag news sentiment scores based on information from textblob    
def Tag_Sentiment_texblob(df):
    #Empty lists to store positive, neutral and negative scores
    textblob_score = []
    textblob_tag = []
    
    texts = df["Content_Cleaned"].tolist()
    
    for text in texts:
        doc_current = TextBlob(text)
        score = doc_current.polarity
        textblob_score.append(score)
        if score > 0:
            textblob_tag.append('positive')
        elif score<0:
            textblob_tag.append('negative')
        else:
            textblob_tag.append('neutral')
    df['textblob_score'] = textblob_score
    df['textblob_sentiment_tag'] = textblob_tag
    
    print("done")

## Generate Score boxplot in Streamlit plotly
def generatebox(df):
    labels = ["Negative Score","Neutral Score","Positive Score","Compound Score"]
    y_textblob = df["textblob_score"].values
    y_ns = df["Negative Score"].values
    y_nts = df["Neutral_Score"].values
    y_ps = df["Positive_Score"].values
    y_comp = df["Compound_Score"].values

    fig = make_subplots(rows=1, cols=2, subplot_titles=("TextBlob", "SIA"))

    fig.add_trace(
        go.Box(y=y_ns),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=y_nts),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=y_ps),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=y_comp),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=y_textblob),
        row=1, col=2
    )

    return fig

## Generate pie chart from streamlit plotly
def generate_pie(df):
    labels_textblob = df["textblob_sentiment_tag"].value_counts().keys()
    values_textblob = df["textblob_sentiment_tag"].value_counts()

    labels_finalscore = df["Final_Score"].value_counts().keys()
    values_finalscore = df["Final_Score"].value_counts()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("TextBlob", "SIA"), specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    fig.add_trace(
        go.Pie(labels=labels_textblob, values=values_textblob),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=labels_finalscore, values=values_finalscore),
        row=1, col=2
    )
    
    fig.update_layout(height=600,width=800)
    return fig

if __name__ == "__main__":
    st.title("ESG Today Scrapper & Sentiment Analyser")
    st.text("Sustainable Investments - Analyse 500 companies based on news from ESG Today")

    ## Open News file
    con = create_connection(db_file=r"ESG_db_new.db")
    df = pd.read_sql_query("SELECT * from  ESG_data", con)

    ## Get Dropdown options
    list_500 = df["Company"].value_counts().keys().tolist()
    option1 = st.selectbox("Select the first fortune 500 company to examine",list_500)
    option2 = st.selectbox("Select the second fortune 500 company to examine",list_500)
    button_option = st.button("Analyse")
    if button_option:
        ## Get information from dataframe and create dataframe in streamlit
        df_option1 = df[df["Company"] == option1]
        df_option2 = df[df["Company"] == option2]
        st.text("There are {} relvant ESG-Related News Content for {}".format(len(df_option1),option1))
        st.text("There are {} relevant ESG-Related News Content for {}".format(len(df_option2), option2))

        ## st.table(df.head())
        st.info("Analysing News Content using Vader")

        ## Clean the data
        df_option1["Content_Cleaned"] = df_option1["Content"].apply(lambda x: ' '.join(text_cleaning(x)))
        df_option2["Content_Cleaned"] = df_option2["Content"].apply(lambda x: ' '.join(text_cleaning(x)))
        
        ## Get SIA tags
        tag_sentiment_SIA(df=df_option1)
        tag_sentiment_SIA(df=df_option2)
        
        ## Generate Pie/Boxplot of the company's ESG data
        st.subheader("Propotion of Positive vs Negative ESG-Related News Sentiments for {}".format(option1))
        labels_finalscore_option1 = df_option1["Final_Score"].value_counts().keys()
        values_finalscore_option1 = df_option1["Final_Score"].value_counts()

        labels_finalscore_option2 = df_option2["Final_Score"].value_counts().keys()
        values_finalscore_option2 = df_option2["Final_Score"].value_counts()

        fig = make_subplots(rows=1, cols=2, subplot_titles=(option1, option2), specs=[[{"type": "pie"}, {"type": "pie"}]])
        
        ## Generate Pie Chart
        fig.add_trace(
            go.Pie(labels=labels_finalscore_option1, values=values_finalscore_option1),
            row=1,col=1
        )

        fig.add_trace(
            go.Pie(labels=labels_finalscore_option2, values=values_finalscore_option2),
            row=1, col=2
        )

        fig.update_layout(height=600,width=800)
        st.plotly_chart(fig, use_container_width=True)

        ## Generate Boxplot of the company's ESG data
        st.subheader("Statistics on Vader's Sentiment Compound Scoring")

        ## Generate Boxplot
        fig = make_subplots(rows=1, cols=2, subplot_titles=(option1, option2))
        fig.add_trace(
            go.Box(y=df_option1["Compound_Score"].values),
            row=1,col=1
        )

        fig.add_trace(
            go.Box(y=df_option2["Compound_Score"].values),
            row=1, col=2
        )

        fig.update_layout(height=600,width=800)
        st.plotly_chart(fig, use_container_width=True)

        ## Generate WordClouds
        # Get negative and positive news text from pandas column
        neg_df = df_option1["Content_Cleaned"][df_option1["Final_Score"] == "negative"]
        pos_df = df_option1["Content_Cleaned"][df_option1["Final_Score"] == "positive"]
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Check if there is neg/pos content and generate wordclouds
        if len(neg_df) > 0: 
            wordcloud_neg = WordCloud().generate("".join(neg_df))

            # Display the generated image:
            st.subheader("Common Words Associated with Negative ESG-Related News Sentiments for {}".format(option1))
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()
        if len(pos_df) > 0:
            wordcloud_pos = WordCloud().generate("".join(pos_df))

            # Display the generated image:
            st.subheader("Common Words Associated with Positive ESG-Related News Sentiments for {}".format(option1))
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()
