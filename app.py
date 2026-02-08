from transformers import pipeline
import requests
import streamlit as st
from datetime import datetime, timedelta
import os

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1, framework="pt")  # Fixed typo

with st.spinner('Loading model...'):
    pipe = load_model()

st.title("📰 News Sentiment Analyzer")
st.write("Analyse the sentiment of recent news articles")

keyword = st.text_input('Enter keyword', placeholder='e.g., Tesla, Bitcoin, Apple')
days_back = st.slider('Select days to lookback', 2, 10, 5)

if st.button('🔍 Analyze', type='primary'):
    if not keyword:
        st.warning("Please enter a keyword!")
        st.stop()

    date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    with st.spinner('Fetching articles...'):
        try:
            api = st.secrets['NEWSAPI_KEY']
            
            url = (
                'https://newsapi.org/v2/everything?'
                f'q={keyword}&'          
                f'from={date}&'        
                f'sortBy=popularity&'
                f'pageSize=100&'
                f'apiKey={api}'
            )
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
          
            if data.get('status') != 'ok':
                st.error(f"API Error: {data.get('message', 'Unknown error')}")
                st.stop()
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch articles: {e}")
            st.stop()
        except KeyError:
            st.error("NEWSAPI_KEY not found in secrets!")
            st.stop()

        articles = data.get('articles', [])

        articles = [
            article for article in articles 
            if keyword.lower() in article.get('title', '').lower() 
            or keyword.lower() in article.get('description', '').lower()
        ]
        
        if not articles:
            st.warning(f"No articles found for '{keyword}' in the last {days_back} days.")
            st.stop()

        st.info(f'Found {len(articles)} articles. Analyzing...')

        total_score = 0
        num_articles = 0
        sentiments_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, article in enumerate(articles):
            text = f"{article.get('title', '')} {article.get('description', '')}"
    
            if len(text.strip()) < 10:
                continue
    
            text = text[:512]
    
            try:
                sentiment = pipe(text)[0]
                
                sentiments_data.append({
                    'title': article.get('title', 'No title'),
                    'label': sentiment['label'],
                    'score': sentiment['score']
                })
                
                if sentiment['label'] == 'positive':
                    total_score += sentiment['score']
                    num_articles += 1
                elif sentiment['label'] == 'negative':
                    total_score -= sentiment['score']
                    num_articles += 1
                elif sentiment['label'] == 'neutral':
                    num_articles += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Error processing article {i+1}: {str(e)[:100]}")
            
           
            progress_bar.progress((i + 1) / len(articles))
            status_text.text(f"Processing {i+1}/{len(articles)}")
        
       
        progress_bar.empty()
        status_text.empty()

        
        if num_articles > 0:
            final_score = total_score / num_articles
            
            
            if final_score >= 0.15:
                overall = "Positive"
                color = "green"
            elif final_score <= -0.15:
                overall = "Negative"
                color = "red"
            else:
                overall = "Neutral"
                color = "orange"
            
            st.success('Analysis Complete!')
            
            col1, col2, col3 = st.columns(3)
            positive_count = sum(1 for s in sentiments_data if s['label'] == 'positive')
            negative_count = sum(1 for s in sentiments_data if s['label'] == 'negative')
            neutral_count = sum(1 for s in sentiments_data if s['label'] == 'neutral')
            
            col1.metric("Positive", positive_count)
            col2.metric("Neutral", neutral_count)
            col3.metric("Negative", negative_count)
            
            st.markdown(f"### Overall Sentiment: :{color}[{overall}]")
            st.metric("Average Score", f"{final_score:.4f}")
            
            with st.expander("View analyzed articles"):
                for item in sentiments_data:
                    st.write(f"**{item['title'][:80]}{'...' if len(item['title']) > 80 else ''}**")
                    st.caption(f"Sentiment: {item['label'].capitalize()} (confidence: {item['score']:.2%})")
                    st.divider()
        else:
            st.warning("No articles were analyzed!")

st.markdown("---")