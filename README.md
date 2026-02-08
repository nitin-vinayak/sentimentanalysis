# News Sentiment Analyzer

A Streamlit web application that analyzes the sentiment of recent news articles using machine learning. Get insights into public sentiment around any topic by analyzing news coverage from the past week.

Link: https://sentiment-analysis-kh6p3nqm3njavtmxjhxjbt.streamlit.app/

## Features

- **Real-time News Analysis**: Fetch recent articles using NewsAPI
- **AI-Powered Sentiment Detection**: Uses DistilBERT transformer model for accurate sentiment classification
- **Customizable Timeframe**: Analyze news from 2-10 days back
- **Visual Metrics**: Clear breakdown of positive and negative articles
- **Detailed Results**: View individual article sentiments with confidence scores
- **Fast Processing**: Efficient caching and batch processing

## How It Works

1. **Article Fetching**: Queries NewsAPI for recent articles matching your keyword
2. **Text Preprocessing**: Combines article titles and descriptions
3. **Sentiment Analysis**: Uses DistilBERT model to classify sentiment
4. **Score Calculation**: Aggregates individual sentiments into overall score
5. **Visualization**: Presents results in an intuitive dashboard

## Sentiment Classification

- **Positive**: Final score ≥ 0.15
- **Neutral**: Final score between -0.15 and 0.15
- **Negative**: Final score ≤ -0.15

## Technical Stack

- **Frontend**: Streamlit
- **ML Model**: DistilBERT (sentiment analysis)
- **News API**: NewsAPI.org
- **NLP Framework**: Hugging Face Transformers

## Prerequisites

- Python 3.8 or higher
- NewsAPI key 

## Requirements

```
streamlit==1.29.0
transformers==4.35.0
torch==2.1.0
requests==2.31.0
```

## Contact & Links

**Author**: Nitin Vinayak  
**Email**: nitinvinayak.m@gmail.com  
**LinkedIn**: [linkedin.com/in/nitin-vinayak](https://linkedin.com/in/nitin-vinayak)
