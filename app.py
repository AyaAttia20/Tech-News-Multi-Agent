import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import requests
import warnings
import pandas as pd
import plotly.express as px
import re

from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_community.llms import HuggingFaceHub

warnings.filterwarnings("ignore")

def fetch_tech_news(topic, api_key):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10
    }
    response = requests.get(url, params=params)
    try:
        data = response.json()
    except Exception:
        return []

    results = []
    for article in data.get("articles", []):
        results.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "author": article.get("author"),
            "url": article.get("url")
        })
    return results

def fetch_wrapper(input, api_key):
    topic = input.get("topic") if isinstance(input, dict) else input
    if not topic:
        return []
    return fetch_tech_news(topic, api_key)

st.set_page_config(page_title="Tech News Trend Analyzer", layout="wide")
st.title("📰 Tech News Trend Analyzer (Multi-Agent, Free API)")

topic = st.text_input("💡 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ News API Key", type="password")
run_button = st.button("🚀 Analyze")

if run_button:
    if not hf_token or not news_api_key:
        st.error("❌ Please enter both the Hugging Face and News API keys.")
    else:
        with st.spinner("Running agents..."):
            try:
                llm = HuggingFaceHub(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=hf_token,
                    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
                )

                tool = Tool(
                    name="TechNewsFetcher",
                    func=lambda x: fetch_wrapper(x, news_api_key),
                    description="Fetch recent technology news articles.",
                    return_direct=True
                )

                fetcher = Agent(
                    role="News Fetcher",
                    goal="Get recent news on a topic",
                    tools=[tool],
                    backstory="Expert at sourcing tech news from the web.",
                    verbose=True,
                    llm=llm
                )

                summarizer = Agent(
                    role="Summarizer",
                    goal="Summarize the fetched news",
                    backstory="Skilled at condensing information.",
                    verbose=True,
                    llm=llm
                )

                trend_analyzer = Agent(
                    role="Trend Detector",
                    goal="Extract main keywords",
                    backstory="NLP expert for trend recognition.",
                    verbose=True,
                    llm=llm
                )

                task_fetch = Task(
                    description=f"Fetch latest news on '{topic}'.",
                    expected_output="List of news with title, summary, author, and URL.",
                    agent=fetcher
                )

                task_summary = Task(
                    description="Summarize the collected news articles in brief.",
                    expected_output="Summary paragraph covering main points.",
                    agent=summarizer,
                    context=[task_fetch]
                )

                task_trend = Task(
                    description="Extract trending keywords from news titles and descriptions.",
                    expected_output="Top 10 keywords.",
                    agent=trend_analyzer,
                    context=[task_fetch]
                )

                crew = Crew(
                    agents=[fetcher, summarizer, trend_analyzer],
                    tasks=[task_fetch, task_summary, task_trend],
                    verbose=True
                )

                result = crew.kickoff(inputs={"topic": topic})
                st.success("✅ Done!")

                news_items = fetch_tech_news(topic, news_api_key)
                st.markdown("## 🗞 News Articles")
                if news_items:
                    for item in news_items:
                        with st.expander(item["title"]):
                            st.write(item["description"])
                            st.write(f"**Author:** {item['author']}")
                            st.markdown(f"[Read more]({item['url']})")
                else:
                    st.warning("No news articles fetched.")

                st.markdown("## 📈 Trending Keywords")
                trends = getattr(task_trend, 'output', "")
                if trends:
                    trends_list = str(trends).strip().split("\n")
                    keywords = [re.sub(r"[-•]\s*", "", k.strip()) for k in trends_list if k.strip()]
                    if keywords:
                        df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
                        fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trending keywords found.")
                else:
                    st.info("No trending keywords output.")

                st.markdown("## 🧠 Summary")
                summary_output = getattr(task_summary, 'output', "")
                if summary_output:
                    st.markdown(str(summary_output))
                else:
                    st.info("No summary output.")

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
