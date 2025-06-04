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
from langchain import HuggingFaceHub

warnings.filterwarnings("ignore")

# ----------------------------
# News API Wrapper
# ----------------------------
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
    except:
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
    if isinstance(input, dict):
        topic = input.get("topic") or next(iter(input.values()))
    else:
        topic = input
    return fetch_tech_news(topic, api_key)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Tech News Trend Analyzer", layout="wide")
st.title("📰 Tech News Trend Analyzer (Multi-Agent, Free API)")

topic = st.text_input("💡 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Enter your FREE Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ Enter your News API Key", type="password")
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
            except Exception as e:
                st.error(f"❌ LLM error: {str(e)}")
                st.stop()

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

            try:
                result = crew.kickoff(inputs={"topic": topic})
                st.success("✅ Done!")

                # Display news results
                st.markdown("## 🗞 News Articles")
                news_items = fetch_tech_news(topic, news_api_key)
                if news_items:
                    for item in news_items:
                        with st.expander(item["title"]):
                            st.write(item["description"])
                            st.write(f"**Author:** {item['author']}")
                            st.markdown(f"[Read more]({item['url']})")
                else:
                    st.warning("No news articles fetched.")

                # Display trends
                st.markdown("## 📈 Trending Keywords")
                trends = str(task_trend.output).strip().split("\n")
                keywords = [re.sub(r"[-\u2022]\s*", "", k) for k in trends if k.strip()]
                if keywords:
                    df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
                    fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
                    st.plotly_chart(fig, use_container_width=True)

                # Display summary
                st.markdown("## 🧠 Summary")
                st.markdown(str(task_summary.output))

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
