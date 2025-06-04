import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import requests
import pandas as pd
import re
import json

from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

# --------------------------
# Fetch news from RSS feed (TechCrunch example)
def fetch_news(topic):
    url = f"https://techcrunch.com/tag/{topic}/feed/"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    import feedparser
    feed = feedparser.parse(resp.text)
    results = []
    for entry in feed.entries[:10]:
        results.append({
            "title": entry.title,
            "summary": entry.get("summary", ""),
            "link": entry.link
        })
    return results

def fetch_wrapper(input):
    topic = input if isinstance(input, str) else input.get("topic", "")
    return fetch_news(topic)

# --------------------------
# Streamlit UI
st.set_page_config(page_title="Tech News Digest", layout="wide")
st.title("📰 Tech News Digest with CrewAI")

topic = st.text_input("🔎 Enter topic (e.g. AI, Security, Blockchain)", "AI")
run = st.button("Run Analysis")

if run:
    if not topic.strip():
        st.error("Please enter a valid topic!")
    else:
        with st.spinner("Running multi-agent system..."):

            # Initialize LLM (use your free OpenRouter or any open model)
            api_key = st.text_input("Enter your OpenRouter API Key (optional for some open models)", type="password")

            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                api_key=api_key or None,
                temperature=0.5,
                max_tokens=512
            )

            # Tool to fetch news
            tool_fetch = Tool(
                name="NewsFetcher",
                func=fetch_wrapper,
                description="Fetch latest news articles by topic",
                return_direct=True
            )

            # Agents
            fetcher = Agent(
                role="News Fetcher",
                goal="Fetch latest news articles on given topic",
                tools=[tool_fetch],
                backstory="Expert at fetching news.",
                verbose=True,
                llm=llm
            )

            analyzer = Agent(
                role="News Analyzer",
                goal="Extract keywords and summarize news articles",
                backstory="Expert in NLP summarization and keyword extraction.",
                verbose=True,
                llm=llm
            )

            reporter = Agent(
                role="Reporter",
                goal="Generate final report and save it to a file",
                backstory="Writes clear reports and saves outputs.",
                verbose=True,
                llm=llm
            )

            # Tasks
            fetch_task = Task(
                description=f"Fetch news on topic '{topic}'",
                expected_output="List of news articles with title, summary, and link",
                agent=fetcher
            )

            analyze_task = Task(
                description="Analyze news and extract top 5 keywords and summaries",
                expected_output="Keywords list and summaries",
                agent=analyzer,
                context=[fetch_task]
            )

            report_task = Task(
                description="Generate a final report combining news and analysis, then save to file",
                expected_output="Report text",
                agent=reporter,
                context=[fetch_task, analyze_task]
            )

            crew = Crew(
                agents=[fetcher, analyzer, reporter],
                tasks=[fetch_task, analyze_task, report_task],
                verbose=True
            )

            inputs = {"topic": topic}
            crew.kickoff(inputs=inputs)

            # Display fetched news
            st.markdown("## 📰 Latest News Articles")
            try:
                news_list = fetch_task.output
                if not news_list or not isinstance(news_list, list):
                    news_list = []
            except Exception:
                news_list = []

            for news in news_list:
                st.markdown(f"### [{news['title']}]({news['link']})")
                st.markdown(news['summary'])

            # Display analysis
            st.markdown("## 🔍 Analysis")
            st.write(analyze_task.output or "No analysis available.")

            # Display report and provide download link
            report_text = report_task.output or "No report generated."
            st.markdown("## 📄 Final Report")
            st.text_area("Report", report_text, height=300)

            # Save report to file
            with open("report.txt", "w", encoding="utf-8") as f:
                f.write(report_text)

            st.markdown("[Download Report](report.txt)")

