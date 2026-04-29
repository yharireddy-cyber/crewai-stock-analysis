### create stock analysis app using streamlit and crew framework
##pip install crew crewai-tools yfinance google-search-results litellm streamlit
import streamlit as st
import os
from crewai import Crew, Agent,Task,LLM
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
from serpapi import GoogleSearch
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Analysis App", page_icon="📈", layout="wide")
st.title("📈 Stock Analysis App Using AI")

st.markdown("Multi Agent system powered by CrewAI and Groq")

## sidebar fo user input
with st.sidebar:
    st.header("API Keys Configuration")
    groq_key = st.text_input("Enter your Groq API Key", type="password")
    st.markdown("[Get your Groq API Key here](https://console.groq.com/keys/)")
    serpapi_key = st.text_input("Enter your SerpAPI Key", type="password")
    st.markdown("[Get your SerpAPI Key here](https://serpapi.com/dashboard)")
    st.markdown("---")
    st.info("Use of Groq and SerpAPI keys is required for the app to function properly. please enter your keys to proceed.")

if not groq_key or not serpapi_key:
    st.warning("Please enter both Groq and SerpAPI keys to use the app.")
    st.stop()

## setup environment variables for crew
os.environ["GROQ_API_KEY"] = groq_key
os.environ["SERPAPI_KEY"] = serpapi_key

## initialize LLM
@st.cache_resource
def get_llm(groq_key):
    ##llama-3.1-8b-instant
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_key, temperature=0.3,
               max_tokens=200)

llm = get_llm(groq_key)

## define tools for agents
class StockSearchInput(BaseModel):
    query: str = Field(..., description="The stock ticker or company name to search for")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol to fetch data for")

class StockSearchTool(BaseTool):
    name: str = "stock_new_search"
    description: str = "Search for latest news and information about a stock using SerpAPI"
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str) -> str:
        "search for news related to the stock using SerpAPI and return the top 5 results"
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "tb": "nws",
                "num": 5
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            news = results.get("news_results", results.get("organic_results"), [])

            output = []
            for item in news[:5]:
                title = item.get("title","")
                snippet = item.get("snippet", item.get("description", ""))
                output.append(f"Title: {title}: {snippet[:100]}")

            return "\n\n".join(output)
        except Exception as e:
            return f"Error fetching news: {str(e)}"

class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceData"
    description: str = "Fetch stock data from Yahoo Finance using yfinance library"
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        "fetch stock data for the given ticker using yfinance and return current price and change percentage"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            if hist.empty:
                return "No data found for the given ticker."
            latest = hist.tail(5)
            current = latest["Close"].iloc[-1]
            change = (latest["Close"].iloc[-1] - latest["Close"].iloc[0]) / latest["Close"].iloc[0] * 100

            return f"""Stock: {ticker.upper()}\n Price: ${current:.2f}\n Change (1 month) {change:.2f}%\nHigh: ${latest["High"].max():.2f}\nLow: ${latest["Low"].min():.2f}"""
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"
        
## initializing the tools
stock_search_tool = StockSearchTool()
yahoo_finance_tool = YahooFinanceTool()

@st.cache_resource
def get_agent(llm):
    analyst = Agent(
        role="Stock Analyst Agent",
        goal="Analyze the stock based on latest news and financial data, and provide insights and recommendations. You MUST use ONLY the two tools available: stock_new_search (for news) and YahooFinanceData (for financial data). return only stock insgihts and recommendations.",
        backstory="You are a stock analyst with expertise in financial markets. You have access to exactly two tools: 1) stock_new_search tool to search for latest news about stocks, and 2) YahooFinanceData tool to fetch stock price and financial data. Use these tools to gather information and provide insights. DO NOT attempt to use any other tools.",
        llm=llm,
        tools=[stock_search_tool, yahoo_finance_tool]
    )

    writer = Agent(
        role="Report Writer Agent",
        goal="Write a simple stock analysis report based on the insights provided by the analyst agent.",
        backstory="You are a skilled writer with expertise in financial writing. Your task is to write a simple stock analysis report based on the insights provided by the analyst agent. The report should be well-structured, informative, and easy to understand for the user and limited in scope  .",
        llm=llm
    )
    return analyst, writer

analyst_agent, writer_agent = get_agent(llm)


## interface for user input
col1,col2 = st.columns((2,1))

with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, GOOGL, TSLA)", value="AAPL", max_chars=6).upper()

with col2:
    analyze_button = st.button("Analyze Stock", type="primary",use_container_width=True)

if analyze_button:
    if not ticker:
        st.error("Please enter a valid stock ticker symbol to anaylze.")

    else:
        with st.spinner(f"Analyzing stock {ticker}..."):
            try:
                new_task = Task(
                    description=f"Search for the latest news about {ticker} and provide insights.",
                    expected_output="Insights and recommendations based on the latest news",
                    agent=analyst_agent
                )

                price_task = Task(
                    description=f"Fetch the latest financial data for {ticker} and provide insights.",
                    expected_output="Insights and recommendations based on the financial data",
                    agent=analyst_agent
                )

                report_task = Task(
                    description=f"Write a comprehensive stock analysis report for {ticker} based on the insights from the analyst agent and keep it under 300 words.",
                    expected_output="A well-structured and informative stock analysis report",
                    agent=writer_agent
                )

                crew = Crew(
                    agents = [analyst_agent, writer_agent],
                    tasks = [new_task, price_task, report_task]
                )
                results = crew.kickoff()

                ## convert to string for display
                result_txt = str(results)
                st.success("Stock analysis completed!")

                st.markdown("-------------------")
                st.markdown(f"### Stock Analysis Report for {ticker}")
                st.markdown(result_txt)
                st.markdown("--------------------")

                ## download option for the report
                st.download_button(
                    label="Download Report",
                    data=result_txt,
                    file_name=f"{ticker}_stock_analysis_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred during stock analysis: {str(e)}")

