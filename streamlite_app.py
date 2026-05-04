### create stock analysis app using streamlit and crew framework
##pip install crew crewai-tools yfinance google-search-results litellm streamlit
import streamlit as st
import os
import math
import datetime
from crewai import Crew, Agent,Task,LLM
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
from serpapi import GoogleSearch
import warnings
import datetime

from yfinance import live
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
    st.info("Use of Groq and SerpAPI keys is required for the app to function properly. For Now keys are integrated with app.")


# Read from Render environment
groq_key = os.getenv("GROQ_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

if not groq_key or not serpapi_key:
    st.warning("Please enter both Groq and SerpAPI keys to use the app.")
    st.stop()

## setup environment variables for crew
os.environ["GROQ_API_KEY"] = groq_key
os.environ["SERPAPI_KEY"] = serpapi_key

## initialize LLM
##@st.cache_resource
def get_llms(groq_key):
    fast_llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.15,
        max_tokens=200
    )

    return fast_llm

fast_llm = get_llms(groq_key)


## define tools for agents
class StockSearchInput(BaseModel):
    query: str = Field(..., description="The stock ticker or company name to search for")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="The stock ticker or company name to fetch data for")

class StockSearchTool(BaseTool):
    name: str = "stock_new_search"
    description: str = "Search for latest news and information about a Stock including after hours as of current date and time only using SerpAPI"
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str) -> str:
        "search for news related to only the stock including after hours as of current date and time  using SerpAPI  and return the top 5 results"
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "tb": "nws",
                "num": 7,
                "sort": "date"
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
    description: str = "Fetch live stock price and recent data from Yahoo Finance using yfinance library"
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            today = datetime.datetime.now().date()

            # --- 1) Get REAL live price (most reliable) ---
            # --- REAL LIVE PRICE (best method) ---
            live = stock.history(period="1d", interval="1m", prepost=True)

            if not live.empty:
                current = float(live["Close"].iloc[-1])
                todays_high = float(live["High"].max())
                ##todays_low = float(live["Low"].min())
                todays_open = float(live["Open"].iloc[0])
            else:
                # fallback to info
                info = stock.get_info()
                current = info.get("regularMarketPrice")
                todays_high = info.get("regularMarketDayHigh")
                ##todays_low = info.get("regularMarketDayLow")
                todays_open = info.get("regularMarketOpen")


            # --- 2) Intraday change ---
            if todays_open:
                intraday_pct = (current - todays_open) / todays_open * 100
            else:
                intraday_pct = 0.0

            # --- 3) one  change ---
            one_year_hist = stock.history(
            start=today - datetime.timedelta(days=365),
            end=today + datetime.timedelta(days=1)
            )

            one_year_change_text = "data unavailable"

            if not one_year_hist.empty and len(one_year_hist) > 1:
                start_price = float(one_year_hist["Close"].iloc[0])
                end_price = float(one_year_hist["Close"].iloc[-1])

            if start_price > 0:
                change = (end_price - start_price) / start_price * 100
                one_year_change_text = f"{change:.2f}%"

            # --- 4) Format output ---
            return (
                f"Stock: {ticker.upper()}\n"
                f"Current price: ${current:.2f}\n"
                f"Intraday change: {intraday_pct:.2f}%\n"
                f"One-year change: {one_year_change_text}\n"
                f"Today's high: ${todays_high}\n"
                ##f"Today's low: ${todays_low}"
            )

        except Exception as e:
            return f"Error fetching stock data: {str(e)}"

                    
## initializing the tools
stock_search_tool = StockSearchTool()
yahoo_finance_tool = YahooFinanceTool()

##@st.cache_resource
def get_agent(fast_llm):
    analyst_and_Writer = Agent(
        role="Stock Analyst and Report Writer Agent",
        goal=(
            "Analyze the stock using ONLY the two provided tools: stock_new_search (for news) and YahooFinanceData (for financial data). "
            "Do NOT use any other tools or external services. Focus only on news related to the company that directly impacts the stock price. "
            "Provide insights and recommendations based on the latest news and financial data. Return ONLY stock insights and recommendations."
            "Write a stock analysis report based on the insights provided by the analyst agent. "
            "Format the output EXACTLY as follows with Markdown bullet points for the Price Snapshot section:\n\n"
            "Summary:\n"
            "- One sentence overview of the company outlook and top 3 factors driving the stock price movement. consider last one year stock price movement and latest news.\n\n"
            "Price Snapshot:\n"
            "- Current price: $[price]\n"
            "- Today's high: $[price]\n"
            ##"- Today's low: $[price]\n"
            "- Intraday change: [percent]\n"
            "- One-year change: [percent]\n\n"
            "Recommendation:\n"
            "- One short conclusion with a buy/hold/sell view.\n"
            "IMPORTANT: Do not present the Price Snapshot as a sentence or paragraph. Each metric must be its own bullet line beginning with '- '. "
            "The summary and recommendation MUST be consistent with the Price Snapshot values. "
            "Keep recommendation aligned with the data and keep it under 100 words. Do not mention specific price or percentage values in the summary, just a general performance overview."
        ),
        backstory=(
            "You are a stock analyst with expertise in financial markets. You have access to ONLY TWO TOOLS: 'stock_new_search' for news and 'YahooFinanceData' for financial data. "
            "Do NOT attempt to use any other tools, functions, or external services. Do NOT invent tools. Only use the tools provided in this system. "
            "Use the available tools to surface the most recent and relevant information. "
            "Focus on news that has a direct impact on the stock price, such as earnings reports, product launches, regulatory changes, or market trends. "
            "Provide insights and recommendations based on the data you gather. Your output should be concise and focused on actionable insights for the user."
            "You are a skilled financial writer. Produce a concise and professional report using the requested sections. "
            "Make sure the summary  matches the price movement and that the recommendation is aligned with the data."
            "DO NOT mention stock price or pertage in summary just keep one line simple summary on stock or company performance from one year from analyze agent."),
        llm=fast_llm,
        tools=[stock_search_tool, yahoo_finance_tool]
    )

    return analyst_and_Writer

analyst_and_Writer = get_agent(fast_llm=fast_llm)


## interface for user input
col1,col2 = st.columns((2,1))

with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, GOOGL, TSLA)", value="AAPL", max_chars=10).upper()

with col2:
    analyze_button = st.button("Analyze Stock", type="primary",use_container_width=True)

if analyze_button:
    if not ticker:
        st.error("Please enter a valid stock ticker symbol to anaylze.")

    else:
        with st.spinner(f"Analyzing stock {ticker}..."):
            try:
                analyst_and_Writer_report_task = Task(
                    description=f"Search and Fetch latest  news and financial data for  {ticker} and provide insights. using ONLY the two provided tools: stock_new_search (for news) and YahooFinanceData (for financial data) Write a comprehensive stock analysis report for {ticker} based on the insights from the analyst agent",
                    expected_output="Insights and recommendations based on the latest news and financial data. A well-structured and informative stock analysis report",
                    agent=analyst_and_Writer
                )
                crew = Crew(
                    agents = [analyst_and_Writer],
                    tasks = [analyst_and_Writer_report_task]
                )
                results = crew.kickoff()
               

                if results is not None:
                    ## convert to string for display
                    result_txt = str(results)
                    st.success("Stock analysis completed. The summary and recommendations reflect one year of historical performance and the most recent news, while the price snapshot uses live market data.")

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

