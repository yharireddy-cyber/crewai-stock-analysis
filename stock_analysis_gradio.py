import gradio as gr
import os
import datetime
import yfinance as yf
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 1. TOOLS
# -----------------------------

class StockSearchInput(BaseModel):
    query: str = Field(..., description="The stock ticker or company name to search for")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="The stock ticker or company name to fetch data for")


class StockSearchTool(BaseTool):
    name: str = "stock_new_search"
    description: str = "Search for latest stock news using SerpAPI"
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str):
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "tb": "nws",
                "num": 4,
                "sort": "date"
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            news = results.get("news_results") or results.get("organic_results") or []

            output = []
            for item in news[:5]:
                title = item.get("title", "")
                snippet = item.get("snippet", item.get("description", ""))
                output.append(f"Title: {title}: {snippet[:100]}")

            return {"news": output}

        except Exception as e:
            return {"news": [f"Error fetching news: {str(e)}"]}


class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceData"
    description: str = "Fetch live stock price and recent data from Yahoo Finance"
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str):
        try:
            stock = yf.Ticker(ticker)
            today = datetime.datetime.now().date()

            # Live price
            live = stock.history(period="1d", interval="1m", prepost=True)

            if not live.empty:
                current = float(live["Close"].iloc[-1])
                todays_high = float(live["High"].max())
                todays_open = float(live["Open"].iloc[0])
            else:
                info = stock.get_info()
                current = info.get("regularMarketPrice")
                todays_high = info.get("regularMarketDayHigh")
                todays_open = info.get("regularMarketOpen")

            change_percent = stock.get_info().get("regularMarketChangePercent")

            # Intraday %
            intraday_pct = (
                (current - todays_open) / todays_open * 100 if todays_open else 0.0
            )

            # One-year %
            one_year_hist = stock.history(
                start=today - datetime.timedelta(days=365),
                end=today + datetime.timedelta(days=1)
            )

            one_year_change = 0.0
            if not one_year_hist.empty and len(one_year_hist) > 1:
                start_price = float(one_year_hist["Close"].iloc[0])
                end_price = float(one_year_hist["Close"].iloc[-1])
                if start_price > 0:
                    one_year_change = (end_price - start_price) / start_price * 100

            return {
                "ticker": ticker.upper(),
                "current": round(current, 2),
                "change_percent": round(change_percent, 2),
                "intraday_pct": round(intraday_pct, 2),
                "one_year_change": round(one_year_change, 2),
                "todays_high": round(todays_high, 2)
            }

        except Exception as e:
            return {"error": str(e)}


# -----------------------------
# 2. AGENT SETUP
# -----------------------------

def get_agent(groq_key):
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.15,
        max_tokens=200
    )

    agent = Agent(
        role="Stock Analyst and Report Writer Agent",
        tools=[StockSearchTool(), YahooFinanceTool()],
        goal=(
            "You MUST use ONLY the JSON fields from YahooFinanceData and stock_new_search. Copy values exactly. No hallucination. Format output as Summary, Price Snapshot, Recommendation."
            "Format the output EXACTLY as follows:\n\n"
            "Summary:\n"
            "- One sentence overview of the company outlook and top 3 factors driving the stock price movement using JSON field news returned by  stock_search_tool.\n\n"
            "Price Snapshot:\n"
            "- Current price: $[current]\n"
            "- Today's percent Change: [change_percent]%\n"
            "- Today's high: $[todays_high]\n"
            "- Intraday percent change: [intraday_pct]%\n"
            "- One-year percent change: [one_year_change]%\n\n"
            "Recommendation:\n"
            "- One short conclusion with a buy/hold/sell view based on the bews provided by YahooFinanceData and stock_search_tool. do not include any stock price or percentage details here\n\n"
            "IMPORTANT RULES:\n"
            "- DO NOT change any numbers.\n"
        ),
        backstory=(
            "You are a stock analyst with expertise in financial markets. You have access to ONLY TWO TOOLS: 'stock_new_search' for news and 'YahooFinanceData' for financial data. and donot hallucinate data if yfinance fails to fetch live data. Use these tools to gather information and provide insights. Your task is to analyze the stock based on the latest news and financial data, and provide insights and recommendations. "
            "Do NOT attempt to use any other tools, functions, or external services. Do NOT invent tools. Only use the tools provided in this system. "
            "Focus on news that has a direct impact on the stock price, such as earnings reports, product launches, regulatory changes, or market trends. "
            "Make sure the summary  matches the price movement and that the recommendation is aligned with the data."
            "DO NOT mention stock price or pertage in summary just keep one line simple summary on stock or company performance from one year from analyze agent."),
        llm=llm
        )
    return agent


# -----------------------------
# 3. MAIN FUNCTION FOR GRADIO
# -----------------------------

def analyze_stock(ticker, groq_key, serpapi_key):
    if not groq_key or not serpapi_key:
        return "❌ Missing API keys. Please enter both keys."

    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["SERPAPI_KEY"] = serpapi_key

    agent = get_agent(groq_key)

    task = Task(
        description=f"Analyze stock {ticker} using tools.",
        expected_output="Stock analysis report",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    return str(result)


# -----------------------------
# 4. GRADIO UI
# -----------------------------

with gr.Blocks(title="Stock Analysis App") as demo:

    gr.Markdown("## 📈 Stock Analysis App (CrewAI + Groq + SerpAPI)")

    groq_key = gr.Textbox(label="Groq API Key", type="password")
    serpapi_key = gr.Textbox(label="SerpAPI Key", type="password")
    ticker = gr.Textbox(label="Stock Ticker", value="AAPL")

    analyze_btn = gr.Button("Analyze Stock")

    output = gr.Textbox(label="Stock Analysis Report", lines=20)

    analyze_btn.click(
        analyze_stock,
        inputs=[ticker, groq_key, serpapi_key],
        outputs=output
    )

demo.launch()
