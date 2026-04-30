### create stock analysis app using gradio and crew framework
##pip install creai creai-tools yfinance google-search-results litellm gradio pydantic serpapi
import os
import functools
import warnings
import gradio as gr
from crewai import Crew, Agent, Task, LLM
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
from serpapi import GoogleSearch

warnings.filterwarnings("ignore")

@functools.lru_cache(maxsize=4)
def get_llm(groq_key: str) -> LLM:
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.3,
        max_tokens=200,
    )

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
                    "q": f"{query} stock news",
                    "api_key": os.getenv("SERPAPI_KEY"),
                    "tb": "nws",
                    "num": 5,
                    "sort": "date",
        }
            search = GoogleSearch(params)
            results = search.get_dict()

            news = results.get("news_results", results.get("organic_results"), [])
            output = []
            for item in news[:5]:
                title = item.get("title", "")
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
            ##hist = stock.history(period="1mo")
            hist = stock.history(period="5d", interval="1m")
            if hist.empty:
                return "No data found for the given ticker."

            latest = hist.tail(5)
            current = hist["Close"].iloc[-1]
            change = (latest["Close"].iloc[-1] - latest["Close"].iloc[0]) / latest["Close"].iloc[0] * 100
            return (
                f"Stock: {ticker.upper()}\n"
                f"Price: ${current:.2f}\n"
                f"Change (1 month): {change:.2f}%\n"
                f"High: ${latest['High'].max():.2f}\n"
                f"Low: ${latest['Low'].min():.2f}"
            )
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"

stock_search_tool = StockSearchTool()
yahoo_finance_tool = YahooFinanceTool()

def get_agent(llm: LLM):
    analyst = Agent(
        role="Stock Analyst Agent",
        goal=(
            "Analyze the stock based on latest news and financial data, and provide insights "
            "and recommendations. You MUST use ONLY the two tools available: stock_new_search "
            "(for news) and YahooFinanceData (for financial data). Return only stock insights "
            "and recommendations."
        ),
        backstory=(
            "You are a stock analyst with expertise in financial markets. You have access to "
            "exactly two tools: 1) stock_new_search tool to search for latest news about stocks, "
            "and 2) YahooFinanceData tool to fetch stock price and financial data. Use these tools "
            "to gather information and provide insights. DO NOT attempt to use any other tools."
        ),
        llm=llm,
        tools=[stock_search_tool, yahoo_finance_tool],
    )

    writer = Agent(
        role="Report Writer Agent",
        goal="Write a simple stock analysis report based on the insights provided by the analyst agent.",
        backstory=(
            "You are a skilled writer with expertise in financial writing. Your task is to write a simple "
            "stock analysis report based on the insights provided by the analyst agent. The report should "
            "be well-structured, informative, and easy to understand for the user."
        ),
        llm=llm,
    )
    return analyst, writer


def run_analysis(ticker: str, groq_key: str, serpapi_key: str) -> str:
    if not groq_key or not serpapi_key:
        return "Please enter both Groq and SerpAPI keys."
    if not ticker:
        return "Please enter a stock ticker symbol."

    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["SERPAPI_KEY"] = serpapi_key

    try:
        llm = get_llm(groq_key)
        analyst_agent, writer_agent = get_agent(llm)

        new_task = Task(
            description=f"Search for the latest news about {ticker} and provide insights.",
            expected_output="Insights and recommendations based on the latest news",
            agent=analyst_agent,
        )

        price_task = Task(
            description=f"Fetch the latest financial data for {ticker} and provide insights.",
            expected_output="Insights and recommendations based on the financial data",
            agent=analyst_agent,
        )

        report_task = Task(
            description=(
                f"Write a comprehensive stock analysis report for {ticker} based on the insights from "
                "the analyst agent and keep it under 300 words."
            ),
            expected_output="A well-structured and informative stock analysis report",
            agent=writer_agent,
        )

        crew = Crew(agents=[analyst_agent, writer_agent], tasks=[new_task, price_task, report_task])
        results = crew.kickoff()
        return str(results)
    except Exception as e:
        return f"Error during stock analysis: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Stock Analysis App Using AI")
    gr.Markdown("Multi Agent system powered by CrewAI and Groq")

    with gr.Row():
        ticker_input = gr.Textbox(label="Stock Ticker", placeholder="AAPL, GOOGL, TSLA", value="AAPL", max_lines=1)
    with gr.Row():
        groq_input = gr.Textbox(label="Groq API Key", type="password", placeholder="Enter your Groq API key")
        serpapi_input = gr.Textbox(label="SerpAPI Key", type="password", placeholder="Enter your SerpAPI key")

    run_button = gr.Button("Analyze Stock")
    output_box = gr.Textbox(label="Stock Analysis Report", lines=20, interactive=False)

    run_button.click(run_analysis, inputs=[ticker_input, groq_input, serpapi_input], outputs=[output_box])

if __name__ == "__main__":
    demo.launch(share=True)

