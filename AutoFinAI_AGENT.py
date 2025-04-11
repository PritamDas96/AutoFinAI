
import streamlit as st
st.set_page_config(page_title="📊 Finance Analyst Agent", layout="wide") 

from phi.agent import Agent
from phi.model.groq import Groq
import os
import yfinance as yf

# ✅ Set your Groq API key
os.environ["GROQ_API_KEY"] = "gsk_..."

# ✅ Company to symbol map
company_symbols = {
    "Infosys": "INFY",
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
}

# ✅ Tool 1: Get symbol for a company
def get_company_symbol(company: str) -> str:
    return company_symbols.get(company, "Unknown")

# ✅ Tool 2: Fetch latest quarterly financials using yfinance
import matplotlib.pyplot as plt

def get_latest_quarterly_financials(symbol: str) -> str:
    ticker = yf.Ticker(symbol)
    try:
        df = ticker.quarterly_financials.T
        if df.empty:
            return f"No data available for {symbol}"
        df.index = df.index.strftime('%b %Y')

        # ✅ All metrics for table
        markdown_table = df.to_markdown()

        # ✅ Top 3 metrics for plotting
        plot_metrics = [col for col in ["Total Revenue", "Net Income", "Gross Profit"] if col in df.columns]
        if plot_metrics:
            fig, ax = plt.subplots(figsize=(4, 2))  # Compact figure size
            df[plot_metrics].plot(kind='bar', ax=ax)

            ax.set_title(f"{symbol} - Key Financials", fontsize=10)
            ax.set_ylabel("USD", fontsize=9)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.legend(fontsize=8)

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        # ✅ Return markdown table (with all metrics)
        return """### {} - Quarterly Financials
{}""".format(symbol, df.to_markdown())
    except Exception as e:
        return f"Error fetching data for {symbol}: {e}"



# ✅ Cached Agent
@st.cache_resource
def create_agent():
    return Agent(
        model=Groq(id="llama3-8b-8192"),
        tools=[get_company_symbol, get_latest_quarterly_financials],
        instructions=[
            "Use markdown tables to display financial data.",
            "Use get_company_symbol to resolve company names.",
            "Use get_latest_quarterly_financials to retrieve data.",
        ],
        markdown=True,
        debug_mode=True,
    )

agent = create_agent()

# ✅ Streamlit UI
st.title("💼 AutoFin-AI Agent ")

selected_companies = st.multiselect(
    "Select Companies:",
    options=list(company_symbols.keys()),
    default=["Microsoft"]
)

user_question = st.text_area("Ask a financial question:")

# 🔁 Enhance vague user prompts by injecting selected stock symbols
symbols = [company_symbols[c] for c in selected_companies]

if not user_question:
    # If no question, auto-generate a default prompt
    user_question = f"Show me the latest quarterly financials for {' and '.join(symbols)}."
else:
    lowered_question = user_question.lower()
    if (
        "stock" in lowered_question 
        or "review" in lowered_question 
        or "compare" in lowered_question 
        or "financial" in lowered_question
        or "quarter" in lowered_question
    ):
        # Append symbol names if user gives a vague question
        user_question += f" for {' and '.join(symbols)}"


if st.button("🔍 Analyze"):
    if not selected_companies:
        st.warning("Please select at least one company.")
    else:
        with st.spinner("Analyzing..."):
            response = agent.run(user_question)

            if hasattr(response, "output"):
                st.markdown(response.output)
            elif hasattr(response, "content"):
                st.markdown(response.content)
            elif hasattr(response, "text"):
                st.markdown(response.text)
            else:
                st.write("❗ Unknown response structure:")
                st.write(response)
  
         
