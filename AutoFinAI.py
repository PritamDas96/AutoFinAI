import yfinance as yf

# Define stock ticker
ticker = "AAPL"

# ================================================
# ✅ 1. Fetch Stock Data from Yahoo Finance (No API Key Required)
# ================================================
stock = yf.Ticker(ticker)

# Get historical stock prices (Last 5 years)
historical_data = stock.history(period="5y")
print("📌 Yahoo Finance: Historical Stock Data")
print(historical_data.head())

# Get company net income (Replaces deprecated Ticker.earnings)
income_statement = stock.income_stmt
if "Net Income" in income_statement.index:
    net_income = income_statement.loc["Net Income"]
    print("\n📌 Yahoo Finance: Net Income from Income Statement")
    print(net_income)
else:
    print("\n⚠️ Net Income data not available.")

# Get balance sheet
balance_sheet = stock.balance_sheet
print("\n📌 Yahoo Finance: Balance Sheet")
print(balance_sheet)

# Get cash flow statement
cash_flow = stock.cashflow
print("\n📌 Yahoo Finance: Cash Flow Statement")
print(cash_flow)

# Get stock dividends history
dividends = stock.dividends
if not dividends.empty:
    print("\n📌 Yahoo Finance: Dividend History")
    print(dividends.tail())
else:
    print("\n⚠️ No dividend data available.")

# Get stock splits history
splits = stock.splits
if not splits.empty:
    print("\n📌 Yahoo Finance: Stock Splits History")
    print(splits)
else:
    print("\n⚠️ No stock split data available.")

# Get recent analyst recommendations
recommendations = stock.recommendations
if recommendations is not None and not recommendations.empty:
    print("\n📌 Yahoo Finance: Analyst Recommendations")
    print(recommendations.tail())
else:
    print("\n⚠️ No analyst recommendation data available.")
#--------------------------------------------------------------------------------------------------------------------------------


import yfinance as yf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================================================
# ✅ Step 1: Fetch Stock Data (From Step 1)
# ================================================
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get financial statements
income_statement = stock.income_stmt
balance_sheet = stock.balance_sheet
cash_flow = stock.cashflow
historical_data = stock.history(period="5y")

# Print available cash flow keys to debug missing values
print("\n📌 Available Cash Flow Keys:", cash_flow.index)

# Extract the most recent values properly
def get_latest_value(data, keys):
    """Returns the latest available value for a given list of possible keys from financial statements."""
    for key in keys:
        if key in data.index:
            latest_value = data.loc[key].dropna().values  # Drop NaN values and get latest
            if len(latest_value) > 0:
                return latest_value[0]  # Return first valid value
    return "Data not available"

# Fetch key financial metrics with alternative keys
net_income = get_latest_value(income_statement, ["Net Income"])
total_assets = get_latest_value(balance_sheet, ["Total Assets"])
operating_cash_flow = get_latest_value(cash_flow, [
    "Total Cash From Operating Activities",
    "Operating Cash Flow",
    "Net Cash Provided by Operating Activities",
    "Cash Flow from Operations"
])

# Convert financial data into structured text
documents = [
    f"Net Income: {net_income}",
    f"Total Assets: {total_assets}",
    f"Operating Cash Flow: {operating_cash_flow}",
    f"Stock history (Recent 5 days):\n{historical_data.tail(5).to_string()}"
]

# ================================================
# ✅ Step 2: Implement RAG with FAISS
# ================================================

# Load a pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert financial statements into vector embeddings
vectors = model.encode(documents)

# Create a FAISS index for fast retrieval
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# ================================================
# ✅ Step 3: Query the AI Financial Analyst
# ================================================
def get_financial_insight(query):
    """Retrieves the most relevant financial insight for a given query."""
    query_vec = model.encode([query])
    _, result = index.search(query_vec, k=1)
    return documents[result[0][0]]

# Example Queries
print("\n📌 AI Financial Analyst (RAG) - Example Queries")
user_query = "What is Apple's recent net income?"
insight = get_financial_insight(user_query)
print(f"🧠 Insight: {insight}")

user_query2 = "How much cash flow does Apple generate?"
insight2 = get_financial_insight(user_query2)
print(f"🧠 Insight: {insight2}")

#-----------------------------------------------------------------------------------------------------------------------

import yfinance as yf
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ================================================
# ✅ Step 1: Fetch Stock Data
# ================================================
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get financial statements
income_statement = stock.income_stmt
balance_sheet = stock.balance_sheet
cash_flow = stock.cashflow
historical_data = stock.history(period="5y")

# Extract the most recent values properly
def get_latest_value(data, keys):
    """Returns the latest available value for a given key from financial statements."""
    for key in keys:
        if key in data.index:
            latest_value = data.loc[key].dropna().values
            if len(latest_value) > 0:
                return latest_value[0]  
    return "Data not available"

# Fetch key financial metrics
net_income = get_latest_value(income_statement, ["Net Income"])
total_assets = get_latest_value(balance_sheet, ["Total Assets"])
operating_cash_flow = get_latest_value(cash_flow, [
    "Total Cash From Operating Activities",
    "Operating Cash Flow",
    "Net Cash Provided by Operating Activities"
])

# Convert financial data into structured text
documents = [
    f"Net Income: {net_income}",
    f"Total Assets: {total_assets}",
    f"Operating Cash Flow: {operating_cash_flow}",
    f"Stock history (Recent 5 days):\n{historical_data.tail(5).to_string()}"
]

# ================================================
# ✅ Step 2: Implement RAG with FAISS
# ================================================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert financial statements into vector embeddings
vectors = embedding_model.encode(documents)

# Create a FAISS index for fast retrieval
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# ================================================
# ✅ Step 3: Integrate TinyLlama for AI-Powered Financial Insights
# ================================================

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"


tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cpu"  # Force CPU usage on Streamlit Cloud

# Load model safely with error handling
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device)
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = None  # Prevent Streamlit from crashing




def generate_response(prompt):
    """Generates AI response using TinyLlama."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_financial_insight(query):
    """Retrieves the most relevant financial insight and generates a response using TinyLlama."""
    query_vec = embedding_model.encode([query])
    _, result = index.search(query_vec, k=1)
    retrieved_info = documents[result[0][0]]

    prompt = f"""
    You are a financial analyst providing stock insights based on real data. 
    Given the following financial information:
    {retrieved_info}

    Answer the question in a detailed and structured way:
    {query}
    """

    return generate_response(prompt)

# Example Queries
print("\n📌 AI Financial Analyst (TinyLlama + RAG) - Example Queries")

user_query1 = "What is Apple's recent net income?"
insight1 = get_financial_insight(user_query1)
print(f"🧠 Insight: {insight1}")

user_query2 = "Should I invest in Apple based on its cash flow?"
insight2 = get_financial_insight(user_query2)
print(f"🧠 Insight: {insight2}")


#------------------------------------------------------------------------------------------------------------


import streamlit as st
import yfinance as yf
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ================================================
# ✅ Step 1: Stock Symbol Selection
# ================================================
STOCK_SYMBOLS = {
    "Apple Inc. (AAPL)": "AAPL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Meta Platforms Inc. (META)": "META",
    "Netflix Inc. (NFLX)": "NFLX",
    "JP Morgan Chase (JPM)": "JPM",
    "Bank of America (BAC)": "BAC"
}

st.title("📊 AI-Powered Financial Analyst")
st.markdown("**Analyze stock performance using AI-driven insights.**")

stock_choice = st.selectbox("Choose a stock to analyze:", list(STOCK_SYMBOLS.keys()))
ticker = STOCK_SYMBOLS[stock_choice]

# ================================================
# ✅ Step 2: Fetch Stock Data
# ================================================
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)

    income_statement = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    historical_data = stock.history(period="5y")

    def get_latest_value(data, keys):
        for key in keys:
            if key in data.index:
                latest_value = data.loc[key].dropna().values
                if len(latest_value) > 0:
                    return latest_value[0]  
        return "Data not available"

    net_income = get_latest_value(income_statement, ["Net Income"])
    total_assets = get_latest_value(balance_sheet, ["Total Assets"])
    operating_cash_flow = get_latest_value(cash_flow, [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Net Cash Provided by Operating Activities"
    ])

    documents = [
        f"Net Income: {net_income}",
        f"Total Assets: {total_assets}",
        f"Operating Cash Flow: {operating_cash_flow}",
        f"Stock history (Recent 5 days):\n{historical_data.tail(5).to_string()}"
    ]

    return documents

# ================================================
# ✅ Step 3: Implement RAG with FAISS
# ================================================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(documents):
    vectors = embedding_model.encode(documents)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index

# ================================================
# ✅ Step 4: Load TinyLlama with Fixed Model Loading
# ================================================
@st.cache_resource
def load_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None  # Auto-detect GPU
    ).to(device)

    return tokenizer, model, device

tokenizer, model, device = load_tinyllama()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=250)  # ✅ FIXED: Use max_new_tokens instead of max_length
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_financial_insight(query, documents, index):
    query_vec = embedding_model.encode([query])
    _, result = index.search(query_vec, k=1)
    retrieved_info = documents[result[0][0]]

    prompt = f"""
    You are a financial analyst providing stock insights based on real data. 
    Given the following financial information:
    {retrieved_info}

    Answer the question in a detailed and structured way:
    {query}
    """

    return generate_response(prompt)

# ================================================
# ✅ Step 5: Streamlit Web App
# ================================================
if "documents" not in st.session_state:
    st.session_state.documents = None
    st.session_state.index = None

if st.button("Analyze"):
    with st.spinner("Fetching financial data..."):
        st.session_state.documents = fetch_stock_data(ticker)
        st.session_state.index = create_faiss_index(st.session_state.documents)

    st.success("✅ Data fetched successfully! Now ask your financial questions.")

if st.session_state.documents:
    query = st.text_area("Ask a financial question about the stock:", "What is the recent net income?")

    if st.button("Get AI Insights"):
        with st.spinner("Generating AI-powered financial analysis..."):
            insight = get_financial_insight(query, st.session_state.documents, st.session_state.index)
            st.subheader("🧠 AI Insight")
            st.write(insight)
#-------------------------------------------------------------------------------------------------------------------------------
