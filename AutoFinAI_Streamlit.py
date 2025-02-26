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
