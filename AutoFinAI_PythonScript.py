# 4 STEPS
# 🔹 Step 1: Fetch Financial Data Using Yahoo Finance (yfinance)
# 🔹 Step 2: Implement Retrieval-Augmented Generation (FAISS + Sentence Transformers) 
# 🔹 Step 3: Use TinyLlama for AI-Powered Financial Analysis
# 🔹 Step 4: Example Execution
#==================================================================================================
import yfinance as yf
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import streamlit as st

# 🔹 Step 1: Fetch Financial Data Using Yahoo Finance (yfinance)
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
  
#--------------------------------------------------------------------------------------------------------------------------------

# 🔹 Step 2: Implement Retrieval-Augmented Generation (FAISS + Sentence Transformers)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(documents):
    vectors = embedding_model.encode(documents)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index

#--------------------------------------------------------------------------------------------------------------------------------

# 🔹 Step 3: Use TinyLlama for AI-Powered Financial Analysis
def load_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto" if torch.cuda.is_available() else None
    ).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_tinyllama()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=250)
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

#--------------------------------------------------------------------------------------------------------------------------------

# 🔹 Step 4: Example Execution
ticker = "AAPL"
documents = fetch_stock_data(ticker)
index = create_faiss_index(documents)

# Example Queries
query1 = "What is Apple's recent net income?"
insight1 = get_financial_insight(query1, documents, index)
print(f"🧠 Insight: {insight1}")

query2 = "Should I invest in Apple based on its cash flow?"
insight2 = get_financial_insight(query2, documents, index)
print(f"🧠 Insight: {insight2}")

#==================================================================================================
