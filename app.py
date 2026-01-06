import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="Yasmin AI", layout="wide")
st.title("🤖 Yasmin: PDF Asistanın")

with st.sidebar:
    api_key = st.text_input("Google API Anahtarı:", type="password")
    uploaded_file = st.file_uploader("PDF Yükle", type="pdf")
    process_button = st.button("Öğren")

if process_button and api_key and uploaded_file:
    try:
        # API Yapılandırması (Doğrudan Google Motoru)
        genai.configure(api_key=api_key)
        
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        # Embeddings ve Veritabanı
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(pages, embeddings)
        st.session_state.vs = vector_store
        st.success("PDF başarıyla analiz edildi!")
    except Exception as e:
        st.error(f"Hata: {e}")

soru = st.text_input("Sorunu sor:")
if soru and "vs" in st.session_state:
    # İlgili metinleri bul
    docs = st.session_state.vs.similarity_search(soru, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # Yanıt Oluşturma (v1beta hatasını atlamak için doğrudan model çağırma)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Bilgi: {context}\n\nSoru: {soru}")
    st.write(response.text)
