import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Sayfa Başlığı
st.set_page_config(page_title="Yasmin AI", layout="wide")
st.title("🤖 Yasmin: PDF Asistanın")

# Yan Menü
with st.sidebar:
    st.header("Ayarlar")
    # Değişken adını burada 'api_key' olarak sabitliyoruz
    api_key = st.text_input("Google API Anahtarını Gir:", type="password")
    uploaded_file = st.file_uploader("PDF Dosyanı Yükle", type="pdf")
    process_button = st.button("Öğren")

# Ana İşlem (Öğren Butonuna Basılınca)
if process_button and api_key and uploaded_file:
    with st.spinner("Dosya analiz ediliyor..."):
        try:
            # 1. Google Ayarları (Genel)
            genai.configure(api_key=api_key)
            
            # 2. Dosyayı Geçici Olarak Kaydet
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 3. PDF'i Oku ve Böl
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            
            # 4. Embeddings (Google'ın Hızlı Modeli - Hata vermez)
            # Burada 'api_key' değişkenini kullanıyoruz, artık isim hatası almazsın
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )
            
            # 5. Veritabanını Kur
            vector_store = FAISS.from_documents(pages, embeddings)
            st.session_state.vs = vector_store
            st.success("Harika! Dosyayı öğrendim. Şimdi bana soru sorabilirsin.")
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

# Soru Sorma Kısmı
soru = st.text_input("Sorunu buraya yaz:")

if soru and "vs" in st.session_state:
    try:
        # En alakalı kısımları bul
        docs = st.session_state.vs.similarity_search(soru, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Cevap Üretme (Doğrudan Google'a bağlanır, v1beta hatası vermez)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"Sen yardımsever bir asistansın. Bu bilgilere göre cevap ver:\n\nBilgi: {context}\n\nSoru: {soru}")
        
        st.write(response.text)
        
    except Exception as e:
        st.error(f"Cevap üretirken hata: {e}")


