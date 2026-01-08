import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yasmin AI", page_icon="🧚‍♀️", layout="wide")

# --- TASARIM (CSS) ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(to right, #141e30, #243b55); color: white; }
    h1 { text-align: center; color: #FFD700; text-shadow: 0 0 10px #FFD700; }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 10px; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Yasmin: Hafızalı Asistanın ✨</h1>", unsafe_allow_html=True)

# --- SOHBET GEÇMİŞİNİ BAŞLAT (HAFIZA) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # 1. API Anahtarı
    api_key = st.text_input("Google API Anahtarını Gir:", type="password")
    
    # 2. Model Seçici (Arkadaşların da kullanabilsin diye)
    secilen_model = st.selectbox(
        "Kullanılacak Zeka Modeli",
        ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro")
    )
    
    # 3. Dosya Yükleme
    uploaded_file = st.file_uploader("Bir PDF Dosyası Yükle", type="pdf")
    
    # 4. Eğit Butonu
    if st.button("🧠 Yasmin'i Eğit"):
        if not api_key or not uploaded_file:
            st.error("Lütfen önce API anahtarı ve dosya gir.")
        else:
            with st.spinner("Dosyayı inceliyorum... 🧚‍♀️"):
                try:
                    genai.configure(api_key=api_key)
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader("temp.pdf")
                    pages = loader.load_and_split()
                    
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004", 
                        google_api_key=api_key
                    )
                    
                    st.session_state.vs = FAISS.from_documents(pages, embeddings)
                    st.balloons() 
                    st.success("Hazırım! Artık sohbet edebiliriz. 🎉")
                    # Yeni dosya yüklenince hafızayı temizle
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Hata: {e}")

# --- SOHBET GEÇMİŞİNİ EKRANA YAZDIR ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- KULLANICI GİRİŞİ (CHAT INPUT) ---
# Artık text_input yerine chat_input kullanıyoruz, bu daha modern durur
soru = st.chat_input("Yasmin'e bir şey sor...")

if soru:
    if "vs" not in st.session_state:
        st.warning("Lütfen önce sol menüden PDF yükle ve 'Eğit' butonuna bas.")
    else:
        # 1. Kullanıcı mesajını ekrana ve hafızaya ekle
        st.session_state.messages.append({"role": "user", "content": soru})
        with st.chat_message("user"):
            st.write(soru)

        # 2. Cevap Üret
        try:
            # Bağlamı PDF'ten çek
            docs = st.session_state.vs.similarity_search(soru, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Geçmiş sohbeti de modele gönderiyoruz ki "Bunu çöz" dediğinde neyi kastettiğini anlasın
            gecmis_sohbet = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            
            model = genai.GenerativeModel(secilen_model)
            
            prompt = f"""
            Sen Yasmin adında yardımcı bir asistansın.
            
            Geçmiş Konuşmalarımız:
            {gecmis_sohbet}
            
            PDF Bilgisi:
            {context}
            
            Son Soru: {soru}
            
            Lütfen geçmiş konuşmaları dikkate alarak cevap ver. Eğer kullanıcı "bunu çöz" derse, bir önceki soruyu çöz.
            """
            
            with st.chat_message("assistant", avatar="🧚‍♀️"):
                with st.spinner("Yazıyorum..."):
                    response = model.generate_content(prompt)
                    st.write(response.text)
            
            # 3. Yasmin'in cevabını hafızaya ekle
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Hata oluştu: {e}")
