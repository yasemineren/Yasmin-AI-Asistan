import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yasmin AI", page_icon="🧚‍♀️", layout="wide")

# --- ÖZEL TASARIM (CSS) ---
st.markdown("""
<style>
    /* 1. Arka Plan Rengi (Gradient - Mor/Mavi Geçişli) */
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    
    /* 2. Başlık Stili (Ortalı ve Parlayan Efekt) */
    h1 {
        text-align: center;
        color: #FFD700;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #FF4500;
    }
    
    /* 3. Buton Stili */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    
    /* 4. Sohbet Baloncukları İçin Hazırlık */
    .user-msg {
        background-color: #2b5876;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: right;
    }
    .bot-msg {
        background-color: #4e4376;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
st.markdown("<h1>✨ Yasmin: Kişisel Asistanın ✨</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Senin için notları okurum, özetlerim ve sorularını yanıtlarım.</p>", unsafe_allow_html=True)

# --- YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Google API Anahtarını Gir:", type="password")
    uploaded_file = st.file_uploader("Bir PDF Dosyası Yükle", type="pdf")
    process_button = st.button("🧠 Yasmin'i Eğit")
    
    st.markdown("---")
    st.info("💡 İpucu: Notların ne kadar uzun olursa olsun, saniyeler içinde öğrenirim.")

# --- ANA İŞLEM (MOTOR KISMI) ---
if process_button and api_key and uploaded_file:
    with st.spinner("Dosyayı inceliyorum, lütfen bekle... 🧚‍♀️"):
        try:
            # 1. API Bağlantısı
            genai.configure(api_key=api_key)
            
            # 2. Dosyayı Kaydet
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 3. Okuma
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            
            # 4. Embeddings (Hızlı Model)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )
            
            # 5. Hafıza
            vector_store = FAISS.from_documents(pages, embeddings)
            st.session_state.vs = vector_store
            
            # --- HAVAİ FİŞEK EFEKTİ ---
            st.balloons() 
            st.success("Harika! Tüm notları hafızama kaydettim. Hazırım! 🎉")
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

# --- SOHBET ARAYÜZÜ ---
soru = st.text_input("Merak ettiğin soruyu buraya yaz:")

if soru and "vs" in st.session_state:
    try:
        # Alakalı yerleri bul
        docs = st.session_state.vs.similarity_search(soru, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # CEVAP ÜRETME (Çalışan Model: gemini-2.5-flash)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Sen Yasmin adında çok yardımsever, nazik ve zeki bir asistansın.
        Öğrenciye derslerinde yardımcı oluyorsun.
        Cevap verirken samimi bir dil kullan.
        
        Bağlam:
        {context}
        
        Soru: {soru}
        """
        
        with st.chat_message("user"):
            st.write(soru)
            
        with st.spinner("Yasmin düşünüyor..."):
            response = model.generate_content(prompt)
            
        with st.chat_message("assistant", avatar="🧚‍♀️"):
            st.write(response.text)
        
    except Exception as e:
        st.error(f"Cevap veremedim: {e}")
