import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yasmin AI", page_icon="🎓", layout="wide")

# --- TASARIM (CSS) ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(to right, #141e30, #243b55); color: white; }
    h1 { text-align: center; color: #FFD700; text-shadow: 0 0 10px #FFD700; }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 10px; width: 100%; }
    .stChatMessage { background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎓 Yasmin: Akademik Asistan</h1>", unsafe_allow_html=True)

# --- SOHBET GEÇMİŞİNİ BAŞLAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- YAN MENÜ (AYARLAR) ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    # 1. API Anahtarı (Önce Secrets'a bakar, yoksa kutu açar)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Anahtar Sistemden Çekildi")
    else:
        api_key = st.text_input("Google API Anahtarını Gir:", type="password")
        st.markdown("[🔑 Anahtar Al](https://aistudio.google.com/app/apikey)", unsafe_allow_html=True)
    
    # 2. Model Seçimi (Hatanın Çözümü Burada: Değişkeni kesin olarak tanımlıyoruz)
    secilen_model = st.selectbox(
        "Zeka Modeli Seç:",
        ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"),
        index=0
    )
    
    # 3. Dosya Yükleme
    uploaded_file = st.file_uploader("Ders Notu / Makale (PDF)", type="pdf")
    
    # 4. Eğit Butonu
    if st.button("🧠 Dokümanı Analiz Et"):
        if not api_key or not uploaded_file:
            st.error("Lütfen API anahtarı ve dosya eksik olmasın.")
        else:
            with st.spinner("Doküman akademik seviyede inceleniyor..."):
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
                    st.success("Analiz Tamamlandı. Yasmin sorularını bekliyor.")
                    st.session_state.messages = [] # Yeni dosya gelince hafızayı temizle
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")

# --- SOHBET EKRANI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- KULLANICI GİRİŞİ ---
soru = st.chat_input("Akademik sorunuzu buraya yazın...")

if soru:
    if "vs" not in st.session_state:
        st.warning("Lütfen önce sol menüden PDF yükleyip analiz işlemini başlatın.")
    else:
        # Mesajı ekle
        st.session_state.messages.append({"role": "user", "content": soru})
        with st.chat_message("user"):
            st.markdown(soru)

        # Cevap Üret
        try:
            docs = st.session_state.vs.similarity_search(soru, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            gecmis_sohbet = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            
            # Seçilen Modeli Tanımla (Hata burada çözülüyor)
            model = genai.GenerativeModel(secilen_model)
            
            # --- PROFESÖR PROMPT (SENİN İSTEDİĞİN SERT VERSİYON) ---
            prompt = f"""
            ### ROL TANIMI
            Sen Yasmin; fizik, matematik, hukuk ve mühendislik başta olmak üzere tüm akademik disiplinlerde **kıdemli profesör** seviyesinde bilgiye sahip, otoriter ve teknik bir asistansın.
            Amacın, öğrencinin sorularını en derin, teknik ve doğru şekilde yanıtlamaktır.

            ### KATI KURALLAR
            1. **Üslup:** "Sıfır Nezaket, %100 Bilgi". Asla "Merhaba", "Rica ederim" kullanma. Doğrudan teknik cevaba gir.
            2. **Matematik:** Tüm formülleri LaTeX formatında yaz (Örn: $E=mc^2$).
            3. **Analiz:** Cevabı vermeden önce bağlamı mantıksal olarak analiz et.
            4. **Bağlam:** Cevabını öncelikle "PDF BİLGİSİ"ne dayandır. Yetersizse akademik bilgini kullan.
            
            ### VERİLER
            **Geçmiş:** {gecmis_sohbet}
            **PDF Bağlamı:** {context}
            **Soru:** {soru}
            
            **YANITIN:**
            """
            
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Analiz ediliyor..."):
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Hata: {e}")

