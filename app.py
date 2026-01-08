import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
# EKSİK OLAN SATIR BURASI:
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
import os

# --- AYARLAR ---
cwd = os.getcwd()
cache = os.path.join(cwd, 'hf_cache')
os.environ['HF_HOME'] = cache
os.environ['TORCH_HOME'] = cache

# --- KÜTÜPHANELER ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Yasmin AI", page_icon="🎆", layout="wide")

# --- YAN MENÜ (GİRİŞ) ---
with st.sidebar:
    st.header("🔑 Giriş")
    kullanici_api_key = st.text_input(
        "Google API Key", 
        type="password", 
        help="Buraya kendi Gemini API anahtarınızı yapıştırın."
    )
    
    st.divider()
    
    st.header("🎨 Tasarım")
    renk1 = st.color_picker("Tema", "#d63384") 
    renk2 = st.color_picker("Arka Plan", "#0e1117")
    renk3 = st.color_picker("Yazı", "#fafafa")
    st.divider()

# --- CSS ---
s = "<style>"
s += f".stApp {{background-color: {renk2} !important;}} "
s += f"section[data-testid='stSidebar'] {{background-color: {renk2};}} "
s += f"h1,h2,p,li,label {{color: {renk3} !important;}} "
s += f"div.stButton>button {{background-color: {renk1} !important;}} "
s += "div.stButton>button {color: white !important; border: none;} "
s += "header {background: transparent !important;} "
s += "</style>"
st.markdown(s, unsafe_allow_html=True)

# --- BAŞLIK (GARANTİ LİNK) ---
# Bu link Giphy'nin resmi CDN sunucusudur, kolay kolay bozulmaz.
img = "https://media.giphy.com/media/peAFQfg7Ol6IE/giphy.gif"

h = ""
h += "<div style='display:flex; justify-content:center; align-items:center;'>"
h += f"<img src='{img}' style='width:120px; mix-blend-mode: screen;'>"
h += f"<h1 style='color:{renk1}; margin:0 20px; text-align:center;'>Yasmin: Asistanın</h1>"
h += f"<img src='{img}' style='width:120px; mix-blend-mode: screen; transform:scaleX(-1);'>"
h += "</div>"
st.markdown(h, unsafe_allow_html=True)

st.write("Merhaba! PDF'lerinle konuşmaya hazırım. Lütfen önce sol taraftan API anahtarını gir. 🧠")
st.divider()

# --- ANAHTAR KONTROLÜ ---
if not kullanici_api_key:
    st.warning("⚠️ Lütfen sol menüden Google API Anahtarınızı giriniz.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = kullanici_api_key

# --- YÜKLEME ---
with st.sidebar:
    st.header("📂 Yükle")
    dosyalar = st.file_uploader("PDF Seç", accept_multiple_files=True, type="pdf")
    buton = st.button("🧠 Öğren")

# --- FONKSİYONLAR ---
def kur_database(gelen_dosyalar):
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    docs = []
    for d in gelen_dosyalar:
        yol = os.path.join("temp", d.name)
        with open(yol, "wb") as f:
            f.write(d.getbuffer())
        loader = PyPDFLoader(yol)
        docs.extend(loader.load())
    
    parcalayici = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parcalar = parcalayici.split_documents(docs)
    
    # Google'ın kendi embedding motorunu kullanıyoruz (Hata vermez)
    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
    
    db = FAISS.from_documents(parcalar, embed)
    db.save_local("faiss_db")
    return db

def getir_database():
    embed = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        cache_folder="./hf_cache"
    )
    try:
        return FAISS.load_local("faiss_db", embed, allow_dangerous_deserialization=True)
    except:
        return None

# --- ANA PROGRAM ---
if buton and dosyalar:
    with st.spinner("İşleniyor..."):
        try:
            kur_database(dosyalar)
            st.sidebar.success("Hazır!")
            st.balloons()
        except Exception as e:
            st.error(f"Hata: {e}")

if "msg" not in st.session_state:
    st.session_state.msg = []

for m in st.session_state.msg:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

soru = st.chat_input("Sorunu yaz...")
if soru:
    st.session_state.msg.append({"role": "user", "content": soru})
    with st.chat_message("user"):
        st.markdown(soru)

    db = getir_database()
    if db:
        with st.chat_message("assistant"):
            with st.spinner("..."):
                bulunan = db.similarity_search(soru, k=4)
                icerik = "\n".join([b.page_content for b in bulunan])
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=kullanici_api_key
                )
                
                sablon = "Bilgi: {c}\nSoru: {q}"
                prompt = ChatPromptTemplate.from_template(sablon)
                zincir = prompt | llm
                
                cevap = zincir.invoke({"c": icerik, "q": soru})
                st.markdown(cevap.content)
                st.session_state.msg.append({"role": "assistant", "content": cevap.content})
    else:
        st.error("Önce dosya yükle!")


