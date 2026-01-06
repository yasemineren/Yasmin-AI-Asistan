import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Sayfa Ayarları
st.set_page_config(page_title="Yasmin: Asistanın", layout="wide")

st.title("🤖 Yasmin: Kişisel PDF Asistanın")
st.write("Ders notlarını yükle, bana sorular sor!")

# Yan Menü (API Anahtarı)
with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("Google API Anahtarını Gir:", type="password")
    st.markdown("[Anahtar almak için tıkla](https://aistudio.google.com/app/apikey)")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Bir PDF Yükle", type="pdf")

# Ana İşlem Butonu
if st.sidebar.button("Öğren ve Hazırlan"):
    if not api_key:
        st.error("Lütfen önce API anahtarını gir!")
    elif not uploaded_file:
        st.error("Lütfen bir PDF dosyası yükle!")
    else:
        with st.spinner("Dosya okunuyor ve yapay zeka eğitiliyor..."):
            try:
                # 1. Dosyayı Kaydet
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. PDF'i Yükle
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()

                # 3. Parçalara Böl
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)

                # 4. Vektör Veritabanı (Embeddings)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectors = FAISS.from_documents(final_documents, embeddings)
                
                # Hafızaya al
                st.session_state.vectors = vectors
                st.success("Hazırım! İstediğini sorabilirsin.")
            
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

# Soru Sorma Kısmı
soru = st.text_input("Sorunu yaz:")

if soru:
    if "vectors" in st.session_state:
        # LLM Modeli (En Güncel Sürüm)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        # Prompt Şablonu
        prompt = ChatPromptTemplate.from_template(
            """
            Aşağıdaki bağlama göre soruyu cevapla.
            Eğer cevap bağlamda yoksa "Bilmiyorum" de.
            
            <context>
            {context}
            </context>

            Soru: {input}
            """
        )

        # Zinciri Oluştur
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Cevabı Al
        with st.spinner("Düşünüyorum..."):
            try:
                response = retrieval_chain.invoke({"input": soru})
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Cevap üretirken hata: {e}")
    else:
        st.warning("Lütfen önce sol menüden PDF yükleyip 'Öğren' butonuna bas.")
