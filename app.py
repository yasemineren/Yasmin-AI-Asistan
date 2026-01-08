import streamlit as st
import google.generativeai as genai

st.title("🔍 Model Dedektifi")
st.write("Hata mesajındaki tavsiyeye uyuyoruz: ListModels çalıştırılıyor...")

api_key = st.text_input("Google API Anahtarını Gir:", type="password")

if st.button("Modelleri Listele"):
    if not api_key:
        st.error("Lütfen anahtarı gir.")
    else:
        try:
            genai.configure(api_key=api_key)
            st.write("### ✅ Senin Hesabında Açık Olan Modeller:")
            
            # İşte hatanın bizden yapmamızı istediği o sihirli komut:
            bulunanlar = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    st.code(f"{m.name}")
                    bulunanlar.append(m.name)
            
            if not bulunanlar:
                st.warning("Hiçbir model bulunamadı! Anahtarda bir kısıtlama olabilir.")
                
        except Exception as e:
            st.error(f"Hata oluştu: {e}")


