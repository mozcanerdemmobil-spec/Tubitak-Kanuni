import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Okul Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 KANUNİ MTAL Akıllı Asistan")

# --- 2. VERİ YÜKLEME ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Dizini kontrol et: 'okul_asistani_db' olduğundan emin ol
    vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)
    return vector_db

# --- 3. SIDEBAR ---
with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password").strip()
    if not api_key:
        st.stop()

client = Groq(api_key=api_key)
vector_db = load_vector_db()

# --- 4. AKILLI ARAMA VE SORGULAMA ---
def okul_asistani_sorgula(soru):
    # SINIF TESPİTİ (Otomatik: 9-A, 10B, 12ATP gibi her şeyi yakalar)
    # Kullanıcının sorusundaki sınıfı bulup aramayı o sınıfa odaklar.
    sinif_match = re.search(r"(\d{1,2})[-/\s]?([A-Z]{1,3})", soru.upper())
    arama_terimi = soru
    if sinif_match:
        bulunan_sinif = f"{sinif_match.group(1)}-{sinif_match.group(2)}"
        arama_terimi = f"{bulunan_sinif} sınıfı {soru}"

    # k=8 idealdir; hem bilgi verir hem token sınırını (6000) aşmaz.
    docs = vector_db.similarity_search(arama_terimi, k=8)
    baglam = "\n".join([doc.page_content for doc in docs])

    # TOKEN TASARRUFLU SİSTEM MESAJI
    system_prompt = f"""Sen okul asistanısın. SADECE 'Bağlam'ı kullan. 
Kurallar:
1- Bilgi yoksa 'Programda net bilgi bulamadım' de.
2- Sınıfları (Örn: 9A ile 9B) asla karıştırma.
3- Cevaplar madde madde ve resmi olsun.
4- Saatler: 1.(08:20), 2.(09:00), 3.(09:50), 4.(10:30), 5.(11:20), 6.(12:00), 7.(13:25), 8.(14:05), 9.(14:55), 10.(15:35).

Bağlam:
{baglam}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": soru}
            ],
            model="llama-3.1-8b-instant", # Token dostu model
            temperature=0,
            max_tokens=500 # Cevabı kısa tutar, hata vermez
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower() or "token" in str(e).lower():
            return "⚠️ Çok fazla veri çekildi (Token sınırı). Lütfen daha spesifik bir soru sorun."
        return f"Hata: {str(e)}"

# --- 5. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sınıfını ve sorunu yaz (Örn: 10-A pazartesi 3. ders)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
