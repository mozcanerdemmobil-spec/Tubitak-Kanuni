import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

st.set_page_config(page_title="Okul Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 KANUNİ MTAL Okul Asistanı")

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./okul_asistanı_db_kanuni", embedding_function=embeddings)
    return vector_db

with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()
    if not api_key:
        st.warning("Lütfen API Key giriniz.")
        st.stop()

client = Groq(api_key=api_key)
vector_db = load_vector_db()

# ====================== GELİŞTİRİLMİŞ SINIF TESPİT ======================
def sinif_filtresi_bul(prompt: str):
    pattern = re.compile(r'\b(9[ -/]?[A-Ia-i]?|10[ -/]?[A-Ea-e]?|11[ -/]?[A-Da-d]?|12[ -/]?[A-Ga-g]?)(?:[ -/]?(?:BL|EL|EN|HB|ATP))?\b', re.IGNORECASE)
    match = pattern.search(prompt)
    if not match:
        return None, None
    
    raw = match.group(0).strip().upper()
    
    # 1. Temizlenmiş tam format (en çok çalışan)
    sinif_tam = re.sub(r'([0-9]+)\s*([A-Z])', r'\1-\2', raw)
    sinif_tam = re.sub(r'[-/ ]?BL', '/BL', sinif_tam, flags=re.I)
    sinif_tam = re.sub(r'[-/ ]?EL', '/EL', sinif_tam, flags=re.I)
    sinif_tam = re.sub(r'[-/ ]?EN', '/EN', sinif_tam, flags=re.I)
    sinif_tam = re.sub(r'[-/ ]?HB', '/HB', sinif_tam, flags=re.I)
    sinif_tam = re.sub(r'[-/ ]?ATP', '/ATP', sinif_tam, flags=re.I)
    
    # 2. Basit format (sadece 9-A gibi)
    sinif_basit = re.sub(r'/.*', '', sinif_tam)   # /BL kısmını at → 9-A
    
    return sinif_tam, sinif_basit

# ====================== ANA FONKSİYON ======================
def okul_asistani_sorgula(soru: str):
    sinif_tam, sinif_basit = sinif_filtresi_bul(soru)
    
    st.info(f"**Sorgu:** {soru}")
    if sinif_tam:
        st.success(f"📌 Tespit edildi → Tam: **{sinif_tam}** | Basit: **{sinif_basit}**")

    docs = []
    used = None

    # Önce tam formatla dene (9-A/BL)
    if sinif_tam:
        try:
            docs = vector_db.similarity_search(soru, k=5, filter={"sinif": sinif_tam})
            if docs:
                used = sinif_tam
                st.success(f"✅ Tam filtre çalıştı: **{sinif_tam}** ({len(docs)} sonuç)")
        except:
            pass

    # Tam format çalışmazsa basit formatla dene (9-A)
    if not docs and sinif_basit and sinif_basit != sinif_tam:
        try:
            docs = vector_db.similarity_search(soru, k=5, filter={"sinif": sinif_basit})
            if docs:
                used = sinif_basit
                st.success(f"✅ Basit filtre çalıştı: **{sinif_basit}**")
        except:
            pass

    # Hala sonuç yoksa genel arama
    if not docs:
        st.warning("⚠️ Filtreyle sonuç yok → Genel arama yapılıyor")
        docs = vector_db.similarity_search(soru, k=5)

    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen KANUNİ MTAL Okulu ders programı uzmanısın.
SADECE Bağlam'daki bilgileri kullan.
Bağlamda o sınıf ve gün için bilgi yoksa "Programda net bilgi bulamadım" de.
Cevap kısa ve net olsun.

Bağlam:
{baglam}
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": soru}
            ],
            model="llama-3.1-8b-instant",
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata: {str(e)}"

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın... (Örn: 9-A/BL Salı 3. ders nedir?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ders programı aranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("Debug bilgilerini okuyarak ilerliyoruz.")
