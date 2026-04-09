import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# ====================== SAYFA AYARLARI ======================
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

# ====================== SINIF TESPİT (Colab ile aynı mantık) ======================
def sinif_filtresi_bul(prompt: str):
    # Colab'da çalışan formatı yakalamaya çalışıyoruz: 9-A/BL, 9A, 9-A, 10D/HB vb.
    pattern = re.compile(r'\b(9[ -/]?[A-Ia-i]?|10[ -/]?[A-Ea-e]?|11[ -/]?[A-Da-d]?|12[ -/]?[A-Ga-g]?)(?:[ -/]?(?:BL|EL|EN|HB|ATP))?\b', re.IGNORECASE)
    match = pattern.search(prompt)
    if not match:
        return None
    
    sinif = match.group(0).strip().upper()
    
    # En çok çalışan formatlara dönüştür (Colab'da '9-A/BL' çalışıyor)
    sinif = re.sub(r'([0-9]+)\s*([A-Z])', r'\1-\2', sinif)   # 9A → 9-A
    sinif = re.sub(r'[-/ ]?BL', '/BL', sinif, flags=re.I)
    sinif = re.sub(r'[-/ ]?EL', '/EL', sinif, flags=re.I)
    sinif = re.sub(r'[-/ ]?EN', '/EN', sinif, flags=re.I)
    sinif = re.sub(r'[-/ ]?HB', '/HB', sinif, flags=re.I)
    sinif = re.sub(r'[-/ ]?ATP', '/ATP', sinif, flags=re.I)
    
    return sinif

# ====================== ANA FONKSİYON ======================
def okul_asistani_sorgula(soru: str):
    sinif_filtresi = sinif_filtresi_bul(soru)
    
    st.info(f"**Sorgu:** {soru}")   # debug için
    
    if sinif_filtresi:
        st.success(f"📌 Tespit edilen sınıf: **{sinif_filtresi}** → Filtre uygulanıyor")
    
    search_kwargs = {}
    if sinif_filtresi:
        search_kwargs = {"filter": {"sinif": sinif_filtresi}}

    # Arama yap
    docs = vector_db.similarity_search(soru, k=5, **search_kwargs)

    if not docs and sinif_filtresi:
        st.warning(f"Filtreyle sonuç bulunamadı ({sinif_filtresi}). Genel arama yapılıyor...")
        docs = vector_db.similarity_search(soru, k=5)

    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen KANUNİ MTAL Okulu ders programı uzmanısın.
Kural: SADECE verilen Bağlam'daki bilgileri kullan.
Eğer ilgili sınıf ve gün için net bilgi yoksa kesinlikle "Programda net bilgi bulamadım" de.
Cevaplarını kısa ve net ver.

Bağlam:
{baglam}
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": soru}
            ],
            model="llama-3.1-8b-instant",
            temperature=0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Hata: {str(e)}"

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ders programı aranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
