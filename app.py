import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Okul Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 KANUNİ MTAL Okul Asistanı")

# --- 2. VERİ YÜKLEME ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./okul_asistanı_db_kanuni", embedding_function=embeddings)
    return vector_db

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()
    if not api_key:
        st.warning("Lütfen devam etmek için API Key giriniz.")
        st.stop()

client = Groq(api_key=api_key)
vector_db = load_vector_db()

# ====================== SINIF FİLTRESİ TESPİTİ ======================
# Tüm olası sınıf varyasyonlarını yakalayan regex
sinif_pattern = re.compile(
    r'\b(9[abcdefghi]?|10[abcde]?|11[abcd]?|12[abfg]?)(?:[ -/]?(?:bl|el|en|hb|atp))?\b',
    re.IGNORECASE
)

def sinif_filtresi_bul(prompt: str):
    """Prompt'tan sınıf kodunu tespit eder ve veritabanı için temiz format döndürür"""
    bulunan = sinif_pattern.search(prompt)
    if not bulunan:
        return None
    
    sinif = bulunan.group(0).strip().upper()
    
    # Temizleme: 9A BL → 9-A/BL , 9atp → 9-ATP , 10d hb → 10-D/HB gibi
    sinif = re.sub(r'([0-9]+)([A-Z])', r'\1-\2', sinif)      # 9A → 9-A
    sinif = re.sub(r'\s+', '', sinif)                        # boşlukları kaldır
    sinif = re.sub(r'[-/]?BL', '/BL', sinif, flags=re.I)
    sinif = re.sub(r'[-/]?EL', '/EL', sinif, flags=re.I)
    sinif = re.sub(r'[-/]?EN', '/EN', sinif, flags=re.I)
    sinif = re.sub(r'[-/]?HB', '/HB', sinif, flags=re.I)
    sinif = re.sub(r'[-/]?ATP', '/ATP', sinif, flags=re.I)  # veya -ATP
    
    return sinif

# --- 4. SORGULAMA FONKSİYONU (Yeni Mantık) ---
def okul_asistani_sorgula(soru: str):
    # Sınıf filtresi var mı diye kontrol et
    sinif_filtresi = sinif_filtresi_bul(soru)
    
    print(f"🔍 Sorgu: {soru}")                    # debug için (istediğinde kaldırabilirsin)
    if sinif_filtresi:
        print(f"📌 Tespit edilen sınıf filtresi: {sinif_filtresi}")
    
    # Filtre parametresi hazırla
    search_kwargs = {}
    if sinif_filtresi:
        search_kwargs = {"filter": {"sinif": sinif_filtresi}}
    
    # Vektör DB'de arama (filtre uygulanmış haliyle)
    docs = vector_db.similarity_search(soru, k=5, **search_kwargs)
    
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzmansın.
Kritik Kurallar:
1. SADECE 'Bağlam' içindeki bilgileri kullan.
2. Cevap yoksa 'Programda net bilgi bulamadım' de.
3. Cevaplar maddeler halinde ve resmi olsun.
4. "Evet" veya "Hayır" ile başla (uygunsa).
5. Ders saatleri geneldir...

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

# --- 5. CHAT ARAYÜZÜ ---
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
        with st.spinner("Yönetmelik ve sınıf bilgileri taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
