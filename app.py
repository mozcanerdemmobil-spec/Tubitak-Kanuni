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

# ====================== YENİ SINIF TESPİT FONKSİYONU ======================
def sinif_filtresi_bul(prompt: str):
    """Prompt'tan sınıfı tespit eder ve olası tüm formatlarda dener"""
    # Tüm olası sınıf kodlarını yakala (9a, 9-A, 9A/BL, 10d hb, 9atp vs.)
    pattern = re.compile(r'\b(9[0-9a-z]?|10[0-9a-z]?|11[0-9a-z]?|12[0-9a-z]?)[ -/]?[a-z]*(?:bl|el|en|hb|atp)?\b', re.IGNORECASE)
    match = pattern.search(prompt)
    if not match:
        return None
    
    raw = match.group(0).strip().upper()
    
    # Olası formatlar (veritabanında hangisi varsa biri tutsun)
    candidates = [
        raw,                    # 9-A/BL
        raw.replace(" ", ""),   # 9A/BL
        re.sub(r'[-/]?BL', '/BL', raw),
        re.sub(r'[-/]?EL', '/EL', raw),
        re.sub(r'[-/]?EN', '/EN', raw),
        re.sub(r'[-/]?HB', '/HB', raw),
        re.sub(r'[-/]?ATP', '/ATP', raw),
        raw.replace("-", ""),   # 9A/BL
        raw.replace("/", ""),   # 9ABL
    ]
    
    # Tekrarları temizle
    candidates = list(dict.fromkeys(candidates))
    return candidates  # Birden fazla format deneriz

# --- 4. SORGULAMA FONKSİYONU ---
def okul_asistani_sorgula(soru: str):
    sinif_candidates = sinif_filtresi_bul(soru)
    
    print(f"🔍 Sorgu: {soru}")
    if sinif_candidates:
        print(f"📌 Tespit edilen sınıf adayları: {sinif_candidates}")
    
    # Önce filtreli arama deneriz
    docs = []
    if sinif_candidates:
        for candidate in sinif_candidates:
            try:
                search_kwargs = {"filter": {"sinif": candidate}}
                docs = vector_db.similarity_search(soru, k=6, **search_kwargs)
                if docs:
                    print(f"✅ Filtre başarılı! Kullanılan sınıf: {candidate} ({len(docs)} sonuç)")
                    break
            except Exception as e:
                print(f"Filtre hatası ({candidate}): {e}")
    
    # Eğer filtreyle hiç sonuç yoksa genel arama yap
    if not docs:
        print("⚠️ Filtreyle sonuç bulunamadı, genel arama yapılıyor...")
        docs = vector_db.similarity_search(soru, k=5)
    
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen KANUNİ MTAL okulunun MEB Ortaöğretim Kurumları Yönetmeliği ve ders programı konusunda uzmansın.
Kritik Kurallar:
1. SADECE aşağıda verilen 'Bağlam' içindeki bilgileri kullan.
2. Eğer bağlamda ilgili sınıfın programı yoksa kesinlikle 'Programda net bilgi bulamadım' de.
3. Cevaplar kısa, net ve maddeler halinde olsun.
4. Ders saatleri her zaman aynıdır:
   -1. Ders: 08:20-09:00   -2. Ders: 09:00-09:40   vs...

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
        return f"Hata oluştu: {str(e)}"

# --- 5. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu yazın... (Örn: 9-A/BL pazartesi 1. ders nedir?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sınıf programı taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
