import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 MEB Ortaöğretim Yönetmelik Asistanı")

# --- 2. VERİ YÜKLEME ---

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./okul_asistani_db_yeni", embedding_function=embeddings)
    return vector_db

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()

    if not api_key:
        st.warning("Lütfen devam etmek için API Key giriniz.")
        st.stop()

# Nesneleri başlat
client = Groq(api_key=api_key)
vector_db = load_vector_db()

# --- 4. SORGULAMA ---
def okul_asistani_sorgula(soru):
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzmansın.
Kritik Kurallar:
1. SADECE 'Bağlam' içindeki bilgileri kullan.
2. Cevap yoksa 'Programda net bilgi bulamadım' de.
3. Cevaplar maddeler halinde ve resmi olsun.
4. "Evet" veya "Hayır" ile başla (uygunsa).
5. Ders saatleri geneldir yani hep aynıdır saatler şöyle: 
    -1.Ders: 08:20 - 09:00
    -2.Ders: 09:00 - 09:40
    -3.Ders: 09:50 - 10:30
    -4.Ders: 10:30 - 11:10
    -5.Ders: 11:20 - 12:00
    -6.Ders: 12:00 - 12:40
    -7.Ders: 13:25 - 14:05
    -8.Ders: 14:05 - 14:45
    -9.Ders: 14:55 - 15:35
    -10.Ders: 15:35 - 16:25


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

# --- 5. CHAT ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajlar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Yeni mesaj
if prompt := st.chat_input("Sorunuzu yazın..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yönetmelik taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
