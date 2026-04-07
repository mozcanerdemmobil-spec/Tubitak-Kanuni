import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Sayfa Ayarları ve Görselleştirme
st.set_page_config(page_title="OkulArkadaşım - TÜBİTAK AI", page_icon="🎓")
st.title("🎓 OkulArkadaşım")
st.markdown("### MEB Yönetmelik Akıllı Bilgi Asistanı")

# 2. API Anahtarını Streamlit Secrets'tan Çekme
# Secrets panelinde GROQ_API_KEY olarak tanımladığın için buradan okuyoruz
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error("⚠️ API Key bulunamadı! Lütfen Streamlit Secrets ayarlarını kontrol edin.")
    st.stop()

# 3. Veri Tabanını (ChromaDB) ve Embeddings'i Yükleme
@st.cache_resource # Uygulama her yenilendiğinde DB'yi tekrar yükleyip yavaşlamasın
def load_vector_db():
    # Colab'da kullandığın modelin aynısını kullanmalısın
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # İndirdiğin 'okul_asistani_db' klasörünün app.py ile aynı yerde olması gerekir
    persist_dir = "./okul_asistani_db"
    
    if not os.path.exists(persist_dir):
        st.error(f"❌ '{persist_dir}' klasörü bulunamadı. Lütfen GitHub'a yüklediğinizden emin olun.")
        return None
        
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db

vector_db = load_vector_db()

# 4. Asistan Sorgu Fonksiyonu (Senin paylaştığın mantıkla güncellendi)
def okul_asistani_sorgula(soru):
    if vector_db is None:
        return "Veri tabanı yüklenemedi."

    # Benzerlik araması (k=6 odaklanmış sonuç için ideal)
    docs = vector_db.similarity_search(soru, k=6)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": """Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzman, teknik ve resmi bir asistansın.
                
                Kritik Kurallar:
                1. SADECE sana verilen 'Bağlam' içindeki bilgileri kullan. 
                2. Eğer cevap bağlamda yoksa, 'Bu konuyla ilgili yönetmelikte net bir bilgi bulamadım' de. ASLA dış dünyadan bildiğin bilgileri ekleme.
                3. Cevaplarını maddeler halinde ve resmi bir dille ver.
                KARAR HİYERARŞİSİ:
                - ÖZÜRSÜZ DEVAMSIZLIK: 10 günü aşan başarısız sayılır.
                - SINIF GEÇME: Yıl sonu başarı puanı en az 50 OLMALI ve en fazla 3 zayıf bulunmalıdır.
                4. Kişisel yorum yapma, veli veya okul müdürü gibi rolleri sınav katılımcılarıyla karıştırma.
                """
            },
            {
                "role": "user",
                "content": f"Aşağıdaki yönetmelik metinlerine dayanarak soruyu yanıtla.\n\nBağlam:\n{baglam}\n\nSoru: {soru}"
            }
        ],
        model="llama-3.1-8b-instant", 
        temperature=0, # Kesin bilgi için 0 tutuyoruz
        max_tokens=1000
    )

    return chat_completion.choices[0].message.content

# 5. Sohbet Arayüzü
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Sınıf geçme şartları nedir?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yönetmelik inceleniyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
