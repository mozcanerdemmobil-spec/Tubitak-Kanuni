import streamlit as st
import pandas as pd
import requests
import base64
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 MEB Ortaöğretim Yönetmelik & Ders Programı Asistanı")

# --- 2. VERİ YÜKLEME FONKSİYONLARI ---

@st.cache_resource
def load_vector_db():
    """Yönetmelik veritabanını yükler."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)
    return vector_db

@st.cache_data
def program_yukle(istenen_sinif):
    url = "https://raw.githubusercontent.com/mozcanerdemmobil-spec/Tubitak_llama/main/programlar.xlsx"
    try:
        # Önce Excel dosyasını komple yükleyelim (sayfa isimlerini kontrol etmek için)
        excel_file = pd.ExcelFile(url, engine='openpyxl')
        tum_sayfalar = excel_file.sheet_names # Excel'deki tüm sayfa isimlerini alır
        
        # Kullanıcının aradığı sınıfı, Excel'deki sayfalarla karşılaştıralım
        # (Hem kullanıcı girdisini hem sayfa isimlerini küçük harfe çevirip boşlukları siliyoruz)
        hedef_sayfa = None
        for sayfa in tum_sayfalar:
            if sayfa.lower().strip() == istenen_sinif.lower().strip():
                hedef_sayfa = sayfa
                break
        
        if hedef_sayfa:
            # Eşleşen sayfayı oku
            df = pd.read_excel(url, sheet_name=hedef_sayfa, engine='openpyxl')
            return df
        else:
            # Debug için: Hangi sayfalar var ekrana yazdıralım (Hata çözülünce bu satırı silebilirsin)
            st.warning(f"Excel içindeki gerçek sayfalar: {tum_sayfalar}")
            return None
            
    except Exception as e:
        st.error(f"Excel okuma hatası: {e}")
        return None

# --- 3. SIDEBAR (AYARLAR) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()
    
    

    if not api_key:
        st.warning("Lütfen devam etmek için API Key giriniz.")
        st.stop()

# Gerekli nesneleri başlatıyoruz
client = Groq(api_key=api_key)
vector_db = load_vector_db()

# --- 4. SORGULAMA FONKSİYONU (LLM) ---
def okul_asistani_sorgula(soru):
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzmansın.
    Kritik Kurallar:
    1. SADECE 'Bağlam' içindeki bilgileri kullan.
    2. Cevap yoksa 'Yönetmelikte net bilgi bulamadım' de.
    3. Cevaplar maddeler halinde ve resmi olsun.
    4. "Evet" veya "Hayır" ile başla (uygunsa).
    5. Cevap formatı:
       Cevap:
       - ...

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

# --- 5. CHAT ARAYÜZÜ VE MANTIK ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Yeni girdi kontrolü
if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: 12a programı nedir?)..."):
    
    # Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # DERS PROGRAMI YAKALAMA (INTERCEPT)
    kucuk_prompt = prompt.lower()
    istenen_sinif = None
    sinif_listesi = ["9a", "10a", "11a", "12a"] # Excel sheet isimlerinle aynı olmalı
    
    # Eğer mesajda "program" veya "ders" geçiyorsa sınıfı ara
    if "program" in kucuk_prompt or "ders" in kucuk_prompt:
        for s in sinif_listesi:
            if s in kucuk_prompt:
                istenen_sinif = s
                break

    # EĞER DERS PROGRAMI SORULDUYSA
    if istenen_sinif:
        with st.chat_message("assistant"):
            with st.spinner(f"{istenen_sinif.upper()} programı yükleniyor..."):
                df_program = program_yukle(istenen_sinif)
                
                if df_program is not None:
                    cevap_metni = f"İşte **{istenen_sinif.upper()}** sınıfının haftalık ders programı:"
                    st.write(cevap_metni)
                    st.table(df_program) # Tabloyu basar
                    st.session_state.messages.append({"role": "assistant", "content": cevap_metni})
                else:
                    hata_mesaji = f"Üzgünüm, Excel dosyasında '{istenen_sinif}' isimli bir sayfa bulamadım."
                    st.error(hata_mesaji)
                    st.session_state.messages.append({"role": "assistant", "content": hata_mesaji})
    
    # NORMAL YÖNETMELİK SORUSU (Program sorulmadıysa)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Yönetmelik taranıyor..."):
                cevap = okul_asistani_sorgula(prompt)
                st.write(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
