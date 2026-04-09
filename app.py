import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Okul Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 KANUNİ MTAL Ders Programı Asistanı")

# --- 2. VERİ YÜKLEME ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Kendi dizin adına göre burayı kontrol et (okul_asistani_db veya okul_asistanı_db_kanuni)
    vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)
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
    # KULLANICI SORUSUNU FİLTRELEMEYE UYGUN HALE GETİRME (NORMALİZASYON)
    # Kullanıcı "9A" yazsa bile veritabanında daha net bulması için zenginleştiriyoruz.
    gelistirilmis_soru = soru
    sorgu_buyuk = soru.upper().replace(" ", "")
    
    if "9A" in sorgu_buyuk or "9-A" in sorgu_buyuk:
        gelistirilmis_soru += " (9-A/BL sınıfı)"
    elif "11C" in sorgu_buyuk or "11-C" in sorgu_buyuk:
        gelistirilmis_soru += " (11C/EN sınıfı)"
    elif "12ATP" in sorgu_buyuk or "12/ATP" in sorgu_buyuk:
        gelistirilmis_soru += " (12/ATP sınıfı)"

    # K değerini 5'ten 15'e çıkardık ki benzer isimli sınıflar yüzünden asıl aradığımız kaynamasın.
    docs = vector_db.similarity_search(gelistirilmis_soru, k=15)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # YENİ VE ÇOK KATI SİSTEM PROMPTU
    system_prompt = f"""Sen Kanuni MTAL'nin resmi ders programı asistanısın. Görevin, sana 'Bağlam' olarak verilen verileri kullanarak soruları %100 doğrulukla yanıtlamaktır.

KRİTİK KURALLAR:
1. SADECE 'Bağlam' içindeki bilgileri kullan. Kendi hafızandan bilgi uydurma.
2. SINIF İSİMLERİNDE AKILLI EŞLEŞTİRME: Kullanıcı "9A", "11C" gibi eksik isimler yazabilir. Bağlamdaki uzantılı hallerini (9-A/BL, 11C/EN vb.) aranan sınıf olarak kabul et.
3. Bağlamda birden fazla sınıfın bilgisi gelirse, SADECE kullanıcının sorduğu sınıfa ait olanları al, diğer sınıfların bilgisini kesinlikle yoksay (Örn: 11-C sorulmuşsa 11-B'yi görmezden gel).
4. Eğer sorulan gün, saat veya sınıf bağlamda HİÇ YOKSA tahmin yürütme. Sadece şunu söyle: 'Programda net bilgi bulamadım.'
5. Cevapları maddeler halinde, resmi bir dille oluştur. Uygunsa "Evet" veya "Hayır" ile başla.
6. Bir saat sorulduğunda şu standart tabloyu referans al:
   - 1.Ders: 08:20 - 09:00 | 2.Ders: 09:00 - 09:40 | 3.Ders: 09:50 - 10:30
   - 4.Ders: 10:30 - 11:10 | 5.Ders: 11:20 - 12:00 | 6.Ders: 12:00 - 12:40
   - 7.Ders: 13:25 - 14:05 | 8.Ders: 14:05 - 14:45 | 9.Ders: 14:55 - 15:35 | 10.Ders: 15:35 - 16:25

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

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Yeni mesaj döngüsü
if prompt := st.chat_input("Ders programı ile ilgili sorunuzu yazın... (Örn: 9A Salı ilk ders ne?)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ders programı taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
