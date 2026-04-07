import streamlit as st
import pandas as pd
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
    """Ders programını Excel'den yükler."""
    url = "https://raw.githubusercontent.com/mozcanerdemmobil-spec/Tubitak_llama/main/programlar.xlsx"
    try:
        excel_file = pd.ExcelFile(url, engine='openpyxl')
        tum_sayfalar = excel_file.sheet_names
        
        hedef_sayfa = next((s for s in tum_sayfalar if s.lower().strip() == istenen_sinif.lower().strip()), None)
        if hedef_sayfa:
            df = pd.read_excel(url, sheet_name=hedef_sayfa, engine='openpyxl')
            return df
        else:
            st.warning(f"Excel içindeki sayfalar: {tum_sayfalar}")
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

# --- 4. NESNELER ---
client = Groq(api_key=api_key)
vector_db = load_vector_db()

# --- 5. YÖNETMELİK SORGULAMA ---
def okul_asistani_sorgula(soru):
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzmansın.
Kritik Kurallar:
1. SADECE 'Bağlam' içindeki bilgileri kullan.
2. Cevap yoksa 'Yönetmelikte net bilgi bulamadım' de.
3. Cevaplar maddeler halinde ve resmi olsun.
4. Genel Bilgiler:
    -Ders Saati Süresi Okulda 40 dakika , işletmelerde 60 dakikadır
    -Ders Yılı Tanımı Derslerin başladığı tarihten kesildiği tarihe kadar olan süredir.
    -Eğitim ve Öğretim Yılı Ders yılının başladığı tarihten ertesi ders yılının başladığı tarihe kadar olan süredir.
5. Kayıt ve Geçiş:
    -Yaş Sınırı Kayıt günü itibarıyla 18 yaşını bitirmemiş olmak gerekir.
    -Evlilik Durumu Evli olanların kaydı yapılmaz; öğrenciyken evlenenlerin kaydı Açık Lise'ye aktarılır.
    -Hazırlıkta Başarısızlık Üst üste iki yıl başarısız olan öğrenci, hazırlık sınıfı olmayan bir okulun 9. sınıfına nakledilir.
6. Devamsızlık:
    -Özürsüz Sınırı Özürsüz 10 günü geçen öğrenci başarısız sayılır.
    -Toplam Sınır Özürlü ve özürsüz toplam devamsızlık 30 günü aşamaz.
    -60 Günlük İstisna Organ nakli, ağır hastalık veya tutukluluk gibi hallerde toplam sınır 60 gündür.
    -Geç Gelme Sadece birinci ders saati için geçerlidir; sonrası devamsızlıktan sayılır.
7. Sınıf Geçme:
    -Yazılı Sınav Sayısı Haftalık ders saatinden bağımsız olarak her dersten en az 2 yazılı yapılır.
    -Başarı Puanı Bir dersten başarılı sayılmak için yılsonu puanının en az 50 olması gerekir.
    -Sorumlu Geçme Bir sınıfta en fazla 3 dersten başarısız olanlar sorumlu geçer.
    -Sınıf Tekrarı Toplam başarısız ders sayısı 6'yı geçerse öğrenci sınıf tekrar eder.
    -Beceri Sınavı Puanın %80'i sınavdan, %20'si iş dosyasından gelir
8. Nakil İşlemleri:
    -Başvuru Zamanı Aralık ve Mayıs ayları hariç her ayın ilk iş günü başvurulabilir.
    -Ders Seçimi (9.Sınıf) Yeni başlayanlar için ders seçimi ders yılının ilk haftasında yapılır.
9. Disiplin ve Ödül:
    -Disiplin Cezaları Kınama, kısa süreli uzaklaştırma, okul değiştirme, örgün eğitim dışına çıkarma.
    -Kopya ve Sigara Kopya çekmek veya tütün mamulü kullanmak "Kınama" cezası gerektirir.
    -Teşekkür Belgesi Ağırlıklı ortalaması 70,00 - 84,99 arası olanlara verilir
    -Takdir Belgesi Ağırlıklı ortalaması 85,00 ve üzeri olanlara verilir.







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

# --- 6. CHAT ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: 12a programı nedir?)..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Ders Programı Kontrolü ---
    kucuk_prompt = prompt.lower()
    istenen_sinif = None
    sinif_listesi = ["9a", "10a", "11a", "12a"]

    if "program" in kucuk_prompt or "ders" in kucuk_prompt:
        for s in sinif_listesi:
            if s in kucuk_prompt:
                istenen_sinif = s
                break

    # --- Ders Programı Sorulursa (Tablo gösterilmez, AI’ya prompt olarak verilir)
    if istenen_sinif:
        with st.chat_message("assistant"):
            with st.spinner(f"{istenen_sinif.upper()} programı analiz ediliyor..."):
                df_program = program_yukle(istenen_sinif)
                if df_program is not None:
                    # Tabloyu string hâline getir
                    program_text = df_program.to_string(index=False)
                    ai_prompt = f"""
Aşağıda kullanıcının {istenen_sinif.upper()} sınıfı için ders programı bulunmaktadır:

{program_text}

Kullanıcının sorusu: {prompt}

Lütfen bu programa göre öneri ve yorum yap, cevabı maddeler halinde ver.
"""
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": ai_prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0
                    )
                    cevap = chat_completion.choices[0].message.content
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                else:
                    hata_mesaji = f"Üzgünüm, Excel dosyasında '{istenen_sinif}' isimli bir sayfa bulamadım."
                    st.error(hata_mesaji)
                    st.session_state.messages.append({"role": "assistant", "content": hata_mesaji})

    # --- Normal Yönetmelik Sorusu ---
    else:
        with st.chat_message("assistant"):
            with st.spinner("Yönetmelik taranıyor..."):
                cevap = okul_asistani_sorgula(prompt)
                st.write(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
