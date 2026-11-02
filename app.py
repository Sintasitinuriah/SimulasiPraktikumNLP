import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import io
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

# Import Sastrawi & NLTK
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Modul Praktikum NLP & Generator Colab (.ipynb)",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Laboratorium Virtual NLP & Generator Latihan Google Colab")
st.markdown("""
Aplikasi ini dirancang untuk **mensimulasikan tahapan NLP secara interaktif** mulai dari **Pembacaan Data**, **Text Preprocessing**, **Representasi Numerik (`BoW` & `TF-IDF`)**, hingga **Pemilihan & Simulasi 6 Aplikasi Pemodelan NLP yang Menyesuaikan Dataset Anda**, sekaligus **menghasilkan file latihan Jupyter Notebook (`.ipynb`)** untuk **Google Colab**.
""")

# ==========================================
# FUNGSI CACHING & DATASET CONTOH
# ==========================================
@st.cache_resource
def load_sastrawi():
    if SASTRAWI_AVAILABLE:
        stemmer = StemmerFactory().create_stemmer()
        stops = set(StopWordRemoverFactory().get_stop_words())
        return stemmer, stops
    return None, set()

@st.cache_resource
def load_nltk():
    if NLTK_AVAILABLE:
        ps = PorterStemmer()
        lemma = WordNetLemmatizer()
        try:
            en_stops = set(nltk_stopwords.words("english"))
        except Exception:
            en_stops = {"the", "a", "an", "in", "on", "of", "and", "is", "are", "was", "it", "to", "for"}
        return ps, lemma, en_stops
    return None, None, set()

stemmer_id, stopwords_id_base = load_sastrawi()
stemmer_en, lemmatizer_en, stopwords_en_base = load_nltk()

def get_sample_dataframe(lang="Bahasa Indonesia", preset="Sentimen"):
    if preset == "Spam Detection":
        return pd.DataFrame({
            "id_dokumen": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "teks_pesan": [
                "SELAMAT! Nomor Anda terpilih mendapatkan hadiah undian 50 juta. Klik link bit.ly/klaim sekarang!",
                "Butuh dana tunai cepat cair 5 menit? Pinjol resmi tanpa agunan bunga rendah hubungi WA 081234",
                "PROMO DISKON 90% khusus hari ini! Beli 1 gratis 2 voucher belanja, klaim kupon gratis di sini!",
                "Halo kak, jangan lupa nanti siang kita ada jadwal kuliah praktikum NLP jam 13.00 di laboratorium.",
                "Tolong kirimkan revisi laporan tugas kelompok kita ke email sebelum jam 5 sore ya, terima kasih.",
                "Nanti malam jadi makan seblak dan bakso di warung dekat kampus tidak?"
            ],
            "label": ["Spam", "Spam", "Spam", "Ham", "Ham", "Ham"]
        })
    elif preset == "Chatbot QA":
        return pd.DataFrame({
            "id_faq": ["FAQ-1", "FAQ-2", "FAQ-3", "FAQ-4", "FAQ-5"],
            "pertanyaan": [
                "Apa itu tahapan preprocessing dalam NLP?",
                "Apa perbedaan antara Bag of Words dan TF-IDF?",
                "Bagaimana cara menginstal library Sastrawi di Google Colab?",
                "Kapan batas waktu pengumpulan tugas praktikum NLP?",
                "Algoritma apa saja yang sering digunakan untuk analisis sentimen?"
            ],
            "jawaban": [
                "Preprocessing NLP adalah tahap pembersihan teks mentah meliputi Cleansing, Case Folding, Tokenization, Normalization, Stopword Removal, dan Stemming.",
                "Bag of Words hanya menghitung frekuensi kemunculan kata, sedangkan TF-IDF membobotkan kata berdasarkan frekuensi (TF) dan keunikannya di seluruh dokumen (IDF).",
                "Jalankan perintah !pip install Sastrawi pada cell Google Colab, lalu import StemmerFactory.",
                "Batas waktu pengumpulan file .ipynb praktikum NLP adalah hari Jumat pukul 23.59 WIB.",
                "Algoritma yang populer antara lain Multinomial Naive Bayes, Support Vector Machine (SVM), Logistic Regression, dan IndoBERT."
            ]
        })
    elif preset == "Machine Translation":
        return pd.DataFrame({
            "id_kalimat": ["D1", "D2", "D3", "D4", "D5"],
            "teks_indonesia": [
                "Saya sangat suka makan seblak pedas di warung ini",
                "Pelayanan di warung ini sangat ramah dan cepat",
                "Harga bakso dan mie pedas ini sangat murah",
                "Belajar pemrosesan bahasa alami sangat menyenangkan",
                "Kami sedang mengerjakan tugas praktikum algoritma di kampus"
            ],
            "teks_inggris": [
                "I really like eating spicy seblak at this shop",
                "The service at this shop is very friendly and fast",
                "The price of meatballs and spicy noodles is very cheap",
                "Learning natural language processing is very fun",
                "We are working on algorithm lab assignments at campus"
            ]
        })
    if lang == "Bahasa Indonesia":
        return pd.DataFrame({
            "id_dokumen": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "teks_ulasan": [
                "Wah, seblak di warung ini SANGAT pedas sekali...!!! 🥵 Aku suka bangettt, harganya 10ribu @kuliner_tasik https://seblak.com",
                "Seblak pedas itu enak bgt, kuahnya kental dan kerupuknya empuk mantulll 100%!",
                "Pelayanan di warung ini agak lambat, tapi rasa seblak dan bakso pedasnya juara 👍",
                "Kurang suka sama bakso di sini, kuahnya keasinan dan harganya 25ribu terlalu mahal :(",
                "Rekomendasi banget buat pecinta kuliner pedas! Seblak dan mie pedasnya bikin nagih terus.",
                "Makanan datang dingin, pelayanannya buruk dan mengecewakan sekali, kapok beli di sini."
            ],
            "label": ["Positif", "Positif", "Positif", "Negatif", "Positif", "Negatif"]
        })
    else:
        return pd.DataFrame({
            "id_dokumen": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "teks_ulasan": [
                "Wow, the coffee at this shop is REALLY bad...!!! 😠 I absolutely hate itttt, costs $10 @NYCafe https://coffee.com",
                "The spicy noodles and hot coffee here are super delicious and affordable!",
                "Service at this shop was quite slow, waiting 30 mins, but the coffee tastes great 👍",
                "I really hate the food here, too salty and overpriced at $25 :(",
                "Highly recommended for coffee lovers! Their espresso and pastries are amazing.",
                "Terrible experience, cold food and rude staff, never coming back again."
            ],
            "label": ["Negative", "Positive", "Positive", "Negative", "Positive", "Negative"]
        })

# Fungsi Deteksi Karakteristik Dataset & Rekomendasi Model NLP
def analisis_kesesuaian_dataset(df, kol_teks):
    semua_kolom = list(df.columns)
    kolom_lain = [c for c in semua_kolom if c != kol_teks]
    rata_kata = df[kol_teks].astype(str).apply(lambda x: len(x.split())).mean() if len(df) > 0 else 0

    # Cek kandidat kolom label kategorikal (2 s.d. 15 kelas unik)
    kandidat_label = []
    for c in kolom_lain:
        n_uniq = df[c].dropna().nunique()
        if 2 <= n_uniq <= 15 and n_uniq < len(df):
            kandidat_label.append(c)

    # Cek kandidat kolom teks kedua (untuk QA Chatbot / Translasi Paralel)
    kandidat_teks_kedua = []
    for c in kolom_lain:
        if df[c].dtype == "object":
            avg_len = df[c].astype(str).apply(lambda x: len(x.split())).mean()
            if avg_len >= 2.0:
                kandidat_teks_kedua.append(c)

    rekomendasi = []
    # 1. Sentiment Analysis
    if kandidat_label:
        rekomendasi.append({
            "Model Aplikasi NLP": "1. Sentiment Analysis",
            "Status Kesesuaian": "✅ Sangat Sesuai (Berlabel)",
            "Alasan & Konfigurasi Kolom": f"Ditemukan kolom target/label `{kandidat_label[0]}` ({df[kandidat_label[0]].nunique()} kelas) untuk klasifikasi opini."
        })
    else:
        rekomendasi.append({
            "Model Aplikasi NLP": "1. Sentiment Analysis",
            "Status Kesesuaian": "⚡ Bisa Digunakan (Auto-Lexicon / Label Manual)",
            "Alasan & Konfigurasi Kolom": "Dataset belum memiliki kolom label khusus; sistem menyediakan pelabelan otomatis berbasis kamus sentimen."
        })

    # 2. Spam Detection
    if kandidat_label:
        rekomendasi.append({
            "Model Aplikasi NLP": "2. Spam Detection",
            "Status Kesesuaian": "✅ Sangat Sesuai (Berlabel)",
            "Alasan & Konfigurasi Kolom": f"Dapat melatih detektor pesan menggunakan teks `{kol_teks}` dan kolom kategori `{kandidat_label[0]}`."
        })
    else:
        rekomendasi.append({
            "Model Aplikasi NLP": "2. Spam Detection",
            "Status Kesesuaian": "⚡ Bisa Digunakan (Rule/Keyword Tagging)",
            "Alasan & Konfigurasi Kolom": "Dapat mendeteksi indikasi kata pemicu spam pada kolom teks atau melabeli langsung di tabel editor."
        })

    # 3. Machine Translation
    if kandidat_teks_kedua:
        rekomendasi.append({
            "Model Aplikasi NLP": "3. Machine Translation",
            "Status Kesesuaian": "✅ Sangat Sesuai (Korpus Paralel)",
            "Alasan & Konfigurasi Kolom": f"Memiliki pasangan kolom teks `{kol_teks}` dan `{kandidat_teks_kedua[0]}` sebagai teks sumber & target."
        })
    else:
        rekomendasi.append({
            "Model Aplikasi NLP": "3. Machine Translation",
            "Status Kesesuaian": "✅ Siap Digunakan (Translasi Dokumen)",
            "Alasan & Konfigurasi Kolom": f"Setiap baris dokumen pada kolom `{kol_teks}` dapat diterjemahkan langsung antarbahasa."
        })

    # 4. Text Summarization
    rekomendasi.append({
        "Model Aplikasi NLP": "4. Text Summarization",
        "Status Kesesuaian": "✅ Sangat Sesuai" if rata_kata >= 8 else "✅ Siap Digunakan (Multi-Dokumen)",
        "Alasan & Konfigurasi Kolom": f"Rata-rata panjang teks {rata_kata:.1f} kata/baris. Dapat meringkas per dokumen atau gabungan seluruh dokumen korpus."
    })

    # 5. Topic Modeling
    rekomendasi.append({
        "Model Aplikasi NLP": "5. Topic Modeling (LDA / NMF)",
        "Status Kesesuaian": "✅ Sangat Sesuai (Tanpa Label / Unsupervised)",
        "Alasan & Konfigurasi Kolom": f"Langsung mengelompokkan topik tersembunyi dari matriks BoW/TF-IDF kolom `{kol_teks}`."
    })

    # 6. Chatbot
    if kandidat_teks_kedua:
        rekomendasi.append({
            "Model Aplikasi NLP": "6. Chatbot Retrieval",
            "Status Kesesuaian": "✅ Sangat Sesuai (Pasangan Q&A)",
            "Alasan & Konfigurasi Kolom": f"Menggunakan `{kol_teks}` sebagai Pertanyaan Acuan dan `{kandidat_teks_kedua[0]}` sebagai Jawaban Bot."
        })
    else:
        rekomendasi.append({
            "Model Aplikasi NLP": "6. Chatbot Retrieval",
            "Status Kesesuaian": "✅ Siap Digunakan (Pencarian Dokumen Korpus)",
            "Alasan & Konfigurasi Kolom": f"Menjadikan baris dokumen pada `{kol_teks}` sebagai basis pengetahuan (Knowledge Base) yang bisa ditanyakan."
        })

    return pd.DataFrame(rekomendasi), kandidat_label, kandidat_teks_kedua

# ==========================================
# SIDEBAR: NAVIGASI & PENGATURAN BAHASA
# ==========================================
st.sidebar.header("Alur Praktikum NLP")
st.sidebar.markdown("""
1. **Tahap 1:** Pembacaan Data & Pemetaan Kolom
2. **Tahap 2:** *Text Preprocessing*
3. **Tahap 3:** Representasi Numerik (`BoW` & `TF-IDF`)
4. **Tahap 4:** Export `.ipynb` ke Google Colab
5. **Tahap 5:** Simulasi Model Sesuai Dataset
""")
st.sidebar.markdown("---")
bahasa = st.sidebar.radio("Pilih Bahasa Dataset:", ["Bahasa Indonesia", "Bahasa Inggris"])
SHOW_MODELING = st.sidebar.checkbox("Aktifkan Tahap 5 (Pemodelan Sesuai Dataset)", value=True)

# ==========================================
# TAHAP 1: PEMBACAAN DATA (TXT, CSV, EXCEL)
# ==========================================
st.header("Tahap 1: Pembacaan Dataset & Konfigurasi Kolom")

col_src, col_tpl = st.columns([2, 1])
with col_src:
    sumber_data = st.radio(
        "Pilih Sumber Data Simulasi:",
        ["Gunakan Dataset Contoh", "Unggah Dataset Sendiri (.txt, .csv, .xlsx)"],
        horizontal=True
    )

    if sumber_data == "Gunakan Dataset Contoh":
        preset_contoh = st.selectbox(
            "Pilih Jenis Dataset Contoh (Untuk Menguji Berbagai Model):"
            , ["Sentimen Ulasan Kuliner (Teks + Label Sentimen)",
               "Deteksi Pesan Spam (Teks + Label Spam/Ham)",
               "Basis Pengetahuan Chatbot (Kolom Pertanyaan + Kolom Jawaban)",
               "Korpus Paralel Translasi (Bahasa Indonesia + Bahasa Inggris)"]
        )
    else:
        preset_contoh = "Upload"

with col_tpl:
    with st.expander("📥 Unduh File Dataset Contoh (.txt / .csv / .xlsx)"):
        df_tpl = get_sample_dataframe(bahasa, "Sentimen")
        st.download_button(
            "📄 Unduh dataset_contoh.txt",
            data="\n".join(df_tpl["teks_ulasan"].tolist()).encode("utf-8"),
            file_name="dataset_contoh.txt",
            mime="text/plain"
        )
        st.download_button(
            "📊 Unduh dataset_contoh.csv",
            data=df_tpl.to_csv(index=False).encode("utf-8"),
            file_name="dataset_contoh.csv",
            mime="text/csv"
        )
        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
            df_tpl.to_excel(writer, index=False, sheet_name="Sheet1")
        st.download_button(
            "📗 Unduh dataset_contoh.xlsx",
            data=buf_xlsx.getvalue(),
            file_name="dataset_contoh.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

df_raw = None
kolom_teks = None
nama_file_aktif = "dataset_contoh.csv"
tipe_file_aktif = "csv"
delimiter_aktif = ","
sheet_aktif = "Sheet1"

if sumber_data == "Gunakan Dataset Contoh":
    if preset_contoh.startswith("Sentimen"):
        df_raw = get_sample_dataframe(bahasa, "Sentimen")
        kolom_teks = "teks_ulasan"
    elif preset_contoh.startswith("Deteksi"):
        df_raw = get_sample_dataframe(bahasa, "Spam Detection")
        kolom_teks = "teks_pesan"
    elif preset_contoh.startswith("Basis"):
        df_raw = get_sample_dataframe(bahasa, "Chatbot QA")
        kolom_teks = "pertanyaan"
    else:
        df_raw = get_sample_dataframe(bahasa, "Machine Translation")
        kolom_teks = "teks_indonesia"
else:
    uploaded = st.file_uploader("Unggah file .txt, .csv, atau .xlsx Anda:", type=["txt", "csv", "xlsx", "xls"])
    if uploaded is not None:
        nama_file_aktif = uploaded.name
        ext = nama_file_aktif.split(".")[-1].lower()
        try:
            if ext == "txt":
                tipe_file_aktif = "txt"
                isi = uploaded.read().decode("utf-8", errors="ignore")
                baris = [b.strip() for b in isi.split("\n") if b.strip()]
                df_raw = pd.DataFrame({
                    "id_dokumen": [f"D{i+1}" for i in range(len(baris))],
                    "teks_dokumen": baris
                })
                kolom_teks = "teks_dokumen"
            elif ext == "csv":
                tipe_file_aktif = "csv"
                c1, c2 = st.columns(2)
                with c1:
                    delimiter_aktif = st.selectbox("Pilih Delimiter CSV:", [",", ";", "\t", "|"])
                with c2:
                    enc_aktif = st.selectbox("Pilih Encoding:", ["utf-8", "latin1", "cp1252"])
                df_raw = pd.read_csv(uploaded, sep=delimiter_aktif, encoding=enc_aktif)
            elif ext in ["xlsx", "xls"]:
                tipe_file_aktif = "excel"
                xls = pd.ExcelFile(uploaded)
                sheet_aktif = st.selectbox("Pilih Nama Sheet Excel:", xls.sheet_names)
                df_raw = pd.read_excel(uploaded, sheet_name=sheet_aktif)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

if df_raw is None:
    st.info("👆 Silakan unggah file terlebih dahulu atau gunakan Dataset Contoh.")
    st.stop()

# Pemilihan Kolom Teks Utama & Kolom Pendukung secara Dinamis
semua_kol = list(df_raw.columns)
default_idx_teks = semua_kol.index(kolom_teks) if kolom_teks in semua_kol else (1 if len(semua_kol) > 1 else 0)

st.markdown("#### 🎯 Konfigurasi Kolom Dataset")
col_k1, col_k2, col_k3 = st.columns(3)
with col_k1:
    kolom_teks = st.selectbox("1. Pilih Kolom Teks Utama (Wajib):", semua_kol, index=default_idx_teks)

df_raw = df_raw.dropna(subset=[kolom_teks]).reset_index(drop=True)
df_raw[kolom_teks] = df_raw[kolom_teks].astype(str)

df_rekomendasi, list_kandidat_label, list_kandidat_teks2 = analisis_kesesuaian_dataset(df_raw, kolom_teks)

opsi_kolom_opsional = ["(Tidak Ada / Gunakan Simulasi Otomatis)"] + [c for c in semua_kol if c != kolom_teks]
with col_k2:
    default_lbl_idx = opsi_kolom_opsional.index(list_kandidat_label[0]) if list_kandidat_label and list_kandidat_label[0] in opsi_kolom_opsional else 0
    kolom_label_user = st.selectbox(
        "2. Pilih Kolom Label/Target (Untuk Sentimen / Spam):",
        opsi_kolom_opsional,
        index=default_lbl_idx,
        help="Pilih kolom yang berisi label kelas (misal: Positif/Negatif atau Spam/Ham) jika dataset Anda memilikinya."
    )
with col_k3:
    default_txt2_idx = opsi_kolom_opsional.index(list_kandidat_teks2[0]) if list_kandidat_teks2 and list_kandidat_teks2[0] in opsi_kolom_opsional else 0
    kolom_pasangan_user = st.selectbox(
        "3. Pilih Kolom Pasangan Teks (Untuk Chatbot QA / Translasi):",
        opsi_kolom_opsional,
        index=default_txt2_idx,
        help="Pilih kolom jawaban (untuk Chatbot) atau kolom bahasa target (untuk Machine Translation) jika ada."
    )

tab_data_sim, tab_data_rec, tab_data_colab = st.tabs([
    "📊 Pratinjau Dataset Aktif",
    "🧭 Analisis Kesesuaian Model NLP dengan Dataset Ini",
    "💻 Tutorial Kode Pembacaan Data di Google Colab"
])

with tab_data_sim:
    max_baris = len(df_raw)
    if max_baris > 3:
        jml_dok = st.slider(
            "Jumlah baris dokumen yang disimulasikan di layar (agar matriks dan model cepat dieksekusi):",
            min_value=2, max_value=min(100, max_baris), value=min(10, max_baris)
        )
        df_sim = df_raw.iloc[:jml_dok].copy().reset_index(drop=True)
    else:
        df_sim = df_raw.copy().reset_index(drop=True)

    doc_labels = [f"D{i+1}" for i in range(len(df_sim))]
    if "Kode_Doc" not in df_sim.columns:
        df_sim.insert(0, "Kode_Doc", doc_labels)
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

with tab_data_rec:
    st.markdown("Berikut adalah hasil deteksi otomatis struktur dataset Anda beserta rekomendasi model NLP yang dapat dijalankan di **Tahap 5**:")
    st.dataframe(df_rekomendasi, use_container_width=True, hide_index=True)

with tab_data_colab:
    st.markdown("""
    ### Panduan Membaca Dataset di Google Colab
    Salin kode berikut ke Google Colab untuk mengunggah dan membaca file `.txt`, `.csv`, maupun `.xlsx`:
    """)
    st.code(f"""import pandas as pd
from google.colab import files

# 1. Unggah file dari laptop ke Google Colab
uploaded = files.upload()

# 2A. Membaca File CSV
df = pd.read_csv("{nama_file_aktif if tipe_file_aktif=="csv" else "dataset_contoh.csv"}", sep="{delimiter_aktif}", encoding="utf-8")

# 2B. Membaca File Excel (.xlsx)
# df = pd.read_excel("{nama_file_aktif if tipe_file_aktif=="excel" else "dataset_contoh.xlsx"}", sheet_name="{sheet_aktif}")

# 2C. Membaca File TXT (1 baris = 1 dokumen)
# with open("{nama_file_aktif if tipe_file_aktif=="txt" else "dataset_contoh.txt"}", "r", encoding="utf-8") as f:
#     baris_teks = [line.strip() for line in f if line.strip()]
# df = pd.DataFrame({{"{kolom_teks}": baris_teks}})

display(df.head())""", language="python")

# ==========================================
# TAHAP 2: TEXT PREPROCESSING
# ==========================================
st.markdown("---")
st.header("Tahap 2: Text Preprocessing")

c_p1, c_p2, c_p3 = st.columns(3)
with c_p1:
    st.markdown("**1. Cleansing & Case Folding**")
    use_clean_url = st.checkbox("Hapus URL, @Mention, #Hashtag", value=True)
    use_clean_num = st.checkbox("Hapus Angka", value=True)
    use_clean_punct = st.checkbox("Hapus Tanda Baca & Emoji", value=True)
    use_lower = st.checkbox("Case Folding (Lowercase)", value=True)

with c_p2:
    st.markdown("**2. Normalisasi Kata Tidak Baku**")
    use_norm = st.checkbox("Aktifkan Kamus Normalisasi Slang", value=True)
    default_slang = (
        "bangettt: sekali, bgt: sekali, banget: sekali, mantulll: mantap, yg: yang, gak: tidak, gk: tidak, sama: dengan, buat: untuk, ribu: "
        if bahasa == "Bahasa Indonesia"
        else "itttt: it, mins: minutes, u: you, r: are"
    )
    slang_text = st.text_area("Kamus Slang (`kata_slang: kata_baku`):", value=default_slang, height=90)
    slang_dict = {}
    for pair in slang_text.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            slang_dict[k.strip().lower()] = v.strip().lower()

with c_p3:
    st.markdown("**3. Stopword Removal & Stemming**")
    use_stop = st.checkbox("Hapus Stopwords", value=True)
    default_stops = "wah, aku, saya, ini, itu, di, dan, yang, sekali, sangat, agak, tapi, sini, terus" if bahasa == "Bahasa Indonesia" else "wow, really, quite, very, too, absolutely"
    custom_stop_str = st.text_input("Tambahan Stopwords (pisahkan koma):", value=default_stops)
    custom_stops = {w.strip().lower() for w in custom_stop_str.split(",") if w.strip()}
    use_stem = st.checkbox("Aktifkan Stemming / Lemmatization", value=True)

def jalankan_preprocessing(teks):
    t0 = str(teks)
    t1 = t0
    if use_clean_url:
        t1 = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", " ", t1)
    if use_clean_num:
        t1 = re.sub(r"\d+", " ", t1)
    if use_clean_punct:
        t1 = re.sub(r"[^\w\s]", " ", t1)
    t1 = re.sub(r"\s+", " ", t1).strip()

    t2 = t1.lower() if use_lower else t1
    t3 = t2.split()

    if use_norm:
        t4 = [slang_dict.get(w.lower(), w) for w in t3 if slang_dict.get(w.lower(), w) != ""]
    else:
        t4 = list(t3)

    if use_stop:
        base_s = stopwords_id_base if bahasa == "Bahasa Indonesia" else stopwords_en_base
        all_s = base_s.union(custom_stops)
        t5 = [w for w in t4 if w.lower() not in all_s]
    else:
        t5 = list(t4)

    if use_stem:
        if bahasa == "Bahasa Indonesia":
            t6 = [stemmer_id.stem(w) for w in t5] if (SASTRAWI_AVAILABLE and stemmer_id) else list(t5)
        else:
            t6 = [lemmatizer_en.lemmatize(w) for w in t5] if (NLTK_AVAILABLE and lemmatizer_en) else list(t5)
    else:
        t6 = list(t5)

    return {
        "0_asli": t0,
        "1_cleansing": t1,
        "2_case_folding": t2,
        "3_tokenization": t3,
        "4_normalization": t4,
        "5_stopword": t5,
        "6_stemming": t6,
        "teks_akhir": " ".join(t6)
    }

hasil_prep_list = [jalankan_preprocessing(t) for t in df_sim[kolom_teks]]

tab_prep_sim1, tab_prep_sim2, tab_prep_colab = st.tabs([
    "🔬 Simulasi Tabel Tahapan (Per Kalimat)",
    "📋 Simulasi Seluruh Dokumen (Before vs After)",
    "💻 Tutorial Kode Preprocessing di Google Colab"
])

with tab_prep_sim1:
    idx_kal = st.selectbox(
        "Pilih kalimat dokumen yang ingin diamati perubahannya:",
        range(len(df_sim)),
        format_func=lambda i: f"{doc_labels[i]}: {df_sim[kolom_teks].iloc[i][:75]}..."
    )
    hp = hasil_prep_list[idx_kal]
    df_step_kal = pd.DataFrame({
        "Tahapan": [
            "0. Kalimat Asli",
            "1. Cleansing",
            "2. Case Folding",
            "3. Tokenization",
            "4. Normalization",
            "5. Stopword Removal",
            "6. Stemming / Lemmatization"
        ],
        "Hasil Transformasi": [
            hp["0_asli"],
            hp["1_cleansing"],
            hp["2_case_folding"],
            str(hp["3_tokenization"]),
            str(hp["4_normalization"]),
            str(hp["5_stopword"]),
            str(hp["6_stemming"])
        ]
    })
    st.dataframe(df_step_kal, use_container_width=True, hide_index=True)

with tab_prep_sim2:
    df_all_prep = pd.DataFrame({
        "Kode_Doc": doc_labels,
        "Teks Asli": [r["0_asli"] for r in hasil_prep_list],
        "1. Cleansing & Lower": [r["2_case_folding"] for r in hasil_prep_list],
        "2. Normalisasi": [" ".join(r["4_normalization"]) for r in hasil_prep_list],
        "3. Stopword Removal": [" ".join(r["5_stopword"]) for r in hasil_prep_list],
        "4. Teks Bersih (Stemmed)": [r["teks_akhir"] for r in hasil_prep_list]
    })
    st.dataframe(df_all_prep, use_container_width=True, hide_index=True)

with tab_prep_colab:
    st.markdown("### Kode Implementasi Preprocessing di Google Colab")
    st.code(f"""import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download("punkt")
nltk.download("punkt_tab")

kamus_slang = {json.dumps(slang_dict, ensure_ascii=False)}
stopwords_indo = set(StopWordRemoverFactory().get_stop_words()).union({set(custom_stops)})
stemmer = StemmerFactory().create_stemmer()

def tahapan_preprocessing(teks):
    clean = re.sub(r"http\\S+|www\\.\\S+|@\\w+|#\\w+|\\d+|[^\\w\\s]", " ", str(teks))
    clean = re.sub(r"\\s+", " ", clean).strip()
    case_fold = clean.lower()
    tokens = word_tokenize(case_fold)
    norm = [kamus_slang.get(k, k) for k in tokens if kamus_slang.get(k, k) != ""]
    no_stop = [k for k in norm if k not in stopwords_indo]
    stemmed = [stemmer.stem(k) for k in no_stop]
    return " ".join(stemmed)

df["teks_bersih"] = df["{kolom_teks}"].apply(tahapan_preprocessing)
pd.set_option("display.max_colwidth", None)
display(df[["{kolom_teks}", "teks_bersih"]].head())""", language="python")

# ==========================================
# TAHAP 3: REPRESENTASI NUMERIK (BoW & TF-IDF)
# ==========================================
st.markdown("---")
st.header("Tahap 3: Representasi Numerik (Count / BoW & TF-IDF)")

col_m1, col_m2 = st.columns(2)
with col_m1:
    versi_rumus = st.radio(
        "Pilih Versi Rumus Perhitungan TF-IDF:",
        ["1. Rumus Teori Dasar (Hitungan Manual Buku)", "2. Rumus Standar Scikit-Learn (Python Default)"]
    )
with col_m2:
    if versi_rumus == "1. Rumus Teori Dasar (Hitungan Manual Buku)":
        opsi_tf = st.selectbox("Variasi Rumus TF:", ["Raw Count (Frekuensi Kemunculan)", "Normalized TF (Frekuensi / Total Kata)"])
        opsi_l2 = st.checkbox("Gunakan L2 Normalization di akhir?", value=False)
    else:
        st.info("Pada Scikit-Learn: TF menggunakan Raw Count, IDF menggunakan rumus ln((1+N)/(1+DF)) + 1, diikuti L2 Normalization.")
        opsi_tf = "Raw Count (Frekuensi Kemunculan)"
        opsi_l2 = True

tokens_per_doc = [r["6_stemming"] for r in hasil_prep_list]
vocab = sorted(list(set(w for doc in tokens_per_doc for w in doc)))
N = len(tokens_per_doc)

if len(vocab) == 0:
    st.warning("Vocabulary kosong setelah preprocessing. Kurangi filter stopword atau tambah kalimat.")
    st.stop()

# 1. BoW (Count)
bow_mat = np.zeros((N, len(vocab)), dtype=int)
for i, d_tokens in enumerate(tokens_per_doc):
    for j, term in enumerate(vocab):
        bow_mat[i, j] = d_tokens.count(term)
df_bow = pd.DataFrame(bow_mat, index=doc_labels, columns=vocab)

# 2. TF
tf_mat = np.zeros((N, len(vocab)), dtype=float)
for i in range(N):
    tot = len(tokens_per_doc[i])
    for j in range(len(vocab)):
        tf_mat[i, j] = bow_mat[i, j] if opsi_tf.startswith("Raw") else (bow_mat[i, j] / tot if tot > 0 else 0.0)
df_tf = pd.DataFrame(np.round(tf_mat, 4), index=doc_labels, columns=vocab)

# 3. DF & IDF
df_vec = np.sum(bow_mat > 0, axis=0)
idf_vec = np.zeros(len(vocab), dtype=float)
for j, df_v in enumerate(df_vec):
    if versi_rumus.startswith("1."):
        idf_vec[j] = math.log10(N / df_v) if df_v > 0 else 0.0
    else:
        idf_vec[j] = math.log((1 + N) / (1 + df_v)) + 1.0

df_idf_info = pd.DataFrame({
    "Term (Kata)": vocab,
    "Frekuensi Total": np.sum(bow_mat, axis=0),
    "Document Frequency (DF)": df_vec,
    "Nilai IDF": np.round(idf_vec, 4)
})

# 4. TF-IDF
tfidf_raw = tf_mat * idf_vec
if opsi_l2:
    norms = np.linalg.norm(tfidf_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_fin = tfidf_raw / norms
else:
    tfidf_fin = tfidf_raw

df_tfidf = pd.DataFrame(np.round(tfidf_fin, 4), index=doc_labels, columns=vocab)

t_num1, t_num2, t_num3, t_num4, t_num5 = st.tabs([
    "Matriks Count / BoW",
    "Matriks TF",
    "Tabel DF & IDF",
    "Matriks TF-IDF & Bedah Hitungan",
    "Tutorial Kode BoW & TF-IDF di Google Colab"
])

with t_num1:
    st.markdown("#### Matriks Bag of Words (CountVectorizer)")
    st.dataframe(df_bow.style.background_gradient(cmap="Greens", axis=None), use_container_width=True)

with t_num2:
    st.markdown("#### Matriks Term Frequency (TF)")
    st.dataframe(df_tf.style.format("{:.4f}"), use_container_width=True)

with t_num3:
    st.markdown("#### Tabel Document Frequency (DF) & Inverse Document Frequency (IDF)")
    if versi_rumus.startswith("1."):
        st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{DF(t)}\right)")
    else:
        st.latex(r"IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1")
    st.dataframe(df_idf_info, use_container_width=True, hide_index=True)

with t_num4:
    st.markdown("#### Matriks Akhir TF-IDF")
    st.dataframe(df_tfidf.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)

    st.markdown("---")
    st.subheader("Bedah Hitungan Manual per Kata")
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        d_pilih = st.selectbox("Pilih Dokumen:", range(N), format_func=lambda i: f"{doc_labels[i]}: '{hasil_prep_list[i]['teks_akhir']}'")
    with c_b2:
        k_pilih = st.selectbox("Pilih Kata (Term):", vocab)

    idx_k = vocab.index(k_pilih)
    f_val = bow_mat[d_pilih, idx_k]
    tf_val = tf_mat[d_pilih, idx_k]
    df_val = df_vec[idx_k]
    idf_val = idf_vec[idx_k]
    raw_val = tfidf_raw[d_pilih, idx_k]
    fin_val = tfidf_fin[d_pilih, idx_k]

    st.markdown(f"""
    * **Frekuensi Kata (`{k_pilih}` di `{doc_labels[d_pilih]}`):** Muncul **{f_val} kali** → **TF = `{tf_val:.4f}`**
    * **Document Frequency (`DF`):** Muncul di **{df_val} dokumen** dari total **$N = {N}$ dokumen** → **IDF = `{idf_val:.4f}`**
    """)
    st.latex(rf"TF\text{{-}}IDF_{{\text{{awal}}}} = {tf_val:.4f} \times {idf_val:.4f} = {raw_val:.4f}")
    if opsi_l2:
        norm_row = np.linalg.norm(tfidf_raw[d_pilih])
        if norm_row > 0:
            st.latex(rf"TF\text{{-}}IDF_{{\text{{L2 Norm}}}} = \frac{{{raw_val:.4f}}}{{{norm_row:.4f}}} = \mathbf{{{fin_val:.4f}}}")

with t_num5:
    st.markdown("### Kode Implementasi CountVectorizer (BoW) & TfidfVectorizer di Google Colab")
    st.code("""from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# 1. Perhitungan Bag of Words (CountVectorizer)
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(df["teks_bersih"])
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out(),
    index=[f"D{i+1}" for i in range(len(df))]
)
display(df_bow)

# 2. Perhitungan TF-IDF (TfidfVectorizer)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["teks_bersih"])
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray().round(4),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=[f"D{i+1}" for i in range(len(df))]
)
display(df_tfidf)""", language="python")

# ==========================================
# TAHAP 5: PEMILIHAN & SIMULASI MODEL SESUAI DATASET
# ==========================================
st.markdown("---")
if not SHOW_MODELING:
    st.info("🔒 **Tahap 5: Pemodelan Machine Learning (6 Aplikasi NLP)** saat ini dikunci (*hidden*). Centang opsi **'Aktifkan Tahap 5'** pada Sidebar di sebelah kiri untuk membuka simulasi.")
else:
    st.header("Tahap 5: Pemilihan & Simulasi Model NLP Sesuai Dataset Anda")
    st.markdown("""
    Pilih **Model Aplikasi NLP** yang ingin dijalankan pada dataset aktif Anda (`df_sim`). Seluruh model di bawah ini **otomatis terhubung dengan dokumen dan kolom dari dataset yang Anda unggah/pilih di Tahap 1** beserta matriks fitur dari **Tahap 3**.
    """)

    # Tentukan rekomendasi default berdasarkan struktur kolom yang dipilih user di Tahap 1
    daftar_6_model = [
        "1. Sentiment Analysis (Klasifikasi Sentimen Dokumen)",
        "2. Spam Detection (Klasifikasi Pesan Spam vs Normal)",
        "3. Machine Translation (Penerjemahan Dokumen / Korpus Paralel)",
        "4. Text Summarization (Peringkasan Dokumen / Multi-Dokumen Korpus)",
        "5. Topic Modeling (Ekstraksi Topik LDA / NMF pada Korpus)",
        "6. Chatbot Retrieval (Tanya Jawab Berbasis Dataset / Dokumen Korpus)"
    ]

    if kolom_pasangan_user != "(Tidak Ada / Gunakan Simulasi Otomatis)":
        default_model_idx = 5 if "jawab" in kolom_pasangan_user.lower() else 2
    elif kolom_label_user != "(Tidak Ada / Gunakan Simulasi Otomatis)":
        nilai_unik_lbl = [str(x).lower() for x in df_sim[kolom_label_user].unique()]
        default_model_idx = 1 if any("spam" in x for x in nilai_unik_lbl) else 0
    else:
        default_model_idx = 4

    col_sel_mod1, col_sel_mod2 = st.columns([2, 1])
    with col_sel_mod1:
        model_pilihan_utama = st.selectbox(
            "🎯 Pilih Model NLP yang Ingin Dijalankan pada Dataset Anda:",
            daftar_6_model,
            index=default_model_idx
        )
    with col_sel_mod2:
        tampilkan_semua_tab = st.checkbox("Tampilkan Semua 6 Model Sekaligus dalam Tab", value=True)

    # Fungsi Pembantu Pelabelan Otomatis Berbasis Kamus (Jika Dataset Upload Tidak Punya Kolom Label)
    def auto_label_sentimen(teks_list, lang="Bahasa Indonesia"):
        pos_words = {"enak", "suka", "mantap", "juara", "murah", "ramah", "cepat", "rekomendasi", "nagih", "bagus", "puas", "lezat", "good", "great", "delicious", "love", "amazing", "affordable", "recommended"}
        neg_words = {"mahal", "keasinan", "lambat", "buruk", "kecewa", "mengecewakan", "dingin", "kapok", "kurang", "jelek", "parah", "bad", "hate", "slow", "salty", "overpriced", "terrible", "rude", "cold"}
        hasil_lbl = []
        for t in teks_list:
            toks = set(re.findall(r"\w+", str(t).lower()))
            skor = len(toks.intersection(pos_words)) - len(toks.intersection(neg_words))
            if lang == "Bahasa Indonesia":
                hasil_lbl.append("Positif" if skor >= 0 else "Negatif")
            else:
                hasil_lbl.append("Positive" if skor >= 0 else "Negative")
        return hasil_lbl

    def auto_label_spam(teks_list):
        spam_words = {"selamat", "hadiah", "undian", "juta", "klik", "link", "bit", "pinjol", "dana", "cair", "bunga", "agunan", "promo", "diskon", "gratis", "voucher", "kupon", "urgent", "diblokir", "otp", "pin", "kuota", "klaim", "menang", "free", "winner", "prize", "claim", "urgent"}
        hasil_lbl = []
        for t in teks_list:
            toks = set(re.findall(r"\w+", str(t).lower()))
            hasil_lbl.append("Spam" if len(toks.intersection(spam_words)) >= 1 else "Ham")
        return hasil_lbl

    # Fungsi Render masing-masing aplikasi berdasarkan df_sim aktif
    def render_sentiment_analysis():
        st.subheader("1. Sentiment Analysis (Berdasarkan Dataset Aktif)")
        st.write(f"Melatih model klasifikasi sentimen menggunakan kolom teks **`{kolom_teks}`** pada dataset Anda.")

        c_sa1, c_sa2 = st.columns([1, 2])
        with c_sa1:
            kandidat_kol_lbl = ["(Auto-Labeling Berbasis Lexicon / Edit Manual)"] + [c for c in df_sim.columns if c not in ["Kode_Doc", kolom_teks]]
            idx_def_sa = kandidat_kol_lbl.index(kolom_label_user) if kolom_label_user in kandidat_kol_lbl else 0
            kol_lbl_sa = st.selectbox("Pilih Kolom Target/Label Sentimen:", kandidat_kol_lbl, index=idx_def_sa, key="kol_lbl_sa")

            algo_sa = st.selectbox("Pilih Algoritma Klasifikasi:", ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"], key="algo_sa")
            fitur_sa = st.radio("Matriks Fitur Input:", ["TF-IDF Matrix (Tahap 3)", "Bag of Words Matrix (Tahap 3)"], horizontal=True, key="fit_sa")

        if kol_lbl_sa != "(Auto-Labeling Berbasis Lexicon / Edit Manual)":
            y_raw_sa = df_sim[kol_lbl_sa].astype(str).tolist()
        else:
            st.info("💡 Dataset Anda belum memilih kolom label. Sistem menerapkan **Auto-Labeling Sentimen** berbasis kata kunci positif/negatif (Anda juga dapat mengubah label langsung pada tabel di bawah):")
            y_raw_sa = auto_label_sentimen(df_sim[kolom_teks].tolist(), bahasa)

        with c_sa2:
            df_editor_sa = pd.DataFrame({
                "Kode_Doc": doc_labels,
                "Teks_Dokumen": df_sim[kolom_teks].tolist(),
                "Teks_Bersih": [r["teks_akhir"] for r in hasil_prep_list],
                "Label_Sentimen": y_raw_sa
            })
            df_edited_sa = st.data_editor(df_editor_sa, use_container_width=True, hide_index=True, key="editor_sa")
            y_train_sa = df_edited_sa["Label_Sentimen"].astype(str).tolist()

        if len(set(y_train_sa)) < 2:
            st.warning("⚠️ Dibutuhkan minimal 2 kelas label yang berbeda (misal: Positif dan Negatif) pada kolom `Label_Sentimen` untuk melatih model klasifikasi. Silakan ubah minimal 1 baris label pada tabel di atas.")
            return

        X_train_sa = tfidf_fin if fitur_sa.startswith("TF-IDF") else bow_mat
        model_sa = MultinomialNB() if algo_sa.startswith("Multinomial") else (LogisticRegression() if algo_sa.startswith("Logistic") else LinearSVC())
        model_sa.fit(X_train_sa, y_train_sa)
        pred_train_sa = model_sa.predict(X_train_sa)
        acc_sa = accuracy_score(y_train_sa, pred_train_sa) * 100

        st.metric("Akurasi Model pada Dataset Aktif", f"{acc_sa:.1f}%")

        st.markdown("#### 🧪 Uji Prediksi Sentimen pada Kalimat Baru")
        kal_uji_sa = st.text_input("Masukkan kalimat baru untuk diprediksi dengan model yang dilatih dari dataset Anda:", value=df_sim[kolom_teks].iloc[0], key="uji_sa_input")
        if kal_uji_sa:
            prep_u = jalankan_preprocessing(kal_uji_sa)
            toks_u = prep_u["6_stemming"]
            vec_u = np.array([[toks_u.count(w) for w in vocab]], dtype=float)
            if fitur_sa.startswith("TF-IDF"):
                vec_u = vec_u * idf_vec
                if opsi_l2 and np.linalg.norm(vec_u) > 0:
                    vec_u = vec_u / np.linalg.norm(vec_u)
            pred_u = model_sa.predict(vec_u)[0]
            st.success(f"🎯 **Hasil Prediksi Sentimen:** `{pred_u}` (Teks Bersih: `{prep_u['teks_akhir']}`)")

    def render_spam_detection():
        st.subheader("2. Spam Detection (Berdasarkan Dataset Aktif / Korpus SMS)")
        sumber_spam = st.radio(
            "Pilih Sumber Data Pelatihan Deteksi Spam:",
            [f"Gunakan Dataset Aktif Saat Ini (`{nama_file_aktif}` - Kolom `{kolom_teks}`)", "Gunakan Dataset Contoh SMS Spam Bawaan"],
            horizontal=True,
            key="src_spam"
        )

        if sumber_spam.startswith("Gunakan Dataset Aktif"):
            kandidat_kol_sp = ["(Auto-Tagging Indikator Spam / Edit Manual)"] + [c for c in df_sim.columns if c not in ["Kode_Doc", kolom_teks]]
            idx_def_sp = kandidat_kol_sp.index(kolom_label_user) if kolom_label_user in kandidat_kol_sp else 0
            kol_lbl_sp = st.selectbox("Pilih Kolom Label Spam/Ham pada Dataset Anda:", kandidat_kol_sp, index=idx_def_sp, key="kol_lbl_sp")

            if kol_lbl_sp != "(Auto-Tagging Indikator Spam / Edit Manual)":
                lbl_sp_awal = df_sim[kol_lbl_sp].astype(str).tolist()
            else:
                lbl_sp_awal = auto_label_spam(df_sim[kolom_teks].tolist())
                # Pastikan minimal ada 1 Spam dan 1 Ham agar bisa disimulasikan
                if len(set(lbl_sp_awal)) < 2 and len(lbl_sp_awal) >= 2:
                    lbl_sp_awal[0] = "Spam"
                    lbl_sp_awal[1] = "Ham"

            df_sp_edit = st.data_editor(pd.DataFrame({
                "Kode_Doc": doc_labels,
                "Pesan_Teks": df_sim[kolom_teks].tolist(),
                "Kategori": lbl_sp_awal
            }), use_container_width=True, hide_index=True, key="ed_spam")
            teks_latih_sp = [r["teks_akhir"] for r in hasil_prep_list]
            y_sp = df_sp_edit["Kategori"].astype(str).tolist()
        else:
            df_sp_bawaan = get_sample_dataframe(bahasa, "Spam Detection")
            st.dataframe(df_sp_bawaan, use_container_width=True, hide_index=True)
            teks_latih_sp = df_sp_bawaan["teks_pesan"].tolist()
            y_sp = df_sp_bawaan["label"].tolist()

        if len(set(y_sp)) < 2:
            st.warning("⚠️ Pastikan terdapat minimal 2 kategori (misal: `Spam` dan `Ham`) pada kolom Kategori.")
            return

        vec_sp = TfidfVectorizer(lowercase=True)
        X_sp = vec_sp.fit_transform(teks_latih_sp)
        clf_sp = MultinomialNB().fit(X_sp, y_sp)

        vocab_sp = vec_sp.get_feature_names_out()
        df_bobot_sp = pd.DataFrame({
            "Token Kata": vocab_sp,
            f"Log-Prob ({clf_sp.classes_[0]})": np.round(clf_sp.feature_log_prob_[0], 3),
            f"Log-Prob ({clf_sp.classes_[1]})": np.round(clf_sp.feature_log_prob_[1], 3)
        })
        st.markdown("**Tabel Bobot Probabilitas Token terhadap Kelas:**")
        st.dataframe(df_bobot_sp, use_container_width=True, hide_index=True)

        pesan_uji_sp = st.text_input("🧪 Masukkan teks pesan baru untuk diuji dengan Filter Spam:", value=df_sim[kolom_teks].iloc[0], key="uji_sp_in")
        if pesan_uji_sp:
            X_u_sp = vec_sp.transform([pesan_uji_sp.lower()])
            pred_u_sp = clf_sp.predict(X_u_sp)[0]
            prob_u_sp = clf_sp.predict_proba(X_u_sp)[0]
            info_prob = " | ".join([f"{k}: {v*100:.1f}%" for k, v in zip(clf_sp.classes_, prob_u_sp)])
            st.success(f"🔍 **Hasil Deteksi Pesan:** `{pred_u_sp}` ({info_prob})")

    def render_machine_translation():
        st.subheader("3. Machine Translation (Penerjemahan Dokumen Dataset / Korpus Paralel)")
        st.write(f"Menerjemahkan baris dokumen dari kolom **`{kolom_teks}`** pada dataset Anda atau mempelajari kamus dari pasangan kolom paralel.")

        kandidat_kol_target = ["(Gunakan Kamus Dwibahasa Bawaan)"] + [c for c in df_sim.columns if c not in ["Kode_Doc", kolom_teks]]
        idx_def_mt = kandidat_kol_target.index(kolom_pasangan_user) if kolom_pasangan_user in kandidat_kol_target else 0
        kol_target_mt = st.selectbox("Pilih Kolom Bahasa Target (Jika Dataset Anda adalah Korpus Paralel):", kandidat_kol_target, index=idx_def_mt, key="kol_mt")

        default_kamus_mt = (
            "saya: i, aku: i, sangat: really, suka: like, makan: eating, seblak: seblak crackers, "
            "pedas: spicy, dan: and, bakso: meatballs, enak: delicious, di: at, warung: shop, "
            "ini: this, harga: price, harganya: price, murah: cheap, mahal: expensive, pelayanan: service, "
            "cepat: fast, lambat: slow, kopi: coffee, mie: noodles, juara: champion, rekomendasi: recommended"
        )

        # Jika user memiliki kolom pasangan paralel, ekstrak pasangan kata sejajar secara otomatis ke kamus
        if kol_target_mt != "(Gunakan Kamus Dwibahasa Bawaan)":
            pasangan_auto = []
            for s_txt, t_txt in zip(df_sim[kolom_teks].astype(str), df_sim[kol_target_mt].astype(str)):
                s_toks = re.findall(r"\w+", s_txt.lower())
                t_toks = re.findall(r"\w+", t_txt.lower())
                for sw, tw in zip(s_toks, t_toks):
                    pasangan_auto.append(f"{sw}: {tw}")
            if pasangan_auto:
                default_kamus_mt = ", ".join(list(dict.fromkeys(pasangan_auto))) + ", " + default_kamus_mt

        kamus_mt_str = st.text_area("Kamus Translasi Aktif (`kata_sumber: kata_target`):", value=default_kamus_mt, height=90, key="kamus_mt_area")
        kamus_mt_dict = {}
        for item in kamus_mt_str.split(","):
            if ":" in item:
                k1, k2 = item.split(":", 1)
                kamus_mt_dict[k1.strip().lower()] = k2.strip().lower()

        def terjemahkan_kalimat(kal):
            toks = re.findall(r"\w+", str(kal).lower())
            return " ".join([kamus_mt_dict.get(w, w) for w in toks]).capitalize()

        df_hasil_mt = pd.DataFrame({
            "Kode_Doc": doc_labels,
            f"Teks Sumber ({kolom_teks})": df_sim[kolom_teks].tolist(),
            "Hasil Terjemahan Mesin": [terjemahkan_kalimat(t) for t in df_sim[kolom_teks]]
        })
        if kol_target_mt != "(Gunakan Kamus Dwibahasa Bawaan)":
            df_hasil_mt[f"Referensi Target ({kol_target_mt})"] = df_sim[kol_target_mt].tolist()

        st.markdown("**Tabel Hasil Penerjemahan Seluruh Dokumen pada Dataset Aktif:**")
        st.dataframe(df_hasil_mt, use_container_width=True, hide_index=True)

    def render_summarization():
        st.subheader("4. Text Summarization (Peringkasan Dokumen pada Dataset Aktif)")
        mode_sum = st.radio(
            "Pilih Mode Peringkasan dari Dataset Anda:",
            [
                f"A. Multi-Document Summarization (Cari Dokumen Paling Informatif dari Seluruh `{kolom_teks}`)",
                "B. Single-Document Summarization (Ringkas Kalimat dari 1 Dokumen Terpilih)"
            ],
            key="mode_sum"
        )

        if mode_sum.startswith("A."):
            st.write("Menghitung skor kepentingan setiap baris dokumen berdasarkan matriks **TF-IDF (Tahap 3)** untuk memilih dokumen utama yang merangkum keseluruhan isi korpus:")
            skor_dok = tfidf_fin.sum(axis=1)
            top_n_dok = st.slider("Pilih Jumlah Dokumen Inti untuk Ringkasan Korpus:", 1, len(df_sim), min(2, len(df_sim)), key="sl_sum_multi")
            idx_top_dok = sorted(np.argsort(skor_dok)[::-1][:top_n_dok])

            df_sum_korpus = pd.DataFrame({
                "Kode_Doc": doc_labels,
                "Isi Dokumen": df_sim[kolom_teks].tolist(),
                "Skor Informasi TF-IDF": np.round(skor_dok, 4),
                "Status": ["✅ Masuk Ringkasan Utama" if i in idx_top_dok else "❌ Dieliminasi" for i in range(len(df_sim))]
            })
            ringkasan_korpus = " ".join([df_sim[kolom_teks].iloc[i] for i in idx_top_dok])
            st.success(f"📝 **Hasil Ringkasan Multi-Dokumen:**\n\n{ringkasan_korpus}")
            st.dataframe(df_sum_korpus, use_container_width=True, hide_index=True)
        else:
            pilih_d_sum = st.selectbox("Pilih Dokumen dari Dataset:", range(len(df_sim)), format_func=lambda i: f"{doc_labels[i]}: {df_sim[kolom_teks].iloc[i][:70]}...", key="sel_doc_sum")
            teks_dok_sum = st.text_area("Teks Dokumen (Bisa Anda tambahkan beberapa kalimat untuk diuji):", value=df_sim[kolom_teks].iloc[pilih_d_sum], height=100, key="area_doc_sum")
            kal_list = [k.strip() for k in re.split(r"(?<=[.!?])\s+", teks_dok_sum.strip()) if k.strip()]
            if len(kal_list) < 2:
                st.info("ℹ️ Dokumen ini hanya terdiri dari 1 kalimat. Tambahkan minimal 2 kalimat yang dipisahkan tanda titik (`.`) pada kotak di atas atau gunakan **Mode A (Multi-Document Summarization)**.")
            else:
                n_kal = st.slider("Jumlah Kalimat Ringkasan:", 1, len(kal_list), 1, key="sl_sum_single")
                mat_s = TfidfVectorizer().fit_transform(kal_list).toarray()
                skor_s = mat_s.sum(axis=1)
                idx_s = sorted(np.argsort(skor_s)[::-1][:n_kal])
                st.success("📝 **Hasil Ringkasan:** " + " ".join([kal_list[i] for i in idx_s]))

    def render_topic_modeling():
        st.subheader("5. Topic Modeling (Ekstraksi Topik Tersembunyi dari Dataset Aktif)")
        st.write(f"Mengekstrak topik secara otomatis (*unsupervised*) dari matriks fitur kolom **`{kolom_teks}`** tanpa memerlukan kolom label.")

        c_tm1, c_tm2, c_tm3 = st.columns(3)
        with c_tm1:
            algo_tm = st.selectbox("Pilih Algoritma:", ["LDA (Latent Dirichlet Allocation)", "NMF (Non-Negative Matrix Factorization)"], key="algo_tm")
        with c_tm2:
            n_topik = st.slider("Jumlah Topik (K):", 2, max(2, min(6, len(vocab))), 2, key="k_tm")
        with c_tm3:
            n_kata_topik = st.slider("Jumlah Kata Kunci per Topik:", 2, max(2, min(10, len(vocab))), min(5, len(vocab)), key="w_tm")

        if algo_tm.startswith("LDA"):
            model_tm = LatentDirichletAllocation(n_components=n_topik, random_state=42)
            doc_topic_dist = model_tm.fit_transform(bow_mat)
        else:
            model_tm = NMF(n_components=n_topik, random_state=42, init="nndsvda")
            doc_topic_dist = model_tm.fit_transform(tfidf_fin)

        topik_rows = []
        for idx_top, comp in enumerate(model_tm.components_):
            top_indices = comp.argsort()[::-1][:n_kata_topik]
            topik_rows.append({
                "ID Topik": f"Topik {idx_top + 1}",
                "Kata Kunci Dominan": ", ".join([vocab[i] for i in top_indices]),
                "Bobot Kata": " | ".join([f"{vocab[i]} ({comp[i]:.2f})" for i in top_indices])
            })
        st.dataframe(pd.DataFrame(topik_rows), use_container_width=True, hide_index=True)

        norm_dist = doc_topic_dist / np.maximum(doc_topic_dist.sum(axis=1, keepdims=True), 1e-9)
        df_doc_topic = pd.DataFrame(np.round(norm_dist, 4), index=doc_labels, columns=[f"Proporsi Topik {i+1}" for i in range(n_topik)])
        df_doc_topic["Teks Dokumen"] = df_sim[kolom_teks].tolist()
        df_doc_topic["Topik Dominan"] = [f"Topik {i+1}" for i in np.argmax(norm_dist, axis=1)]
        st.markdown("**Hasil Pengelompokan Topik untuk Setiap Baris Dataset Anda:**")
        st.dataframe(df_doc_topic, use_container_width=True)

    def render_chatbot():
        st.subheader("6. Chatbot Retrieval Berbasis Dataset Anda (TF-IDF & Cosine Similarity)")
        st.write(f"Menjadikan dataset yang Anda unggah (`{nama_file_aktif}`) sebagai **Basis Pengetahuan (*Knowledge Base*) Chatbot**.")

        kandidat_kol_ans = ["(Gunakan Teks Dokumen Utama Sebagai Jawaban / Pencarian Korpus)"] + [c for c in df_sim.columns if c not in ["Kode_Doc", kolom_teks]]
        idx_def_cb = kandidat_kol_ans.index(kolom_pasangan_user) if kolom_pasangan_user in kandidat_kol_ans else 0
        kol_jawaban_cb = st.selectbox(
            "Pilih Kolom Jawaban Bot dari Dataset Anda (Pilih opsi pertama jika dataset hanya berisi 1 kolom teks):",
            kandidat_kol_ans,
            index=idx_def_cb,
            key="kol_cb"
        )

        pertanyaan_kb = df_sim[kolom_teks].astype(str).tolist()
        if kol_jawaban_cb == "(Gunakan Teks Dokumen Utama Sebagai Jawaban / Pencarian Korpus)":
            jawaban_kb = [f"[Dokumen {doc_labels[i]}] {df_sim[kolom_teks].iloc[i]}" for i in range(len(df_sim))]
        else:
            jawaban_kb = df_sim[kol_jawaban_cb].astype(str).tolist()

        df_kb_view = pd.DataFrame({
            "Kode": doc_labels,
            f"Teks Acuan Pencarian ({kolom_teks})": pertanyaan_kb,
            "Respons / Jawaban Chatbot": jawaban_kb
        })
        with st.expander("📚 Lihat Basis Pengetahuan Chatbot dari Dataset Aktif", expanded=False):
            st.dataframe(df_kb_view, use_container_width=True, hide_index=True)

        query_cb = st.text_input("💬 Ketik Pertanyaan / Kata Kunci untuk Mencari Jawaban dari Dataset Anda:", value=pertanyaan_kb[0], key="q_cb_input")
        if query_cb:
            vec_cb = TfidfVectorizer(lowercase=True)
            mat_kb = vec_cb.fit_transform([r["teks_akhir"] if r["teks_akhir"] else p.lower() for r, p in zip(hasil_prep_list, pertanyaan_kb)])
            prep_q = jalankan_preprocessing(query_cb)
            q_clean = prep_q["teks_akhir"] if prep_q["teks_akhir"] else query_cb.lower()
            mat_q = vec_cb.transform([q_clean])

            sims = cosine_similarity(mat_q, mat_kb)[0]
            best_i = int(np.argmax(sims))
            st.success(f"🤖 **Respons Chatbot (Kecocokan Tertinggi: `{doc_labels[best_i]}` — Cosine Similarity: `{sims[best_i]:.4f}`):**\n\n{jawaban_kb[best_i]}")

            df_rank_cb = pd.DataFrame({
                "Kode": doc_labels,
                "Teks Acuan di Dataset": pertanyaan_kb,
                "Respons Chatbot": jawaban_kb,
                "Skor Cosine Similarity": np.round(sims, 4)
            }).sort_values(by="Skor Cosine Similarity", ascending=False)
            st.dataframe(df_rank_cb, use_container_width=True, hide_index=True)

    if tampilkan_semua_tab:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1️⃣ Sentiment Analysis",
            "2️⃣ Spam Detection",
            "3️⃣ Machine Translation",
            "4️⃣ Text Summarization",
            "5️⃣ Topic Modeling",
            "6️⃣ Chatbot Dataset"
        ])
        with tab1:
            render_sentiment_analysis()
        with tab2:
            render_spam_detection()
        with tab3:
            render_machine_translation()
        with tab4:
            render_summarization()
        with tab5:
            render_topic_modeling()
        with tab6:
            render_chatbot()
    else:
        if model_pilihan_utama.startswith("1."):
            render_sentiment_analysis()
        elif model_pilihan_utama.startswith("2."):
            render_spam_detection()
        elif model_pilihan_utama.startswith("3."):
            render_machine_translation()
        elif model_pilihan_utama.startswith("4."):
            render_summarization()
        elif model_pilihan_utama.startswith("5."):
            render_topic_modeling()
        else:
            render_chatbot()

# ==========================================
# TAHAP 4: FITUR EXPORT KE .IPYNB (GOOGLE COLAB)
# ==========================================
st.markdown("---")
st.header("Tahap 4: Export ke Jupyter Notebook (`.ipynb`) Sesuai Dataset & Model")
st.markdown("""
File **`.ipynb`** yang dihasilkan di bawah ini akan otomatis memuat **dataset yang sedang aktif**, **nama kolom yang Anda pilih**, serta **kode pemodelan NLP** untuk dijalankan di **Google Colab**.
""")

col_exp1, col_exp2 = st.columns([2, 1])
with col_exp1:
    mode_notebook = st.radio(
        "Pilih Jenis File .ipynb yang Ingin Diunduh:",
        [
            "📘 Mode Tutorial Lengkap (Full Code Siap Jalan)",
            "📝 Mode Lembar Kerja Latihan Mahasiswa (Fill-in-the-Blank / Tugas)"
        ]
    )
    nama_mhs = st.text_input("Nama Mahasiswa / Kelas:", value="Mahasiswa Praktikum NLP")

def to_ipynb_lines(code_str):
    return [line + "\n" for line in code_str.strip().split("\n")]

def build_ipynb_file(is_worksheet=False, student_name="Mahasiswa"):
    kalimat_uji = df_sim[kolom_teks].iloc[0]
    data_dict_export = df_sim.to_dict(orient="list")
    kol_lbl_export = kolom_label_user if kolom_label_user != "(Tidak Ada / Gunakan Simulasi Otomatis)" else "label_auto"

    prep_code = f"""import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download("punkt")
nltk.download("punkt_tab")

kalimat_asli = {json.dumps(kalimat_uji, ensure_ascii=False)}

# 1. Cleansing
hasil_cleansing = re.sub(r"http\\S+|www\\.\\S+|@\\w+|#\\w+|\\d+|[^\\w\\s]", " ", kalimat_asli)
hasil_cleansing = re.sub(r"\\s+", " ", hasil_cleansing).strip()

# 2. Case Folding
hasil_case_folding = hasil_cleansing.lower()

# 3. Tokenization
hasil_tokenization = word_tokenize(hasil_case_folding)

# 4. Normalization
kamus_slang = {json.dumps(slang_dict, ensure_ascii=False)}
hasil_normalization = [kamus_slang.get(k, k) for k in hasil_tokenization if kamus_slang.get(k, k) != ""]

# 5. Stopword Removal
tambahan_stop = {json.dumps(list(custom_stops), ensure_ascii=False)}
stopwords_indo = set(StopWordRemoverFactory().get_stop_words()).union(set(tambahan_stop))
hasil_stopword = [k for k in hasil_normalization if k not in stopwords_indo]

# 6. Stemming
stemmer = StemmerFactory().create_stemmer()
hasil_stemming = [stemmer.stem(k) for k in hasil_stopword]

df_tahapan = pd.DataFrame({{
    "Tahapan": ["0. Asli", "1. Cleansing", "2. Case Folding", "3. Tokenization", "4. Normalization", "5. Stopword Removal", "6. Stemming"],
    "Hasil": [kalimat_asli, hasil_cleansing, hasil_case_folding, str(hasil_tokenization), str(hasil_normalization), str(hasil_stopword), str(hasil_stemming)]
}})
pd.set_option("display.max_colwidth", None)
display(df_tahapan)"""

    vector_code = """from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. Representasi Bag of Words (CountVectorizer)
count_vec = CountVectorizer()
bow_matrix = count_vec.fit_transform(df["teks_bersih"])
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out(), index=df["Kode_Doc"])
print("=== 1. TABEL BAG OF WORDS (COUNT VECTORIZER) ===")
display(df_bow)

# 2. Representasi TF-IDF (TfidfVectorizer)
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(df["teks_bersih"])
df_tfidf = pd.DataFrame(tfidf_matrix.toarray().round(4), columns=tfidf_vec.get_feature_names_out(), index=df["Kode_Doc"])
print("\\n=== 2. MATRIKS AKHIR TF-IDF ===")
display(df_tfidf)"""

    models_code = f"""# ==========================================
# TAHAP 5: PEMODELAN NLP SESUAI DATASET ({nama_file_aktif})
# ==========================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pastikan kolom label tersedia untuk model klasifikasi (Supervised)
if "{kol_lbl_export}" not in df.columns:
    df["{kol_lbl_export}"] = ["Positif" if i % 2 == 0 else "Negatif" for i in range(len(df))]

# 1. KLASIFIKASI TEKS (SENTIMENT ANALYSIS / SPAM DETECTION PADA DATASET)
print("=== 1. KLASIFIKASI TEKS DENGAN NAIVE BAYES ===")
if df["{kol_lbl_export}"].nunique() >= 2:
    clf = MultinomialNB().fit(tfidf_matrix, df["{kol_lbl_export}"])
    df["Prediksi_Model"] = clf.predict(tfidf_matrix)
    display(df[["Kode_Doc", "{kolom_teks}", "{kol_lbl_export}", "Prediksi_Model"]])

# 2. TOPIC MODELING (LDA) PADA DATASET
print("\\n=== 2. TOPIC MODELING (LDA) ===")
lda = LatentDirichletAllocation(n_components=2, random_state=42).fit(bow_matrix)
vocab_names = count_vec.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topik {{idx+1}}: {{', '.join([vocab_names[i] for i in topic.argsort()[:-6:-1]])}}")

# 3. MULTI-DOCUMENT SUMMARIZATION PADA DATASET
print("\\n=== 3. RINGKASAN DOKUMEN TERPENTING (TF-IDF SCORING) ===")
skor_dok = tfidf_matrix.toarray().sum(axis=1)
idx_utama = int(np.argmax(skor_dok))
print(f"Dokumen Paling Informatif ({{df['Kode_Doc'].iloc[idx_utama]}}): {{df['{kolom_teks}'].iloc[idx_utama]}}")

# 4. CHATBOT RETRIEVAL BERBASIS DATASET
print("\\n=== 4. UJI CHATBOT BERBASIS DATASET ===")
query_uji = df["{kolom_teks}"].iloc[0]
sim_scores = cosine_similarity(tfidf_vec.transform([fungsi_preprocessing(query_uji)]), tfidf_matrix)[0]
best_match = int(np.argmax(sim_scores))
print(f"Query: {{query_uji}}\\nDokumen Terdekat ({{df['Kode_Doc'].iloc[best_match]}} - Skor {{sim_scores[best_match]:.4f}}): {{df['{kolom_teks}'].iloc[best_match]}}")"""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": to_ipynb_lines(f"""# 🎓 Modul Latihan Praktikum NLP di Google Colab
**Nama / Kelompok:** `{student_name}`  
**Dataset Aktif:** `{nama_file_aktif}` (Kolom Teks: `{kolom_teks}`)  
**Model Pilihan:** `{model_pilihan_utama if SHOW_MODELING else "Preprocessing & TF-IDF"}`""")
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines("""!pip install -q Sastrawi nltk pandas openpyxl scikit-learn
print("✅ Instalasi selesai!")""")
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines(f"""import pandas as pd

# Memuat dataset yang disimulasikan di Streamlit (Atau gunakan pd.read_csv / pd.read_excel untuk file penuh)
data_aktif = {json.dumps(data_dict_export, ensure_ascii=False, indent=2)}
df = pd.DataFrame(data_aktif)
pd.set_option("display.max_colwidth", None)
display(df)""")
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines(prep_code)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines(f"""def fungsi_preprocessing(teks):
    clean = re.sub(r"http\\S+|www\\.\\S+|@\\w+|#\\w+|\\d+|[^\\w\\s]", " ", str(teks))
    clean = re.sub(r"\\s+", " ", clean).strip().lower()
    tokens = word_tokenize(clean)
    norm = [kamus_slang.get(k, k) for k in tokens if kamus_slang.get(k, k) != ""]
    no_stop = [k for k in norm if k not in stopwords_indo]
    stemmed = [stemmer.stem(k) for k in no_stop]
    return " ".join(stemmed)

df["teks_bersih"] = df["{kolom_teks}"].apply(fungsi_preprocessing)
display(df)""")
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines(vector_code)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_ipynb_lines(models_code)
        }
    ]

    notebook_json = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return json.dumps(notebook_json, indent=2, ensure_ascii=False).encode("utf-8")

is_ws = mode_notebook.startswith("📝")
ipynb_bytes = build_ipynb_file(is_worksheet=is_ws, student_name=nama_mhs)
nama_file_ipynb = "Lembar_Latihan_NLP_Colab.ipynb" if is_ws else "Modul_Lengkap_NLP_Colab.ipynb"

with col_exp2:
    st.markdown("#### Unduh File `.ipynb`")
    st.download_button(
        label=f"🚀 Export & Unduh {nama_file_ipynb}",
        data=ipynb_bytes,
        file_name=nama_file_ipynb,
        mime="application/x-ipynb+json",
        use_container_width=True
    )
    st.caption("Setelah diunduh, buka **Google Colab** → pilih **File > Upload notebook** → pilih file `.ipynb` ini.")

st.sidebar.markdown("---")
st.sidebar.subheader("Export ke Google Colab")
st.sidebar.download_button(
    label="⬇️ Export .ipynb (Colab)",
    data=ipynb_bytes,
    file_name=nama_file_ipynb,
    mime="application/x-ipynb+json",
    use_container_width=True
)