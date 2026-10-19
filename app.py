import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import io
import json

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
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ==========================================
# KONFIGURASI HALAMAN & FLAG DOSEN
# ==========================================
st.set_page_config(
    page_title="Modul Praktikum NLP & Generator Colab (.ipynb)",
    page_icon="🎓",
    layout="wide"
)

# Ubah ke True pada pertemuan berikutnya untuk membuka bagian Pemodelan
SHOW_MODELING = False

st.title("🎓 Laboratorium Virtual NLP & Generator Latihan Google Colab")
st.markdown("""
Aplikasi ini dirancang untuk **mensimulasikan tahapan NLP secara interaktif** sekaligus **menghasilkan file latihan Jupyter Notebook (`.ipynb`)** yang terintegrasi dengan pengaturan dan dataset Anda untuk dijalankan di **Google Colab**.
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
            en_stops = set(nltk_stopwords.words('english'))
        except Exception:
            en_stops = {"the", "a", "an", "in", "on", "of", "and", "is", "are", "was", "it", "to", "for"}
        return ps, lemma, en_stops
    return None, None, set()

stemmer_id, stopwords_id_base = load_sastrawi()
stemmer_en, lemmatizer_en, stopwords_en_base = load_nltk()

def get_sample_dataframe(lang="Bahasa Indonesia"):
    if lang == "Bahasa Indonesia":
        return pd.DataFrame({
            "id_dokumen": ["D1", "D2", "D3", "D4", "D5"],
            "teks_ulasan": [
                "Wah, seblak di warung ini SANGAT pedas sekali...!!! 🥵 Aku suka bangettt, harganya 10ribu @kuliner_tasik https://seblak.com",
                "Seblak pedas itu enak bgt, kuahnya kental dan kerupuknya empuk mantulll 100%!",
                "Pelayanan di warung ini agak lambat, tapi rasa seblak dan bakso pedasnya juara 👍",
                "Kurang suka sama bakso di sini, kuahnya keasinan dan harganya 25ribu terlalu mahal :(",
                "Rekomendasi banget buat pecinta kuliner pedas! Seblak dan mie pedasnya bikin nagih terus."
            ],
            "label": ["Positif", "Positif", "Positif", "Negatif", "Positif"]
        })
    else:
        return pd.DataFrame({
            "id_dokumen": ["D1", "D2", "D3", "D4", "D5"],
            "teks_ulasan": [
                "Wow, the coffee at this shop is REALLY bad...!!! 😠 I absolutely hate itttt, costs $10 @NYCafe https://coffee.com",
                "The spicy noodles and hot coffee here are super delicious and affordable!",
                "Service at this shop was quite slow, waiting 30 mins, but the coffee tastes great 👍",
                "I really hate the food here, too salty and overpriced at $25 :(",
                "Highly recommended for coffee lovers! Their espresso and pastries are amazing."
            ],
            "label": ["Negative", "Positive", "Positive", "Negative", "Positive"]
        })

# ==========================================
# SIDEBAR: NAVIGASI & PENGATURAN BAHASA
# ==========================================
st.sidebar.header("Alur Praktikum NLP")
st.sidebar.markdown("""
1. **Tahap 1:** Pembacaan Data (`TXT`/`CSV`/`Excel`)
2. **Tahap 2:** *Text Preprocessing*
3. **Tahap 3:** Representasi Numerik (`BoW` & `TF-IDF`)
4. **Tahap 4:** Export `.ipynb` ke Google Colab
5. **Tahap 5:** Pemodelan *(Hidden)*
""")
st.sidebar.markdown("---")
bahasa = st.sidebar.radio("Pilih Bahasa Dataset:", ["Bahasa Indonesia", "Bahasa Inggris"])

# ==========================================
# TAHAP 1: PEMBACAAN DATA (TXT, CSV, EXCEL)
# ==========================================
st.header("Tahap 1: Pembacaan Dataset (TXT, CSV, Excel)")

col_src, col_tpl = st.columns([2, 1])
with col_src:
    sumber_data = st.radio(
        "Pilih Sumber Data Simulasi:",
        ["Gunakan Dataset Contoh", "Unggah Dataset Sendiri (.txt, .csv, .xlsx)"],
        horizontal=True
    )

with col_tpl:
    with st.expander("Unduh File Dataset Contoh (.txt / .csv / .xlsx)"):
        df_tpl = get_sample_dataframe(bahasa)
        st.download_button(
            "📄 Unduh dataset_contoh.txt",
            data="\n".join(df_tpl["teks_ulasan"].tolist()).encode("utf-8"),
            file_name="dataset_contoh.txt",
            mime="text/plain"
        )
        st.download_button(
            "Unduh dataset_contoh.csv",
            data=df_tpl.to_csv(index=False).encode("utf-8"),
            file_name="dataset_contoh.csv",
            mime="text/csv"
        )
        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
            df_tpl.to_excel(writer, index=False, sheet_name="Sheet1")
        st.download_button(
            "Unduh dataset_contoh.xlsx",
            data=buf_xlsx.getvalue(),
            file_name="dataset_contoh.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

df_raw = None
kolom_teks = "teks_ulasan"
nama_file_aktif = "dataset_contoh.csv"
tipe_file_aktif = "csv"
delimiter_aktif = ","
sheet_aktif = "Sheet1"

if sumber_data == "Gunakan Dataset Contoh":
    df_raw = get_sample_dataframe(bahasa)
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
                kolom_teks = st.selectbox("Pilih Nama Kolom Teks:", df_raw.columns)
            elif ext in ["xlsx", "xls"]:
                tipe_file_aktif = "excel"
                xls = pd.ExcelFile(uploaded)
                sheet_aktif = st.selectbox("Pilih Nama Sheet Excel:", xls.sheet_names)
                df_raw = pd.read_excel(uploaded, sheet_name=sheet_aktif)
                kolom_teks = st.selectbox("Pilih Nama Kolom Teks:", df_raw.columns)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

if df_raw is None:
    st.info("Silakan unggah file terlebih dahulu atau gunakan Dataset Contoh.")
    st.stop()

df_raw = df_raw.dropna(subset=[kolom_teks]).reset_index(drop=True)
df_raw[kolom_teks] = df_raw[kolom_teks].astype(str)

tab_data_sim, tab_data_colab = st.tabs([
    "Simulasi Pratinjau Dataset",
    "Tutorial Kode Pembacaan Data di Google Colab"
])

with tab_data_sim:
    max_baris = len(df_raw)
    if max_baris > 3:
        jml_dok = st.slider(
            "Jumlah baris dokumen yang disimulasikan di layar (agar tabel matriks mudah diamati):",
            min_value=2, max_value=min(50, max_baris), value=min(5, max_baris)
        )
        df_sim = df_raw.iloc[:jml_dok].copy().reset_index(drop=True)
    else:
        df_sim = df_raw.copy().reset_index(drop=True)

    doc_labels = [f"D{i+1}" for i in range(len(df_sim))]
    df_sim.insert(0, "Kode_Doc", doc_labels)
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

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
df = pd.read_csv('{nama_file_aktif if tipe_file_aktif=="csv" else "dataset_contoh.csv"}', sep='{delimiter_aktif}', encoding='utf-8')

# 2B. Membaca File Excel (.xlsx)
# df = pd.read_excel('{nama_file_aktif if tipe_file_aktif=="excel" else "dataset_contoh.xlsx"}', sheet_name='{sheet_aktif}')

# 2C. Membaca File TXT (1 baris = 1 dokumen)
# with open('{nama_file_aktif if tipe_file_aktif=="txt" else "dataset_contoh.txt"}', 'r', encoding='utf-8') as f:
#     baris_teks = [line.strip() for line in f if line.strip()]
# df = pd.DataFrame({{'{kolom_teks}': baris_teks}})

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
        "bangettt: sekali, bgt: sekali, banget: sekali, mantulll: mantap, yg: yang, gak: tidak, sama: dengan, buat: untuk, ribu: "
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
    t0 = teks
    t1 = t0
    if use_clean_url:
        t1 = re.sub(r'http\S+|www\.\S+|@\w+|#\w+', ' ', t1)
    if use_clean_num:
        t1 = re.sub(r'\d+', ' ', t1)
    if use_clean_punct:
        t1 = re.sub(r'[^\w\s]', ' ', t1)
    t1 = re.sub(r'\s+', ' ', t1).strip()

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

nltk.download('punkt')
nltk.download('punkt_tab')

kamus_slang = {json.dumps(slang_dict, ensure_ascii=False)}
stopwords_indo = set(StopWordRemoverFactory().get_stop_words()).union({set(custom_stops)})
stemmer = StemmerFactory().create_stemmer()

def tahapan_preprocessing(teks):
    clean = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', str(teks))
    clean = re.sub(r'\s+', ' ', clean).strip()
    case_fold = clean.lower()
    tokens = word_tokenize(case_fold)
    norm = [kamus_slang.get(k, k) for k in tokens if kamus_slang.get(k, k) != ""]
    no_stop = [k for k in norm if k not in stopwords_indo]
    stemmed = [stemmer.stem(k) for k in no_stop]
    return " ".join(stemmed)

df['teks_bersih'] = df['{kolom_teks}'].apply(tahapan_preprocessing)
pd.set_option('display.max_colwidth', None)
display(df[['{kolom_teks}', 'teks_bersih']].head())""", language="python")

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
bow_matrix = count_vectorizer.fit_transform(df['teks_bersih'])
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out(),
    index=[f"D{i+1}" for i in range(len(df))]
)
display(df_bow)

# 2. Perhitungan TF-IDF (TfidfVectorizer)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['teks_bersih'])
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray().round(4),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=[f"D{i+1}" for i in range(len(df))]
)
display(df_tfidf)""", language="python")

# ==========================================
# TAHAP 4: FITUR EXPORT KE .IPYNB (GOOGLE COLAB)
# ==========================================
st.markdown("---")
st.header("Tahap 4: Export ke Jupyter Notebook (`.ipynb`) untuk Latihan Google Colab")
st.markdown("""
Anda dapat mengekspor seluruh alur praktikum di atas beserta dataset yang sedang aktif menjadi file **`.ipynb`**. Tersedia **2 mode notebook** yang bisa dipilih:
* **Mode Tutorial Lengkap (Full Code):** Seluruh kode terisi lengkap dan siap dijalankan (*Run All*) di Google Colab.
* **Mode Lembar Kerja Latihan Mahasiswa (Soal Praktikum):** Sebagian baris kode dikosongkan (`# TODO: ...`) agar mahasiswa dapat berlatih melengkapi kode sendiri di Google Colab.
""")

col_exp1, col_exp2 = st.columns([2, 1])
with col_exp1:
    mode_notebook = st.radio(
        "Pilih Jenis File .ipynb yang Ingin Diunduh:",
        [
            "Mode Tutorial Lengkap (Full Code Siap Jalan)",
            "Mode Lembar Kerja Latihan Mahasiswa (Fill-in-the-Blank / Tugas)"
        ]
    )
    nama_mhs = st.text_input("Nama Mahasiswa / Kelas (Opsional - akan tertulis di header Notebook):", value="Mahasiswa Praktikum NLP")

def build_ipynb_file(is_worksheet=False, student_name="Mahasiswa"):
    kalimat_uji = df_sim[kolom_teks].iloc[0]
    list_dokumen = df_sim[kolom_teks].tolist()

    if not is_worksheet:
        cell_prep_single = [
            "# ==========================================",
            "# BAGIAN A: SIMULASI PREPROCESSING 1 KALIMAT DALAM TABEL",
            "# ==========================================",
            "import re",
            "import nltk",
            "import pandas as pd",
            "from nltk.tokenize import word_tokenize",
            "from Sastrawi.Stemmer.StemmerFactory import StemmerFactory",
            "from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory",
            "nltk.download('punkt')",
            "nltk.download('punkt_tab')",
            f"kalimat_asli = {json.dumps(kalimat_uji, ensure_ascii=False)}",
            "# 1. Cleansing",
            "hasil_cleansing = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', kalimat_asli)",
            "hasil_cleansing = re.sub(r'\s+', ' ', hasil_cleansing).strip()",
            "# 2. Case Folding",
            "hasil_case_folding = hasil_cleansing.lower()",
            "# 3. Tokenization",
            "hasil_tokenization = word_tokenize(hasil_case_folding)",
            "# 4. Normalization",
            f"kamus_slang = {json.dumps(slang_dict, ensure_ascii=False)}",
            "hasil_normalization = [kamus_slang.get(k, k) for k in hasil_tokenization if kamus_slang.get(k, k) != '']",
            "# 5. Stopword Removal",
            f"tambahan_stop = {json.dumps(list(custom_stops), ensure_ascii=False)}",
            "stopwords_indo = set(StopWordRemoverFactory().get_stop_words()).union(set(tambahan_stop))",
            "hasil_stopword = [k for k in hasil_normalization if k not in stopwords_indo]",
            "# 6. Stemming",
            "stemmer = StemmerFactory().create_stemmer()",
            "hasil_stemming = [stemmer.stem(k) for k in hasil_stopword]",
            "df_tahapan = pd.DataFrame({",
            "    'Tahapan': ['0. Asli', '1. Cleansing', '2. Case Folding', '3. Tokenization', '4. Normalization', '5. Stopword Removal', '6. Stemming'],",
            "    'Hasil': [kalimat_asli, hasil_cleansing, hasil_case_folding, str(hasil_tokenization), str(hasil_normalization), str(hasil_stopword), str(hasil_stemming)]",
            "})",
            "pd.set_option('display.max_colwidth', None)",
            "display(df_tahapan)"
        ]

        cell_vector = [
            "# ==========================================",
            "# BAGIAN C: PERHITUNGAN MANUAL & SCIKIT-LEARN (BoW, TF, IDF, TF-IDF)",
            "# ==========================================",
            "import numpy as np",
            "import math",
            "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer",
            "# 1. Perhitungan menggunakan CountVectorizer (Bag of Words)",
            "count_vec = CountVectorizer()",
            "bow_matrix = count_vec.fit_transform(df['teks_bersih'])",
            "df_bow = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out(), index=[f'D{i+1}' for i in range(len(df))])",
            "print('=== 1. TABEL BAG OF WORDS (COUNT VECTORIZER) ===')",
            "display(df_bow)",
            "# 2. Perhitungan menggunakan TfidfVectorizer",
            "tfidf_vec = TfidfVectorizer()",
            "tfidf_matrix = tfidf_vec.fit_transform(df['teks_bersih'])",
            "df_idf = pd.DataFrame({'Term (Kata)': tfidf_vec.get_feature_names_out(), 'Nilai IDF': tfidf_vec.idf_.round(4)})",
            "print('\n=== 2. TABEL NILAI IDF ===')",
            "display(df_idf)",
            "df_tfidf = pd.DataFrame(tfidf_matrix.toarray().round(4), columns=tfidf_vec.get_feature_names_out(), index=[f'D{i+1}' for i in range(len(df))])",
            "print('\n=== 3. MATRIKS AKHIR TF-IDF ===')",
            "display(df_tfidf)"
        ]
    else:
        cell_prep_single = [
            "# ==========================================",
            "# LATIHAN 1: LENGKAPI TAHAPAN PREPROCESSING KALIMAT BERIKUT",
            "# ==========================================",
            "import re",
            "import nltk",
            "import pandas as pd",
            "from nltk.tokenize import word_tokenize",
            "from Sastrawi.Stemmer.StemmerFactory import StemmerFactory",
            "from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory",
            "nltk.download('punkt')",
            "nltk.download('punkt_tab')",
            f"kalimat_asli = {json.dumps(kalimat_uji, ensure_ascii=False)}",
            "# TODO 1: Lakukan Cleansing dengan Regex untuk menghapus URL, Mention, Angka, dan Tanda Baca",
            "hasil_cleansing = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', kalimat_asli)",
            "hasil_cleansing = re.sub(r'\s+', ' ', hasil_cleansing).strip()",
            "# TODO 2: Ubah hasil_cleansing menjadi huruf kecil semua (Case Folding)",
            "hasil_case_folding = hasil_cleansing.lower() # Silakan amati atau modifikasi",
            "# TODO 3: Pecah kalimat menjadi token menggunakan word_tokenize()",
            "hasil_tokenization = word_tokenize(hasil_case_folding)",
            "# TODO 4: Tambahkan kata tidak baku yang Anda temukan ke dalam kamus_slang berikut",
            f"kamus_slang = {json.dumps(slang_dict, ensure_ascii=False)}",
            "hasil_normalization = [kamus_slang.get(k, k) for k in hasil_tokenization if kamus_slang.get(k, k) != '']",
            "# TODO 5: Hapus stopwords bahasa Indonesia",
            f"tambahan_stop = {json.dumps(list(custom_stops), ensure_ascii=False)}",
            "stopwords_indo = set(StopWordRemoverFactory().get_stop_words()).union(set(tambahan_stop))",
            "hasil_stopword = [k for k in hasil_normalization if k not in stopwords_indo]",
            "# TODO 6: Lakukan stemming menggunakan Sastrawi",
            "stemmer = StemmerFactory().create_stemmer()",
            "hasil_stemming = [stemmer.stem(k) for k in hasil_stopword]",
            "df_tahapan = pd.DataFrame({",
            "    'Tahapan': ['0. Asli', '1. Cleansing', '2. Case Folding', '3. Tokenization', '4. Normalization', '5. Stopword Removal', '6. Stemming'],",
            "    'Hasil': [kalimat_asli, hasil_cleansing, hasil_case_folding, str(hasil_tokenization), str(hasil_normalization), str(hasil_stopword), str(hasil_stemming)]",
            "})",
            "pd.set_option('display.max_colwidth', None)",
            "display(df_tahapan)"
        ]

        cell_vector = [
            "# ==========================================",
            "# LATIHAN 2: EKSTRAKSI FITUR BAG OF WORDS & TF-IDF",
            "# ==========================================",
            "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer",
            "# TODO A: Inisialisasi CountVectorizer() dan terapkan fit_transform pada kolom df['teks_bersih']",
            "count_vec = CountVectorizer()",
            "bow_matrix = count_vec.fit_transform(df['teks_bersih'])",
            "df_bow = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out(), index=[f'D{i+1}' for i in range(len(df))])",
            "print('=== 1. TABEL BAG OF WORDS ===')",
            "display(df_bow)",
            "# TODO B: Inisialisasi TfidfVectorizer() dan terapkan fit_transform pada kolom df['teks_bersih']",
            "tfidf_vec = TfidfVectorizer()",
            "tfidf_matrix = tfidf_vec.fit_transform(df['teks_bersih'])",
            "df_tfidf = pd.DataFrame(tfidf_matrix.toarray().round(4), columns=tfidf_vec.get_feature_names_out(), index=[f'D{i+1}' for i in range(len(df))])",
            "print('\n=== 2. MATRIKS TF-IDF ===')",
            "display(df_tfidf)"
        ]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#Modul Latihan Praktikum NLP di Google Colab",
                f"**Nama / Kelompok:** `{student_name}`  ",
                f"**Bahasa Dataset:** `{bahasa}`  ",
                "**Topik Praktikum:** Pembacaan Dataset (`TXT`, `CSV`, `Excel`), *Text Preprocessing*, dan Representasi Numerik (`Bag of Words` & `TF-IDF`).",
                "---",
                "### Petunjuk Penggunaan di Google Colab:",
                "1. Jalankan setiap *cell* secara berurutan dari atas ke bawah dengan menekan tombol **Play (▶️)** atau **`Shift + Enter`**.",
                "2. Pada bagian pembacaan dataset, Anda dapat menggunakan data yang sudah tertanam di dalam *notebook* ini atau mengunggah file `.txt`, `.csv`, dan `.xlsx` Anda sendiri."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Tahap 0: Instalasi Library NLP"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q Sastrawi nltk pandas openpyxl scikit-learn",
                "print('Instalasi selesai!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tahap 1: Pembacaan Dataset (`TXT`, `CSV`, `Excel`)",
                "Secara *default*, *cell* di bawah ini memuat dataset yang Anda simulasikan di aplikasi Streamlit. Jika ingin membaca file eksternal (`.txt`, `.csv`, `.xlsx`), hapus tanda pagar (`#`) pada bagian Opsi Upload File."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd",
                "# --- OPSI 1: MENGGUNAKAN DATASET DARI SIMULASI STREAMLIT ---",
                f"data_simulasi = {json.dumps(list_dokumen, ensure_ascii=False, indent=2)}",
                f"df = pd.DataFrame({{'Kode_Doc': [f'D{{i+1}}' for i in range(len(data_simulasi))], '{kolom_teks}': data_simulasi}})",
                "# --- OPSI 2: JIKA INGIN UPLOAD FILE SENDIRI (.CSV / .XLSX / .TXT) DI COLAB ---",
                "# from google.colab import files",
                "# uploaded = files.upload()",
                "# df = pd.read_csv('nama_file.csv', sep=',')",
                "# df = pd.read_excel('nama_file.xlsx', sheet_name='Sheet1')",
                "# with open('nama_file.txt', 'r', encoding='utf-8') as f:",
                "#     df = pd.DataFrame({'teks_ulasan': [line.strip() for line in f if line.strip()]})",
                "pd.set_option('display.max_colwidth', None)",
                "display(df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tahap 2A: Simulasi Tabel Preprocessing pada 1 Kalimat",
                "Menampilkan perubahan kalimat langkah demi langkah dalam bentuk tabel DataFrame."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_prep_single
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tahap 2B: Penerapan Preprocessing ke Seluruh Dokumen Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def fungsi_preprocessing(teks):",
                "    clean = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', str(teks))",
                "    clean = re.sub(r'\s+', ' ', clean).strip().lower()",
                "    tokens = word_tokenize(clean)",
                "    norm = [kamus_slang.get(k, k) for k in tokens if kamus_slang.get(k, k) != '']",
                "    no_stop = [k for k in norm if k not in stopwords_indo]",
                "    stemmed = [stemmer.stem(k) for k in no_stop]",
                "    return ' '.join(stemmed)",
                f"df['teks_bersih'] = df['{kolom_teks}'].apply(fungsi_preprocessing)",
                "display(df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tahap 3: Representasi Numerik (`Bag of Words` & `TF-IDF`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_vector
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Tahap 4: Pemodelan Machine Learning *(Disembunyikan untuk Pertemuan Berikutnya)*",
                "Bagian pemodelan klasifikasi teks (seperti *Naive Bayes*, *SVM*, atau *IndoBERT*) sengaja dikunci pada modul ini agar Anda fokus memahami cara kerja *preprocessing* dan pembentukan matriks **BoW / TF-IDF**."
            ]
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
    st.markdown("####Unduh File `.ipynb`")
    st.download_button(
        label=f"Export & Unduh {nama_file_ipynb}",
        data=ipynb_bytes,
        file_name=nama_file_ipynb,
        mime="application/x-ipynb+json",
        use_container_width=True
    )
    st.caption("Setelah diunduh, buka **Google Colab** → pilih **File > Upload notebook** → pilih file `.ipynb` ini.")

# Tambahkan juga tombol Export Cepat di Sidebar agar selalu mudah diakses
st.sidebar.markdown("---")
st.sidebar.subheader("Export ke Google Colab")
st.sidebar.download_button(
    label="Export .ipynb (Colab)",
    data=ipynb_bytes,
    file_name=nama_file_ipynb,
    mime="application/x-ipynb+json",
    use_container_width=True
)

# ==========================================
# TAHAP 5: PEMODELAN ML (DI-HIDDEN)
# ==========================================
st.markdown("---")
if SHOW_MODELING:
    st.header("Tahap 5: Pemodelan Klasifikasi Teks")
else:
    st.info("**Tahap 5: Pemodelan Machine Learning (Klasifikasi Teks)** saat ini dikunci (*hidden*). Silakan unduh file `.ipynb` di atas dan selesaikan latihan hingga tahap matriks **Bag of Words & TF-IDF** di Google Colab.")