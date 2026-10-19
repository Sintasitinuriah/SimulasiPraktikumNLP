import streamlit as st
import pandas as pd
import numpy as np
import re
import math

# Konfigurasi Halaman
st.set_page_config(
    page_title="Simulasi BoW & TF-IDF - NLP",
    page_icon="📊",
    layout="wide"
)

st.title("Simulasi Interaktif Bag of Words (BoW) & TF-IDF")
st.markdown("""
Aplikasi ini membantu memvisualisasikan cara kerja ekstraksi fitur teks menggunakan **Count Vectorizer (Bag of Words)** dan **TF-IDF** langkah demi langkah, mulai dari teks mentah hingga menjadi matriks angka.
""")

# --- SIDEBAR: PENGATURAN & INPUT ---
st.sidebar.header("Pengaturan Simulasi")

mode_rumus = st.sidebar.radio(
    "Pilih Versi Rumus TF-IDF:",
    [
        "1. Rumus Teori Dasar (Manual/Buku)",
        "2. Rumus Scikit-Learn (Default Python)"
    ],
    help="Mahasiswa sering bingung mengapa hasil hitungan manual berbeda dengan library Scikit-Learn. Gunakan opsi ini untuk melihat perbedaannya."
)

if mode_rumus == "1. Rumus Teori Dasar (Manual/Buku)":
    tipe_tf = st.sidebar.selectbox(
        "Variasi Rumus TF:",
        ["Raw Count (Frekuensi Asli)", "Normalized TF (Frekuensi / Total Kata di Dokumen)"]
    )
    pakai_l2 = st.sidebar.checkbox("Gunakan L2 Normalization di akhir?", value=False)
else:
    st.sidebar.info("Pada mode Scikit-Learn, TF menggunakan Raw Count, IDF menggunakan smoothing natural log, dan hasil akhir dinormalisasi dengan L2 Norm.")
    tipe_tf = "Raw Count (Frekuensi Asli)"
    pakai_l2 = True

st.sidebar.markdown("---")
st.sidebar.subheader("Opsi Preprocessing")
hapus_stopword = st.sidebar.checkbox("Hapus Stopwords Sederhana (dan, di, yang, ini, itu)", value=False)

# --- INPUT DOKUMEN (KORPUS) ---
st.subheader("1. Input Korpus (Kumpulan Dokumen)")
st.caption("Masukkan satu dokumen (kalimat) per baris. Anda bisa mengedit atau menambah kalimat di bawah ini:")

default_corpus = """Saya suka makan seblak pedas
Seblak pedas itu sangat enak sekali
Saya suka makan bakso dan seblak"""

corpus_input = st.text_area("Korpus Teks:", value=default_corpus, height=120)

# Memisahkan baris menjadi list dokumen
raw_docs = [doc.strip() for doc in corpus_input.split("\n") if doc.strip() != ""]
N = len(raw_docs)

if N == 0:
    st.warning("Silakan masukkan minimal 1 kalimat dokumen.")
    st.stop()

doc_labels = [f"D{i+1}" for i in range(N)]

# --- PROSES PREPROCESSING & TOKENISASI ---
stopwords_list = {"dan", "di", "yang", "ini", "itu", "ke", "dari"}

def preprocess(text):
    # Case folding & hapus tanda baca/angka
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    if hapus_stopword:
        tokens = [t for t in tokens if t not in stopwords_list]
    return tokens

tokenized_docs = [preprocess(doc) for doc in raw_docs]

# Membuat Vocabulary unik (diurutkan sesuai abjad seperti CountVectorizer)
vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))

if len(vocab) == 0:
    st.error("Tidak ada kata yang tersisa setelah preprocessing.")
    st.stop()

# --- PERHITUNGAN MATRIKS ---
# 1. Matriks BoW (Raw Count)
bow_matrix = np.zeros((N, len(vocab)), dtype=int)
for i, doc in enumerate(tokenized_docs):
    for j, word in enumerate(vocab):
        bow_matrix[i, j] = doc.count(word)

df_bow = pd.DataFrame(bow_matrix, index=doc_labels, columns=vocab)

# 2. Matriks TF
tf_matrix = np.zeros((N, len(vocab)), dtype=float)
for i in range(N):
    total_words = len(tokenized_docs[i])
    for j in range(len(vocab)):
        if tipe_tf == "Raw Count (Frekuensi Asli)":
            tf_matrix[i, j] = bow_matrix[i, j]
        else:
            tf_matrix[i, j] = bow_matrix[i, j] / total_words if total_words > 0 else 0

df_tf = pd.DataFrame(tf_matrix, index=doc_labels, columns=vocab)

# 3. Perhitungan DF dan IDF
df_counts = np.sum(bow_matrix > 0, axis=0) # Jumlah dokumen yang mengandung kata t
idf_values = np.zeros(len(vocab), dtype=float)

for j, df_val in enumerate(df_counts):
    if mode_rumus == "1. Rumus Teori Dasar (Manual/Buku)":
        # Rumus standar: log10(N / df)
        idf_values[j] = math.log10(N / df_val)
    else:
        # Rumus Scikit-Learn (smooth_idf=True): ln((1 + N) / (1 + df)) + 1
        idf_values[j] = math.log((1 + N) / (1 + df_val)) + 1

df_idf_table = pd.DataFrame({
    "Kata (Term)": vocab,
    "Document Frequency (DF)": df_counts,
    "Nilai IDF": np.round(idf_values, 4)
})

# 4. Perhitungan TF-IDF
tfidf_raw = tf_matrix * idf_values

if pakai_l2:
    # L2 Normalization: membagi setiap baris dengan akar dari jumlah kuadrat elemen baris tersebut
    norms = np.linalg.norm(tfidf_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Menghindari pembagian dengan nol
    tfidf_final = tfidf_raw / norms
else:
    tfidf_final = tfidf_raw

df_tfidf_raw = pd.DataFrame(np.round(tfidf_raw, 4), index=doc_labels, columns=vocab)
df_tfidf_final = pd.DataFrame(np.round(tfidf_final, 4), index=doc_labels, columns=vocab)

# --- TAMPILAN TABS STEP-BY-STEP ---
st.markdown("---")
st.subheader(" 2. Alur Perhitungan Bertahap")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tahap 1: Tokenisasi & Vocab",
    "Tahap 2: Bag of Words (BoW)",
    "Tahap 3: Term Frequency (TF)",
    "Tahap 4: DF & IDF",
    "Tahap 5: Hasil TF-IDF & Simulasi Manual"
])

# TAB 1: TOKENISASI & VOCABULARY
with tab1:
    st.markdown("### Tahap 1: Preprocessing, Tokenisasi, dan Pembentukan Vocabulary")
    st.write("Setiap kalimat diubah menjadi huruf kecil (*case folding*), dihapus tanda bacanya, lalu dipecah menjadi daftar kata (*token*).")
    
    df_prep = pd.DataFrame({
        "Kode": doc_labels,
        "Kalimat Asli": raw_docs,
        "Hasil Tokenisasi": [str(tokens) for tokens in tokenized_docs],
        "Jumlah Kata": [len(tokens) for tokens in tokenized_docs]
    })
    st.dataframe(df_prep, use_container_width=True, hide_index=True)
    
    st.markdown(f"**Daftar Vocabulary Unik ({len(vocab)} kata):**")
    st.info(", ".join([f"`{w}`" for w in vocab]))

# TAB 2: BAG OF WORDS
with tab2:
    st.markdown("### Tahap 2: Matriks Bag of Words (Count Vectorizer)")
    st.write("""
    **Bag of Words (BoW)** atau *Count Vectorizer* menghitung berapa kali setiap kata dalam *Vocabulary* muncul pada masing-masing dokumen. Urutan kata dalam kalimat diabaikan.
    """)
    st.dataframe(df_bow.style.highlight_max(axis=None, color='#d4edda'), use_container_width=True)
    st.caption("Angka di dalam tabel menunjukkan jumlah kemunculan kata (kolom) pada dokumen terkait (baris).")

# TAB 3: TERM FREQUENCY (TF)
with tab3:
    st.markdown("### Tahap 3: Perhitungan Term Frequency (TF)")
    if tipe_tf == "Raw Count (Frekuensi Asli)":
        st.latex(r"TF(t, d) = f_{t, d} \text{ (Jumlah kemunculan kata } t \text{ di dokumen } d\text{)}")
        st.write("Pada pengaturan ini, nilai TF sama persis dengan matriks **Bag of Words (Raw Count)**.")
    else:
        st.latex(r"TF(t, d) = \frac{f_{t, d}}{\text{Total kata dalam dokumen } d}")
        st.write("Nilai frekuensi kata dibagi dengan panjang (jumlah kata) dokumen agar dokumen yang panjang tidak mendominasi dokumen pendek.")
    
    st.dataframe(df_tf.style.format("{:.4f}"), use_container_width=True)

# TAB 4: DF & IDF
with tab4:
    st.markdown("### Tahap 4: Document Frequency (DF) & Inverse Document Frequency (IDF)")
    st.write("""
    * **DF (Document Frequency):** Jumlah dokumen yang memuat kata tersebut (bukan total kemunculan kata).
    * **IDF (Inverse Document Frequency):** Mengukur seberapa penting/unik suatu kata. Kata yang muncul di **semua dokumen** akan memiliki nilai IDF yang **kecil**, sedangkan kata yang jarang muncul memiliki IDF **besar**.
    """)
    
    col_rumus1, col_rumus2 = st.columns(2)
    with col_rumus1:
        st.markdown("**Rumus yang Sedang Aktif:**")
        if mode_rumus == "1. Rumus Teori Dasar (Manual/Buku)":
            st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{DF(t)}\right)")
            st.caption(f"Di mana $N = {N}$ (total dokumen). Jika suatu kata muncul di semua dokumen ($DF = N$), maka $\\log_{{10}}(1) = 0$.")
        else:
            st.latex(r"IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1")
            st.caption("Scikit-Learn menambahkan angka 1 (*smoothing*) agar kata yang muncul di semua dokumen tidak bernilai 0 (tetap memiliki bobot = 1).")
            
    st.dataframe(df_idf_table, use_container_width=True, hide_index=True)

# TAB 5: TF-IDF & BEDAH HITUNGAN
with tab5:
    st.markdown("### Tahap 5: Matriks Akhir TF-IDF")
    st.latex(r"TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)")
    
    if pakai_l2:
        st.write("**A. Matriks TF-IDF Sebelum Normalisasi L2 ($TF \\times IDF$):**")
        st.dataframe(df_tfidf_raw.style.format("{:.4f}"), use_container_width=True)
        
        st.write("**B. Matriks TF-IDF Akhir (Setelah L2 Normalization - Standar Scikit-Learn):**")
        st.latex(r"v_{\text{norm}} = \frac{v}{\sqrt{v_1^2 + v_2^2 + \dots + v_n^2}}")
        st.dataframe(df_tfidf_final.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
    else:
        st.write("**Matriks TF-IDF Akhir ($TF \\times IDF$):**")
        st.dataframe(df_tfidf_final.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)

    st.markdown("---")
    st.subheader("Bedah Hitungan Manual per Kata (Inspektor Rumus)")
    st.write("Pilih **Dokumen** dan **Kata** di bawah ini untuk melihat bagaimana angka pada tabel di atas didapatkan:")
    
    col_pilih1, col_pilih2 = st.columns(2)
    with col_pilih1:
        pilih_doc_idx = st.selectbox("Pilih Dokumen:", range(N), format_func=lambda x: f"D{x+1}: '{raw_docs[x]}'")
    with col_pilih2:
        pilih_kata = st.selectbox("Pilih Kata (Term):", vocab)
        
    kata_idx = vocab.index(pilih_kata)
    
    # Ambil nilai untuk bedah hitungan
    f_td = bow_matrix[pilih_doc_idx, kata_idx]
    total_w = len(tokenized_docs[pilih_doc_idx])
    tf_val = tf_matrix[pilih_doc_idx, kata_idx]
    df_val = df_counts[kata_idx]
    idf_val = idf_values[kata_idx]
    raw_tfidf_val = tfidf_raw[pilih_doc_idx, kata_idx]
    final_tfidf_val = tfidf_final[pilih_doc_idx, kata_idx]
    
    st.markdown(f"#### Detail Perhitungan Kata **`{pilih_kata}`** pada Dokumen **`D{pilih_doc_idx+1}`**:")
    
    # 1. Detail TF
    if tipe_tf == "Raw Count (Frekuensi Asli)":
        st.markdown(f"1. **Hitung TF:** Kata `{pilih_kata}` muncul sebanyak **{f_td} kali** di D{pilih_doc_idx+1}. Maka $\\text{{TF}} = {f_td}$.")
    else:
        st.markdown(f"1. **Hitung TF:** Kata `{pilih_kata}` muncul **{f_td} kali** dari total **{total_w} kata** di D{pilih_doc_idx+1}.")
        st.latex(rf"TF = \frac{{{f_td}}}{{{total_w}}} = {tf_val:.4f}")
        
    # 2. Detail IDF
    st.markdown(f"2. **Hitung IDF:** Kata `{pilih_kata}` muncul di **{df_val} dokumen** dari total **{N} dokumen** ($DF = {df_val}, N = {N}$).")
    if mode_rumus == "1. Rumus Teori Dasar (Manual/Buku)":
        st.latex(rf"IDF = \log_{{10}}\left(\frac{{{N}}}{{{df_val}}}\right) = {idf_val:.4f}")
    else:
        st.latex(rf"IDF = \ln\left(\frac{{1 + {N}}}{{1 + {df_val}}}\right) + 1 = \ln\left(\frac{{{1+N}}}{{{1+df_val}}}\right) + 1 = {idf_val:.4f}")
        
    # 3. Detail TF x IDF
    st.markdown("3. **Kalikan TF dengan IDF:**")
    st.latex(rf"TF\text{{-}}IDF_{{\text{{mentah}}}} = {tf_val:.4f} \times {idf_val:.4f} = {raw_tfidf_val:.4f}")
    
    # 4. Detail L2 Norm jika aktif
    if pakai_l2:
        row_squares = " + ".join([f"({val:.4f})^2" for val in tfidf_raw[pilih_doc_idx] if val > 0])
        norm_val = np.linalg.norm(tfidf_raw[pilih_doc_idx])
        st.markdown(f"4. **Normalisasi L2 (Membagi dengan panjang vektor D{pilih_doc_idx+1}):**")
        if norm_val > 0:
            st.latex(rf"\text{{Panjang Vektor }} D{pilih_doc_idx+1} = \sqrt{{{row_squares}}} = {norm_val:.4f}")
            st.latex(rf"TF\text{{-}}IDF_{{\text{{akhir}}}} = \frac{{{raw_tfidf_val:.4f}}}{{{norm_val:.4f}}} = \mathbf{{{final_tfidf_val:.4f}}}")
        else:
            st.write("Semua elemen bernilai 0, sehingga hasil akhir = 0.")