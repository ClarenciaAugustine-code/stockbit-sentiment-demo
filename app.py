import streamlit as st
import joblib
import torch
import numpy as np
import re
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import os

HF_USERNAME   = "ClarenciaAugustine"    
HF_REPO_ID    = f"{HF_USERNAME}/stockbit-sentiment-models"
FINBERT_NAME  = "michaelmanurung/finbert-indonesia"
ROBERTA_PATH  = f"{HF_USERNAME}/stockbit-indoroberta"

st.set_page_config(
    page_title="Demo Sentimen Stockbit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #0A1F44; margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem; color: #475569; margin-bottom: 1.5rem;
    }
    .result-card {
        background: #F8FAFC; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin: 0.5rem 0;
        border-left: 5px solid #1C7293;
    }
    .result-card.winner {
        background: #ECFDF5; border-left-color: #10B981;
    }
    .model-name {
        font-weight: 700; font-size: 1rem; color: #0A1F44;
    }
    .label-badge {
        display: inline-block; padding: 0.2rem 0.8rem;
        border-radius: 20px; font-weight: 700;
        font-size: 0.9rem; margin-left: 0.5rem;
    }
    .badge-positif  { background: #D1FAE5; color: #065F46; }
    .badge-netral   { background: #FEF3C7; color: #92400E; }
    .badge-negatif  { background: #FEE2E2; color: #991B1B; }
    .consensus-box {
        background: #0A1F44; color: white;
        border-radius: 12px; padding: 1rem 1.5rem;
        margin-top: 1rem; text-align: center;
    }
    .stTextArea textarea { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing ──────────────────────────
SLANG_DICT = {
    'tp': 'ambil untung', 'cl': 'jual rugi', 'cuan': 'untung',
    'nyangkut': 'rugi', 'avg': 'rata rata', 'dyor': 'riset sendiri',
    'hoki': 'untung', 'haka': 'beli', 'serok': 'beli',
    'boncos': 'rugi', 'bel': 'beli', 'arok': 'beli',
    'pompom': 'hasut', 'fomo': 'ikut', 'bullish': 'naik',
    'bearish': 'turun', 'sl': 'jual rugi', 'nyicil': 'beli',
    'guyur': 'jual', 'ara': 'auto reject atas', 'arb': 'auto reject bawah',
}

@st.cache_resource
def load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer  = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

def preprocess_tfidf(text):
    stemmer, stopword = load_sastrawi()
    text = text.lower()
    text = re.sub(r'[\$\#][a-z0-9]+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-z\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    text  = ' '.join([SLANG_DICT.get(w, w) for w in words])
    text  = stopword.remove(text)
    text  = stemmer.stem(text)
    return text

def preprocess_transformer(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

LABEL_MAP   = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_COLOR = {'Negatif': 'badge-negatif', 'Netral': 'badge-netral', 'Positif': 'badge-positif'}
LABEL_EMOJI = {'Negatif': '🔴', 'Netral': '🟡', 'Positif': '🟢'}

# ── Model Loading (cached) ──────────────────
@st.cache_resource(show_spinner="⏳ Loading TF-IDF + SVM + RF...")
def load_classical():
    tfidf = joblib.load(hf_hub_download(HF_REPO_ID, "tfidf_vectorizer_5k_last.pkl"))
    svm   = joblib.load(hf_hub_download(HF_REPO_ID, "svm_linear_tfidf_fix.pkl"))
    rf    = joblib.load(hf_hub_download(HF_REPO_ID, "rf_tfidf_fix.pkl"))
    return tfidf, svm, rf

@st.cache_resource(show_spinner="⏳ Loading FinBERT + SVM...")
def load_finbert():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok      = AutoTokenizer.from_pretrained(FINBERT_NAME)
    model    = AutoModel.from_pretrained(FINBERT_NAME).to(device)
    model.eval()
    svm_head = joblib.load(hf_hub_download(HF_REPO_ID, "svm_finbert_model.pkl"))
    return tok, model, svm_head, device

@st.cache_resource(show_spinner="⏳ Loading Indo-RoBERTa fine-tuned...")
def load_roberta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    model  = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH).to(device)
    model.eval()
    return tok, model, device

# ── Prediction functions ────────────────────
def predict_svm(text, tfidf, svm):
    vec  = tfidf.transform([preprocess_tfidf(text)])
    pred = int(svm.predict(vec)[0])
    if hasattr(svm, 'decision_function'):
        scores = svm.decision_function(vec)[0]
        if np.ndim(scores) == 0:
            scores = np.array([scores, -scores, 0])
        conf = float(softmax(scores).max())
    else:
        conf = 1.0
    return LABEL_MAP.get(pred, str(pred)), conf

def predict_rf(text, tfidf, rf):
    vec   = tfidf.transform([preprocess_tfidf(text)])
    probs = rf.predict_proba(vec)[0]
    return LABEL_MAP.get(int(np.argmax(probs))), float(probs.max())

def predict_finbert(text, tok, model, svm_head, device):
    inputs = tok(preprocess_transformer(text), return_tensors='pt',
                 padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    pred = int(svm_head.predict(emb)[0])
    if hasattr(svm_head, 'decision_function'):
        scores = svm_head.decision_function(emb)[0]
        if np.ndim(scores) == 0:
            scores = np.array([scores, -scores, 0])
        conf = float(softmax(scores).max())
    else:
        conf = 1.0
    return LABEL_MAP.get(pred, str(pred)), conf

def predict_roberta(text, tok, model, device):
    inputs = tok(preprocess_transformer(text), return_tensors='pt',
                 padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return LABEL_MAP.get(int(np.argmax(probs))), float(probs.max())

# ── UI ─────────────────────────────────────
st.markdown('<p class="main-title">📈 Demo Analisis Sentimen Stockbit</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Kelompok 15 — Natural Language Processing LH01 · BINUS University</p>',
    unsafe_allow_html=True
)

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Model Klasifikasi", "4")
with col_b:
    st.metric("Best F1-W Internal", "0.93")
with col_c:
    st.metric("Best F1-W OOD", "0.59")
with col_d:
    st.metric("Data Stockbit", "4.997")

st.divider()

st.markdown("### 💬 Masukkan komentar Stockbit")

EXAMPLES = [
    "BUMI naik terus nih, cuan banyak gw 🚀",
    "duh ELSA nyangkut dalem, boncos parah hari ini",
    "AADI mengumumkan akan melakukan stock split bulan depan",
    "mantap sekali INET turun terus, makin murah buat dijual",
    "ada yang tau alasan DEWA gap down pagi ini?",
]

col1, col2 = st.columns([3, 1])

with col1:
    teks = st.text_area(
        label="Teks komentar",
        placeholder="Contoh: BUMI ARA terus nih, cuan banget!",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**💡 Coba contoh:**")
    for ex in EXAMPLES:
        if st.button(ex[:40] + "...", key=ex, use_container_width=True):
            teks = ex
            st.session_state["teks_input"] = ex

predict_btn = st.button("🔍 Prediksi Sentimen", type="primary", use_container_width=True)

if predict_btn and teks.strip():
    st.markdown("### 📊 Hasil Prediksi")

    with st.spinner("Loading models..."):
        tfidf, svm, rf           = load_classical()
        fin_tok, fin_model, svm_head, fin_dev = load_finbert()
        rob_tok, rob_model, rob_dev           = load_roberta()

    results = {}
    with st.spinner("Menjalankan prediksi..."):
        results["TF-IDF + SVM"]           = predict_svm(teks, tfidf, svm)
        results["TF-IDF + Random Forest"] = predict_rf(teks, tfidf, rf)
        results["FinBERT + SVM"]          = predict_finbert(teks, fin_tok, fin_model, svm_head, fin_dev)
        results["Indo-RoBERTa ⭐"]         = predict_roberta(teks, rob_tok, rob_model, rob_dev)

    for model_name, (label, conf) in results.items():
        is_winner = "⭐" in model_name
        card_class = "result-card winner" if is_winner else "result-card"
        badge_class = LABEL_COLOR[label]
        emoji = LABEL_EMOJI[label]
        bar_pct = int(conf * 100)

        st.markdown(f"""
        <div class="{card_class}">
            <span class="model-name">{model_name}</span>
            <span class="label-badge {badge_class}">{emoji} {label}</span>
            <div style="margin-top:0.5rem; color:#475569; font-size:0.9rem;">
                Confidence: <strong>{conf:.1%}</strong>
                <div style="background:#E2E8F0; border-radius:4px; height:8px; margin-top:4px;">
                    <div style="background:{'#10B981' if label=='Positif' else '#EF4444' if label=='Negatif' else '#F59E0B'};
                                width:{bar_pct}%; height:8px; border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Konsensus
    from collections import Counter
    labels     = [v[0] for v in results.values()]
    konsensus  = Counter(labels).most_common(1)[0]
    rob_label, rob_conf = results["Indo-RoBERTa ⭐"]

    st.markdown(f"""
    <div class="consensus-box">
        🎯 <strong>Konsensus:</strong> {LABEL_EMOJI[konsensus[0]]} {konsensus[0]}
        ({konsensus[1]}/4 model setuju) &nbsp;|&nbsp;
        ⭐ <strong>Indo-RoBERTa:</strong> {LABEL_EMOJI[rob_label]} {rob_label}
        ({rob_conf:.1%} confidence)
    </div>
    """, unsafe_allow_html=True)

elif predict_btn and not teks.strip():
    st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

# Footer info
st.divider()
with st.expander("ℹ️ Tentang model ini"):
    st.markdown("""
    | Model | Internal F1-W | OOD F1-W |
    |---|---|---|
    | TF-IDF + SVM | 0.81 | 0.50 |
    | TF-IDF + Random Forest | 0.79 | 0.33 |
    | FinBERT + SVM | 0.46 | 0.38 |
    | **Indo-RoBERTa Finansial ⭐** | **0.93** | **0.59** |

    **Qwen 2.5-7B-Instruct** berperan sebagai *validation labeler* (bukan model klasifikasi) —
    digunakan untuk men-generate label pada data Stockbit OOD dengan verifikasi manual.

    **Preprocessing:** lowercase → hapus ticker/hashtag → normalisasi slang Stockbit
    → stopword removal + stemming Sastrawi (khusus model TF-IDF).
    """)
