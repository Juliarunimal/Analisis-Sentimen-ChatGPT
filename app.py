import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ---------- Dataset Dummy ----------
data = {
    'text': [
        "Saya suka produk ini, sangat bagus!",
        "Ini pengalaman yang menyebalkan.",
        "Layanan sangat membantu dan cepat.",
        "Saya tidak akan beli lagi.",
        "Produk luar biasa, kualitas hebat.",
        "Buruk sekali, tidak sesuai ekspektasi.",
        "Saya puas dengan pembelian ini.",
        "Sangat mengecewakan dan mahal.",
        "Pengiriman cepat dan produk oke.",
        "Barang rusak dan lambat sampai."
    ],
    'label': ['positif', 'negatif', 'positif', 'negatif', 'positif',
              'negatif', 'positif', 'negatif', 'positif', 'negatif']
}

df = pd.DataFrame(data)

# ---------- Preprocessing ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# ---------- TF-IDF + Model ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# ---------- Fungsi Prediksi ----------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Analisis Sentimen", page_icon="🧠")

st.title("🧠 Analisis Sentimen dengan C5.0 (Decision Tree)")
st.write("Masukkan kalimat untuk mengetahui sentimennya:")

user_input = st.text_area("Teks Input", height=150)

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Masukkan kalimat terlebih dahulu.")
    else:
        result = predict_sentiment(user_input)
        if result == "positif":
            st.success(f"✅ Sentimen: **POSITIF**")
        else:
            st.error(f"❌ Sentimen: **NEGATIF**")
