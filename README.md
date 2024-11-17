### **KITALULUS TECHNICAL TEXT**

#### **1. Struktur Folder**

Pastikan semua file terorganisir seperti ini:

```
kitalulus/
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv
├── models/
│   ├── recommendation_model.pkl
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── model_trainer.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── common.py
│   │   ├── logger.py
│   │   ├── exception.py
│   │   └── __init__.py
│   ├── web/
│   │   ├── api/
│   │   │   ├── routes.py
│   │   │   └── __init__.py
│   │   └── streamlit/
│   │       ├── app.py
│   │       └── __init__.py
├── README.md
└── requirements.txt
```

---

#### **2. Installasi Dependensi**

Tambahkan dependensi berikut ke file `requirements.txt`:

```plaintext
numpy>=1.24.3
pandas>=2.0.0
scikit-learn>=1.2.2
imbalanced-learn>=0.10.1
joblib>=1.2.0
nltk>=3.8.1
spacy>=3.5.2


tensorflow>=2.12.0
transformers>=4.28.1
fastapi>=0.95.1
uvicorn>=0.21.1
python-multipart>=0.0.6
requests>=2.28.2
pydantic>=1.10.7
streamlit>=1.22.0
plotly>=5.14.1
streamlit-option-menu>=0.3.2
streamlit-extras>=0.2.7
matplotlib>=3.7.1
seaborn>=0.12.2
pyyaml>=6.0

-e .
```

Instal dependensi dengan menjalankan:

```bash
pip install -r requirements.txt
```

---

#### **3. Melatih Model**

Jalankan pipeline model_trainer:

```bash
python src/components/model_trainer.py
```

Pipeline ini akan:

1. Membaca data dari folder `data/`.
2. Membersihkan teks menggunakan **NLTK**.
3. Melakukan transformasi data menggunakan **TF-IDF Vectorizer**.
4. Melatih beberapa model (Random Forest, Logistic Regression, dll.) untuk **rekomendasi** dan **analisis sentimen**.
5. Menyimpan model terbaik di folder `models/`:
   - `recommendation_model.pkl`
   - `sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`

---

#### **4. Menjalankan API**

API berbasis **FastAPI** dapat digunakan untuk prediksi review. Jalankan perintah:

```bash
uvicorn src.web.api.routes:app --reload
```

#### **Endpoint API**

- **URL**: `http://127.0.0.1:8000/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "I bought this lovely silk/velvet shirt in the &quot;sky&quot; color but it is more on the teal blue side than sky blue, which disappointed me. it is definitely darker than appears in photo. still a luxurious well-made beauty with sassy appeal. it drapes like a snake slithering down your body. it comes with attitude."
  }
  ```
- **Response**:
  ```json
  {
    "recommendation": true,
    "sentiment": 5,
    "recommendation_confidence": 0.95,
    "sentiment_confidence": 0.87,
    "recommendation_probabilities": {
      "Not Recommended": 0.05,
      "Recommended": 0.95
    },
    "sentiment_probabilities": {
      "Rating 1": 0.02,
      "Rating 2": 0.03,
      "Rating 3": 0.05,
      "Rating 4": 0.15,
      "Rating 5": 0.75
    },
    "cleaned_text": "bought lovely silk velvet shirt quot sky quot color teal blue side sky blue disappointed definitely darker appears photo still luxurious well made beauty sassy appeal drape like snake slithering body come attitude"
  }
  ```

---

#### **5. Menjalankan Dashboard Streamlit**

Untuk menampilkan visualisasi data dan menggunakan prediksi interaktif, jalankan Streamlit:

```bash
streamlit run src/web/streamlit/app.py
```

**Fitur di Streamlit**:

1. **Summary**: Halaman eksplorasi data (EDA).
2. **Prediction**: Masukkan teks ulasan dan dapatkan prediksi.
3. **Model Evaluation**: Evaluasi performa model, termasuk metrik seperti akurasi, precision, recall, F1-score, dll.
4. **Advanced Analysis**: Analisis atribut seperti qaulity, size, comfort, dan topik utama menggunakan LDA.

---

#### **6. Fitur Tambahan**

##### **Analisis Kesalahan (Error Analysis)**

Pada halaman evaluasi, Anda dapat melihat:

- **False Positive**: Review yang salah diprediksi sebagai "Recommended".
- **False Negative**: Review yang salah diprediksi sebagai "Not Recommended".

##### **Pentingnya Fitur**

Anda dapat melihat fitur (kata) mana yang paling berpengaruh pada keputusan model:

- **Recommendation**: Fitur yang penting untuk prediksi rekomendasi.
- **Sentiment**: Fitur yang penting untuk prediksi sentimen.

##### **Model Terbaik**

Pipeline akan memilih model terbaik berdasarkan:

- **F1-Score** dan **Cross-Validation** untuk rekomendasi.
- **Akurasi** dan **RMSE** untuk analisis sentimen.

---
