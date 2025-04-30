# Sistem Rekomendasi Film

Sebuah ****Content-Based Film Recommendation System**** yang dibangun menggunakan Streamlit dan algoritma K-Nearest Neighbors (KNN) untuk merekomendasikan film berdasarkan fitur seperti judul, genre, aktor, sutradara, dan penulis.

## Fitur
- Cari film berdasarkan **judul**, **genre**, **aktor**, **sutradara**, atau **penulis**.
- Menampilkan rekomendasi dalam tampilan modern berbasis kartu lengkap dengan detail (judul, genre, rating, pemeran, sutradara).
- Unduh rekomendasi dalam format file CSV.
- Desain responsif dengan tema gelap dan tampilan kustom.

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama.
- **Streamlit**: Framework web untuk antarmuka pengguna.
- **Pandas**: Manipulasi dan pra-pemrosesan data.
- **Scikit-learn**: Model KNN dan CountVectorizer untuk ekstraksi fitur.
- **Joblib**: Menyimpan dan memuat model serta data.

## Instalasi

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd film-recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Unduh model dan file data**:
   - Tempatkan `best_knn_model.pkl`, `count_vectorizer.pkl`, `feature_df.pkl`, dan `base_df.pkl` di direktori `model/`.
   - File-file ini dihasilkan dari skrip praproses dan pemodelan (lihat [Buku Catatan Pemodelan](#)).

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Penggunaan
1. Buka aplikasi di peramban Anda (default: `http://localhost:8501`).
2. Pilih jenis pencarian (Judul, Genre, Pemeran, Sutradara, atau Penulis) dari menu dropdown.
3. Pilih nilai tertentu (misalnya, "The Lion King" untuk Judul atau "Action" untuk Genre).
4. Klik **"Tampilkan Rekomendasi"** untuk melihat rekomendasi film.
5. Unduh rekomendasi sebagai file CSV menggunakan tautan yang disediakan.

## Cara Kerjanya
- **Praproses Data**: Menggabungkan rating film, pemeran, sutradara, dan data penulis dari kumpulan data CSV.
- **Rekayasa Fitur**: Membuat fitur "lengkap" dengan menggabungkan pemeran, genre, sutradara, dan penulis menjadi satu rangkaian teks.
- **Pemodelan**: Menggunakan CountVectorizer untuk memvektorisasi fitur dan KNN dengan kesamaan kosinus guna menemukan film yang serupa.
- **UI**: Streamlit menyediakan antarmuka interaktif dengan CSS khusus untuk tampilan modern.

## Datasets
- `movie_rating_df.csv`: Peringkat film dan metadata.
- `directors_writers.csv`: Informasi sutradara dan penulis.
- `actor_name.csv`: Nama aktor dan judul film terkait.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Built with ❤️ using Streamlit and Scikit-learn.
- Thanks to DQLab for providing the datasets.
```
