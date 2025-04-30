import streamlit as st
import pandas as pd
import joblib
import base64

# Set page configuration for modern look
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for modern UI and card styling
st.markdown("""
    <style>
    /* General styling */
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    /* Title styling */
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #00d4b4;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2em;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Selectbox styling */
    .stSelectbox label {
        color: #ffffff;
        font-size: 1.1em;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 10px;
    }
    /* Button styling */
    .stButton button {
        background-color: #00d4b4;
        color: #1e1e1e;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #00b89a;
    }
    /* Card styling */
    .card {
        background-color: #2a2a2a;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.05);
    }
    .card-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #00d4b4;
        margin-bottom: 10px;
    }
    .card-text {
        font-size: 1em;
        color: #b0b0b0;
        margin: 5px 0;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2a2a2a;
    }
    .sidebar h3 {
        color: #00d4b4;
    }
    .sidebar p {
        color: #b0b0b0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_data():
    knn_model = joblib.load('model/best_knn_model.pkl')
    vectorizer = joblib.load('model/count_vectorizer.pkl')
    feature_df = pd.read_pickle('model/feature_df.pkl')
    base_df = pd.read_pickle('model/base_df.pkl')
    return knn_model, vectorizer, feature_df, base_df

knn_model, vectorizer, feature_df, base_df = load_data()

# Function to get recommendations
def content_recommender_ml(search_type, search_value, model, vectorizer, feature_data, base_data):
    try:
        if search_type == "Title":
            idx = feature_data.index[feature_data['title'] == search_value].tolist()[0]
            soup = feature_data.loc[idx, 'soup']
        else:
            # Cari indeks berdasarkan fitur lain
            if search_type == "Cast":
                idx = feature_data[feature_data['cast_name'].apply(lambda x: search_value.lower() in [i.lower() for i in x])].index
            elif search_type == "Genre":
                idx = feature_data[feature_data['genres'].apply(lambda x: search_value.lower() in [i.lower() for i in x])].index
            elif search_type == "Director":
                idx = feature_data[feature_data['director_name'].apply(lambda x: search_value.lower() in [i.lower() for i in x])].index
            elif search_type == "Writer":
                idx = feature_data[feature_data['writer_name'].apply(lambda x: search_value.lower() in [i.lower() for i in x])].index
            if len(idx) == 0:
                return pd.DataFrame({"Error": ["Tidak ada film yang cocok dengan kriteria."]})
            idx = idx[0]  # Ambil film pertama yang cocok
            soup = feature_data.loc[idx, 'soup']
        
        matrix = vectorizer.transform([soup])
        distances, indices = model.kneighbors(matrix, n_neighbors=11)
        movie_indices = indices[0][1:]  # Skip film itu sendiri
        return base_data.iloc[movie_indices][['title', 'genres', 'rating', 'cast_name', 'director_name']]
    except:
        return pd.DataFrame({"Error": ["Terjadi kesalahan atau input tidak ditemukan."]})

# Sidebar
with st.sidebar:
    st.markdown("<h3>Tentang Aplikasi</h3>", unsafe_allow_html=True)
    st.write(
        "Aplikasi ini adalah sistem rekomendasi film berbasis konten yang menggunakan "
        "informasi seperti genre, aktor, sutradara, dan penulis untuk menemukan film serupa yang mungkin Anda sukai."
    )
    st.markdown(
        "<p>Dibangun dengan Streamlit dan algoritma KNN untuk pengalaman rekomendasi yang cerdas dan personal.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 13px; color: gray;'>📦 @Project Akhir DQLab<br>👨‍💻 Created by Ahmads</p>",
        unsafe_allow_html=True
    )


# Main content
st.markdown("<div class='title'>Sistem Rekomendasi Film</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Cari film berdasarkan judul, genre, aktor, sutradara, atau penulis!</div>", unsafe_allow_html=True)

# Input: Dropdown for search type
search_type = st.selectbox("Pilih Jenis Pencarian:", ["Title", "Genre", "Cast", "Director", "Writer"], help="Pilih jenis kriteria untuk pencarian film.")


# Input: Dropdown for search value based on search type
if search_type == "Title":
    options = feature_df['title'].sort_values().tolist()
    selected_value = st.selectbox(f"Pilih {search_type}:", options, help=f"Pilih judul film untuk rekomendasi.")
elif search_type == "Genre":
    # Ambil semua genre unik
    genres = set()
    feature_df['genres'].apply(lambda x: genres.update(x))
    options = sorted(list(genres))
    selected_value = st.selectbox(f"Pilih {search_type}:", options, help=f"Pilih genre untuk rekomendasi.")
elif search_type == "Cast":
    # Ambil semua aktor unik
    cast = set()
    feature_df['cast_name'].apply(lambda x: cast.update(x))
    options = sorted(list(cast))
    selected_value = st.selectbox(f"Pilih {search_type}:", options, help=f"Pilih aktor untuk rekomendasi.")
elif search_type == "Director":
    # Ambil semua sutradara unik
    directors = set()
    feature_df['director_name'].apply(lambda x: directors.update(x))
    options = sorted(list(directors))
    selected_value = st.selectbox(f"Pilih {search_type}:", options, help=f"Pilih sutradara untuk rekomendasi.")
elif search_type == "Writer":
    # Ambil semua penulis unik
    writers = set()
    feature_df['writer_name'].apply(lambda x: writers.update(x))
    options = sorted(list(writers))
    selected_value = st.selectbox(f"Pilih {search_type}:", options, help=f"Pilih penulis untuk rekomendasi.")


# Button to trigger recommendation
if st.button("Tampilkan Rekomendasi"):
    with st.spinner("Menghasilkan rekomendasi..."):
        recommendations = content_recommender_ml(search_type, selected_value, knn_model, vectorizer, feature_df, base_df)
        if "Error" not in recommendations.columns:
            st.success(f"Rekomendasi untuk '{selected_value}' ({search_type}) berhasil dibuat!")
            # Format genres and cast for display
            recommendations['genres'] = recommendations['genres'].apply(lambda x: ', '.join(x))
            recommendations['cast_name'] = recommendations['cast_name'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            recommendations['director_name'] = recommendations['director_name'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
            # Display recommendations as cards
            st.markdown("### Hasil Rekomendasi")
            cols = st.columns(3)  # 3 cards per row
            for idx, row in recommendations.iterrows():
                with cols[idx % 3]:
                    st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>{row['title']}</div>
                            <div class='card-text'><b>Genre:</b> {row['genres']}</div>
                            <div class='card-text'><b>Rating:</b> {row['rating']}</div>
                            <div class='card-text'><b>Aktor:</b> {row['cast_name']}</div>
                            <div class='card-text'><b>Sutradara:</b> {row['director_name']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error(recommendations['Error'][0])

# Add a download button for recommendations
if 'recommendations' in locals() and "Error" not in recommendations.columns:
    csv = recommendations.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="recommendations_{selected_value.replace(' ', '_')}.csv">Unduh Rekomendasi</a>'
    st.markdown(href, unsafe_allow_html=True)