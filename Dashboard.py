import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Mengatur layout Streamlit
st.set_page_config(page_title="Bike Sharing Dashboard", page_icon="🚲", layout="wide")

# 1. LOAD DATA & PREPROCESSING
@st.cache_data
def load_data():
    # Pastikan file cleaned_hour.csv ada di direktori yang sama
    df = pd.read_csv("cleaned_hour.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# 2. SIDEBAR (FILTER)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=150)
    st.title("Filter Data")
    
    # Filter Rentang Tanggal
    min_date = df['date'].min()
    max_date = df['date'].max()
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Terapkan filter ke dataframe utama
main_df = df[(df["date"] >= str(start_date)) & (df["date"] <= str(end_date))]

# 3. HEADER & METRIC DASHBOARD
st.title("🚲 Bike Sharing Analytics Dashboard")
st.markdown("Dashboard ini menganalisis pola peminjaman sepeda berdasarkan faktor lingkungan, waktu, dan hari kerja.")

col1, col2, col3 = st.columns(3)
with col1:
    total_rentals = main_df['total_count'].sum()
    st.metric("Total Penyewaan", value=f"{total_rentals:,}")
with col2:
    total_registered = main_df['registered'].sum()
    st.metric("Total Pengguna Terdaftar", value=f"{total_registered:,}")
with col3:
    total_casual = main_df['casual'].sum()
    st.metric("Total Pengguna Kasual", value=f"{total_casual:,}")

st.markdown("---")

# 4. VISUALISASI JAWABAN PERTANYAAN BISNIS
st.subheader("1. Pengaruh Cuaca Terhadap Peminjaman (2011 vs 2012)")
weather_summary = main_df.groupby(by=["year", "weather_condition"]).agg({"total_count": "mean"}).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="weather_condition", y="total_count", hue="year", data=weather_summary, palette=["#D3D3D3", "#72BCD4"], ax=ax)
ax.set_title("Rata-rata Peminjaman Berdasarkan Cuaca", fontsize=15)
ax.set_ylabel("Rata-rata Penyewaan")
ax.set_xlabel("Kondisi Cuaca")
st.pyplot(fig)

st.subheader("2. Tren Peminjaman Berdasarkan Jam (Hari Kerja vs Libur)")
hour_summary = main_df.groupby(by=["workingday", "hour"]).agg({"total_count": "mean"}).reset_index()
hour_summary['workingday'] = hour_summary['workingday'].map({1: 'Hari Kerja', 0: 'Akhir Pekan/Libur'})

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(x="hour", y="total_count", hue="workingday", data=hour_summary, marker="o", linewidth=2, palette=["#72BCD4", "#FF9999"], ax=ax2)
ax2.set_title("Tren Penyewaan Sepeda per Jam", fontsize=15)
ax2.set_ylabel("Rata-rata Penyewaan")
ax2.set_xlabel("Jam")
ax2.set_xticks(range(0, 24))
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.markdown("---")

# 5. ADVANCED ANALYSIS (CLUSTERING & RFM)
st.subheader("💡 Advanced Analysis")

tab1, tab2 = st.tabs(["Time Clustering (Binning)", "RFM Analysis (Daily Proxy)"])

with tab1:
    st.markdown("**Pengelompokan Waktu Peminjaman (Manual Grouping / Binning)**")
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Pagi (05:00 - 11:59)'
        elif 12 <= hour < 17:
            return 'Siang (12:00 - 16:59)'
        elif 17 <= hour < 21:
            return 'Sore (17:00 - 20:59)'
        else:
            return 'Malam (21:00 - 04:59)'

    main_df['time_of_day'] = main_df['hour'].apply(get_time_of_day)
    time_cluster_summary = main_df.groupby(by=["workingday", "time_of_day"]).agg({"total_count": "mean"}).reset_index()
    time_cluster_summary['workingday'] = time_cluster_summary['workingday'].map({1: 'Hari Kerja', 0: 'Akhir Pekan/Libur'})
    time_cluster_summary['time_of_day'] = pd.Categorical(time_cluster_summary['time_of_day'], categories=['Pagi (05:00 - 11:59)', 'Siang (12:00 - 16:59)', 'Sore (17:00 - 20:59)', 'Malam (21:00 - 04:59)'], ordered=True)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(x="time_of_day", y="total_count", hue="workingday", data=time_cluster_summary, palette=["#72BCD4", "#FF9999"], ax=ax3)
    ax3.set_title("Penyewaan Berdasarkan Waktu (Clustering)", fontsize=15)
    ax3.set_ylabel("Rata-rata Penyewaan")
    ax3.set_xlabel(None)
    st.pyplot(fig3)

with tab2:
    st.markdown("**RFM Analysis (Menggunakan Tanggal sebagai proksi evaluasi performa)**")
    rfm_df = df.groupby('date').agg({'date': 'max', 'instant': 'count', 'total_count': 'sum'}).rename(columns={'date': 'max_date', 'instant': 'frequency', 'total_count': 'monetary'})
    rfm_df['max_date'] = rfm_df['max_date'].dt.date
    recent_date = df['date'].dt.date.max()
    rfm_df['recency'] = rfm_df['max_date'].apply(lambda x: (recent_date - x).days)
    rfm_df.drop('max_date', axis=1, inplace=True)
    rfm_df.reset_index(inplace=True)
    rfm_df['date_str'] = rfm_df['date'].dt.strftime('%Y-%m-%d')

    fig4, ax4 = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    colors = ["#72BCD4"] * 5
    
    sns.barplot(y="recency", x="date_str", hue="date_str", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax4[0], legend=False)
    ax4[0].set_title("Top 5 Tanggal (Recency)", fontsize=15)
    ax4[0].tick_params(axis='x', rotation=45)

    sns.barplot(y="frequency", x="date_str", hue="date_str", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax4[1], legend=False)
    ax4[1].set_title("Top 5 Tanggal (Frequency)", fontsize=15)
    ax4[1].tick_params(axis='x', rotation=45)

    sns.barplot(y="monetary", x="date_str", hue="date_str", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax4[2], legend=False)
    ax4[2].set_title("Top 5 Tanggal (Monetary - Total Sewa)", fontsize=15)
    ax4[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig4)

st.caption("Priadi Cuanda - Data Analyst | Dataset: Bike Sharing Hourly (2011-2012)")