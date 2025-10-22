import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Aplikasi Prediksi Penjualan",
    page_icon="📊",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_data
def load_model():
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Silakan jalankan script training terlebih dahulu.")
        return None, None

# Load model and scaler
model, scaler = load_model()

if model is not None and scaler is not None:
    # Title
    st.title("📊 Prediksi Pendapatan Penjualan")
    st.markdown("Prediksi pendapatan penjualan berdasarkan alokasi budget iklan")
    
    # Sidebar for input
    st.sidebar.header("💰 Masukkan Budget Iklan")
    
    # Input fields
    tv_budget = st.sidebar.number_input(
        "Budget TV ($)", 
        min_value=0, 
        max_value=1000, 
        value=100,
        help="Masukkan budget iklan TV"
    )
    
    radio_budget = st.sidebar.number_input(
        "Budget Radio ($)", 
        min_value=0, 
        max_value=1000, 
        value=50,
        help="Masukkan budget iklan Radio"
    )
    
    newspaper_budget = st.sidebar.number_input(
        "Budget Koran ($)", 
        min_value=0, 
        max_value=1000, 
        value=25,
        help="Masukkan budget iklan Koran"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Prediksi Pendapatan Penjualan", type="primary"):
        # Prepare input data
        input_data = {
            'TV': tv_budget,
            'Radio': radio_budget,
            'Newspaper': newspaper_budget
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        cols_to_scale = ['TV', 'Radio', 'Newspaper']
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display results
        st.success(f"🎯 **Hasil Prediksi Pendapatan Penjualan: ${prediction:.2f}**")
        
        # Show input summary
        st.subheader("📋 Ringkasan Input")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Budget TV", f"${tv_budget}")
        with col2:
            st.metric("Budget Radio", f"${radio_budget}")
        with col3:
            st.metric("Budget Koran", f"${newspaper_budget}")
        
        # Budget allocation visualization
        st.subheader("📊 Alokasi Budget")
        
        budget_data = {
            'Channel': ['TV', 'Radio', 'Koran'],
            'Budget': [tv_budget, radio_budget, newspaper_budget]
        }
        
        budget_df = pd.DataFrame(budget_data)
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        ax1.pie(budget_df['Budget'], labels=budget_df['Channel'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribusi Budget')
        
        # Bar chart
        ax2.bar(budget_df['Channel'], budget_df['Budget'], color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('Budget per Channel')
        ax2.set_ylabel('Budget ($)')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informasi Model")
    st.sidebar.info("""
    Aplikasi ini menggunakan model Random Forest Regressor untuk memprediksi pendapatan penjualan berdasarkan alokasi budget iklan di channel TV, Radio, dan Koran.
    """)
    
    # Main content area
    st.markdown("---")
    st.subheader("📈 Cara Penggunaan")
    st.markdown("""
    1. **Atur Budget**: Gunakan slider di sidebar untuk mengatur budget iklan untuk setiap channel
    2. **Prediksi**: Klik tombol "Prediksi Pendapatan Penjualan" untuk mendapatkan prediksi
    3. **Analisis**: Lihat chart alokasi budget dan prediksi pendapatan
    """)
    
    # Sample predictions
    st.subheader("🎯 Contoh Prediksi")
    
    # Define cols_to_scale for sample predictions
    cols_to_scale = ['TV', 'Radio', 'Newspaper']
    
    sample_data = [
        {"TV": 200, "Radio": 50, "Newspaper": 30, "Expected": "Fokus TV Tinggi"},
        {"TV": 100, "Radio": 100, "Newspaper": 50, "Expected": "Pendekatan Seimbang"},
        {"TV": 50, "Radio": 150, "Newspaper": 100, "Expected": "Fokus Radio/Koran"}
    ]
    
    for i, sample in enumerate(sample_data):
        with st.expander(f"Contoh {i+1}: {sample['Expected']}"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**TV:** ${sample['TV']}")
            with col2:
                st.write(f"**Radio:** ${sample['Radio']}")
            with col3:
                st.write(f"**Koran:** ${sample['Newspaper']}")
            with col4:
                # Quick prediction for sample
                sample_input = pd.DataFrame([{
                    'TV': sample['TV'],
                    'Radio': sample['Radio'],
                    'Newspaper': sample['Newspaper']
                }])
                sample_input[cols_to_scale] = scaler.transform(sample_input[cols_to_scale])
                sample_pred = model.predict(sample_input)[0]
                st.write(f"**Prediksi:** ${sample_pred:.2f}")

else:
    st.error("❌ File model tidak ditemukan. Pastikan file 'rf_model.joblib' dan 'scaler.joblib' ada di direktori saat ini.")
    st.info("💡 Jalankan script training terlebih dahulu untuk menghasilkan file model.")
