import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Aplikasi Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_data
def load_model():
    # Try multiple possible paths for model files
    model_paths = [
        'diabetes_rf_model.joblib',  # Current directory
        '../diabetes_rf_model.joblib',  # Parent directory
        'src/diabetes_rf_model.joblib',  # src subdirectory
        '../src/diabetes_rf_model.joblib'  # src in parent directory
    ]
    
    scaler_paths = [
        'diabetes_scaler.joblib',  # Current directory
        '../diabetes_scaler.joblib',  # Parent directory
        'src/diabetes_scaler.joblib',  # src subdirectory
        '../src/diabetes_scaler.joblib'  # src in parent directory
    ]
    
    model_path = None
    scaler_path = None
    
    # Find model file
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    # Find scaler file
    for path in scaler_paths:
        if os.path.exists(path):
            scaler_path = path
            break
    
    if model_path and scaler_path:
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    else:
        st.error("File model tidak ditemukan. Silakan jalankan script training terlebih dahulu.")
        st.error(f"Model path: {model_path}, Scaler path: {scaler_path}")
        return None, None

# Load model and scaler
model, scaler = load_model()

if model is not None and scaler is not None:
    # Title
    st.title("🩺 Aplikasi Prediksi Diabetes")
    st.markdown("Prediksi risiko diabetes berdasarkan parameter kesehatan pasien")
    
    # Sidebar for input
    st.sidebar.header("📊 Data Pasien")
    
    # Input fields
    high_bp = st.sidebar.selectbox(
        "Tekanan Darah Tinggi", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien memiliki tekanan darah tinggi?"
    )
    
    high_chol = st.sidebar.selectbox(
        "Kolesterol Tinggi", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien memiliki kolesterol tinggi?"
    )
    
    chol_check = st.sidebar.selectbox(
        "Pengecekan Kolesterol", 
        [0, 1],
        index=1,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien pernah melakukan pengecekan kolesterol?"
    )
    
    bmi = st.sidebar.number_input(
        "BMI (kg/m²)", 
        min_value=10.0, 
        max_value=60.0, 
        value=25.5,
        step=0.1,
        help="Body Mass Index"
    )
    
    smoker = st.sidebar.selectbox(
        "Perokok", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien merokok?"
    )
    
    stroke = st.sidebar.selectbox(
        "Riwayat Stroke", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien pernah mengalami stroke?"
    )
    
    heart_disease = st.sidebar.selectbox(
        "Penyakit Jantung", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien memiliki penyakit jantung?"
    )
    
    phys_activity = st.sidebar.selectbox(
        "Aktivitas Fisik", 
        [0, 1],
        index=1,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien melakukan aktivitas fisik?"
    )
    
    fruits = st.sidebar.selectbox(
        "Konsumsi Buah", 
        [0, 1],
        index=1,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien mengonsumsi buah?"
    )
    
    veggies = st.sidebar.selectbox(
        "Konsumsi Sayuran", 
        [0, 1],
        index=1,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien mengonsumsi sayuran?"
    )
    
    heavy_alcohol = st.sidebar.selectbox(
        "Konsumsi Alkohol Berat", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien mengonsumsi alkohol berat?"
    )
    
    any_healthcare = st.sidebar.selectbox(
        "Akses Kesehatan", 
        [0, 1],
        index=1,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien memiliki akses ke layanan kesehatan?"
    )
    
    no_doc_cost = st.sidebar.selectbox(
        "Tidak ke Dokter karena Biaya", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien tidak ke dokter karena masalah biaya?"
    )
    
    gen_health = st.sidebar.slider(
        "Kesehatan Umum (1-5)", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Rating kesehatan umum (1=sangat buruk, 5=sangat baik)"
    )
    
    mental_health = st.sidebar.slider(
        "Kesehatan Mental (0-30)", 
        min_value=0, 
        max_value=30, 
        value=5,
        help="Jumlah hari dengan kesehatan mental buruk dalam 30 hari terakhir"
    )
    
    physical_health = st.sidebar.slider(
        "Kesehatan Fisik (0-30)", 
        min_value=0, 
        max_value=30, 
        value=5,
        help="Jumlah hari dengan kesehatan fisik buruk dalam 30 hari terakhir"
    )
    
    diff_walk = st.sidebar.selectbox(
        "Kesulitan Berjalan", 
        [0, 1],
        index=0,
        format_func=lambda x: "Tidak" if x == 0 else "Ya",
        help="Apakah pasien mengalami kesulitan berjalan?"
    )
    
    sex = st.sidebar.selectbox(
        "Jenis Kelamin", 
        [0, 1],
        index=0,
        format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki",
        help="Jenis kelamin pasien"
    )
    
    age = st.sidebar.slider(
        "Usia (tahun)", 
        min_value=1, 
        max_value=13, 
        value=9,
        help="Kategori usia (1=18-24, 2=25-29, ..., 13=80+)"
    )
    
    education = st.sidebar.slider(
        "Pendidikan (1-6)", 
        min_value=1, 
        max_value=6, 
        value=4,
        help="Tingkat pendidikan (1=tidak sekolah, 6=graduate school)"
    )
    
    income = st.sidebar.slider(
        "Pendapatan (1-8)", 
        min_value=1, 
        max_value=8, 
        value=3,
        help="Kategori pendapatan (1=<$10k, 8=>$75k)"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Prediksi Risiko Diabetes", type="primary"):
        # Prepare input data
        input_data = {
            'HighBP': high_bp,
            'HighChol': high_chol,
            'CholCheck': chol_check,
            'BMI': bmi,
            'Smoker': smoker,
            'Stroke': stroke,
            'HeartDiseaseorAttack': heart_disease,
            'PhysActivity': phys_activity,
            'Fruits': fruits,
            'Veggies': veggies,
            'HvyAlcoholConsump': heavy_alcohol,
            'AnyHealthcare': any_healthcare,
            'NoDocbcCost': no_doc_cost,
            'GenHlth': gen_health,
            'MentHlth': mental_health,
            'PhysHlth': physical_health,
            'DiffWalk': diff_walk,
            'Sex': sex,
            'Age': age,
            'Education': education,
            'Income': income
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.keys())
        
        # Make prediction
        prediction = model.predict(input_scaled_df)[0]
        prediction_proba = model.predict_proba(input_scaled_df)[0]
        
        # Display results based on multiclass prediction
        if prediction == 0:
            st.success(f"✅ **TIDAK DIABETES**")
            st.success(f"Probabilitas Tidak Diabetes: {prediction_proba[0]:.2%}")
        elif prediction == 1:
            st.error(f"🚨 **DIABETES TIPE 1**")
            st.error(f"Probabilitas Diabetes Tipe 1: {prediction_proba[1]:.2%}")
        else:  # prediction == 2
            st.warning(f"⚠️ **DIABETES TIPE 2**")
            st.warning(f"Probabilitas Diabetes Tipe 2: {prediction_proba[2]:.2%}")
        
        # Show detailed results
        st.subheader("📋 Hasil Prediksi Detail")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tidak Diabetes", f"{prediction_proba[0]:.2%}")
        with col2:
            st.metric("Diabetes Tipe 1", f"{prediction_proba[1]:.2%}")
        with col3:
            st.metric("Diabetes Tipe 2", f"{prediction_proba[2]:.2%}")
        
        # Risk interpretation
        st.subheader("📊 Interpretasi Risiko")
        max_prob = max(prediction_proba)
        if max_prob == prediction_proba[0]:
            st.success("✅ **RISIKO RENDAH** - Tetap jaga pola hidup sehat")
        elif max_prob == prediction_proba[1]:
            st.error("🚨 **RISIKO TINGGI - DIABETES TIPE 1** - Segera konsultasi dengan dokter")
        else:  # max_prob == prediction_proba[2]
            st.warning("⚠️ **RISIKO SEDANG - DIABETES TIPE 2** - Disarankan untuk melakukan pemeriksaan lebih lanjut")
        
        # Risk visualization
        st.subheader("📈 Visualisasi Risiko Diabetes")
        
        # Create risk visualization
        risk_data = {
            'Kategori': ['Tidak Diabetes', 'Diabetes Tipe 1', 'Diabetes Tipe 2'],
            'Probabilitas': [prediction_proba[0], prediction_proba[1], prediction_proba[2]],
            'Warna': ['#2E8B57', '#DC143C', '#FF8C00']
        }
        
        risk_df = pd.DataFrame(risk_data)
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Pie chart for probabilities
        ax1.pie(risk_df['Probabilitas'], labels=risk_df['Kategori'], autopct='%1.1f%%', 
                colors=risk_df['Warna'], startangle=90)
        ax1.set_title('Distribusi Probabilitas Diabetes')
        
        # 2. Bar chart for probabilities
        bars = ax2.bar(risk_df['Kategori'], risk_df['Probabilitas'], color=risk_df['Warna'])
        ax2.set_title('Probabilitas per Kategori')
        ax2.set_ylabel('Probabilitas')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, risk_df['Probabilitas']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.2%}', ha='center', va='bottom')
        
        # 3. Risk level gauge
        risk_level = "RENDAH" if max_prob == prediction_proba[0] else "TINGGI" if max_prob == prediction_proba[1] else "SEDANG"
        
        # Create a simple gauge
        ax3.text(0.5, 0.7, f'RISIKO: {risk_level}', ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.5, 0.5, f'Probabilitas Tertinggi: {max_prob:.2%}', ha='center', va='center', 
                fontsize=12, transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Tingkat Risiko')
        
        # 4. Feature importance (if available)
        try:
            # Get feature importance from the model
            feature_names = ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Age', 'DiffWalk', 
                           'PhysHlth', 'HeartDisease', 'Income', 'Education']
            importance_values = [0.195, 0.099, 0.168, 0.206, 0.071, 0.068, 
                               0.038, 0.036, 0.033, 0.016]  # From model training output
            
            # Take top 8 features
            top_features = feature_names[:8]
            top_importance = importance_values[:8]
            
            ax4.barh(top_features, top_importance, color='skyblue')
            ax4.set_title('Faktor Risiko Terpenting')
            ax4.set_xlabel('Importance')
        except:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title('Faktor Risiko')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informasi Model")
    st.sidebar.info("""
    Aplikasi ini menggunakan model Random Forest Classifier untuk memprediksi risiko diabetes berdasarkan parameter kesehatan pasien.
    
    **Parameter yang digunakan:**
    - Jumlah kehamilan
    - Konsentrasi glukosa
    - Tekanan darah
    - Ketebalan kulit
    - Insulin serum
    - BMI
    - Riwayat keluarga diabetes
    - Usia
    """)
    
    # Main content area
    st.markdown("---")
    st.subheader("📈 Cara Penggunaan")
    st.markdown("""
    1. **Masukkan Data Pasien**: Gunakan form di sidebar untuk memasukkan parameter kesehatan pasien
    2. **Prediksi**: Klik tombol "Prediksi Risiko Diabetes" untuk mendapatkan prediksi
    3. **Interpretasi**: Lihat hasil prediksi dan interpretasi tingkat risiko
    """)
    
    # Dataset analysis visualization
    st.subheader("📊 Analisis Dataset Diabetes")
    
    # Load and analyze dataset
    @st.cache_data
    def load_dataset_analysis():
        # Try multiple possible paths for dataset
        dataset_paths = [
            'datasets.csv',  # Current directory
            '../datasets.csv',  # Parent directory
            'src/datasets.csv',  # src subdirectory
            '../src/datasets.csv'  # src in parent directory
        ]
        
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    return df
                except:
                    continue
        return None
    
    df_analysis = load_dataset_analysis()
    
    if df_analysis is not None:
        # Create analysis visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Target distribution
        target_counts = df_analysis['Diabetes_012'].value_counts()
        labels = ['Tidak Diabetes', 'Diabetes Tipe 1', 'Diabetes Tipe 2']
        colors = ['#2E8B57', '#DC143C', '#FF8C00']
        
        ax1.pie(target_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Distribusi Target Variable')
        
        # 2. Age distribution by diabetes status
        age_diabetes = df_analysis.groupby(['Age', 'Diabetes_012']).size().unstack(fill_value=0)
        age_diabetes.plot(kind='bar', stacked=True, ax=ax2, color=colors)
        ax2.set_title('Distribusi Usia berdasarkan Status Diabetes')
        ax2.set_xlabel('Kategori Usia')
        ax2.set_ylabel('Jumlah')
        ax2.legend(['Tidak Diabetes', 'Diabetes Tipe 1', 'Diabetes Tipe 2'])
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. BMI distribution
        ax3.hist([df_analysis[df_analysis['Diabetes_012'] == 0]['BMI'], 
                 df_analysis[df_analysis['Diabetes_012'] == 1]['BMI'],
                 df_analysis[df_analysis['Diabetes_012'] == 2]['BMI']], 
                bins=20, alpha=0.7, label=labels, color=colors)
        ax3.set_title('Distribusi BMI berdasarkan Status Diabetes')
        ax3.set_xlabel('BMI')
        ax3.set_ylabel('Frekuensi')
        ax3.legend()
        
        # 4. Health factors correlation
        health_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']
        factor_counts = df_analysis[health_factors].sum()
        
        ax4.barh(health_factors, factor_counts, color='lightcoral')
        ax4.set_title('Faktor Risiko Kesehatan (Total Kasus)')
        ax4.set_xlabel('Jumlah Kasus')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Dataset statistics
        st.subheader("📋 Statistik Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sampel", f"{len(df_analysis):,}")
        with col2:
            st.metric("Tidak Diabetes", f"{target_counts[0]:,} ({target_counts[0]/len(df_analysis)*100:.1f}%)")
        with col3:
            st.metric("Diabetes Tipe 2", f"{target_counts[2]:,} ({target_counts[2]/len(df_analysis)*100:.1f}%)")
    else:
        st.warning("Dataset tidak tersedia untuk analisis")
    
    # Sample predictions
    st.subheader("🎯 Contoh Prediksi")
    
    sample_data = [
        {
            "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 25.5, "Smoker": 0,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 3, "MentHlth": 5, "PhysHlth": 5, "DiffWalk": 0, "Sex": 0,
            "Age": 9, "Education": 4, "Income": 3,
            "Description": "Pasien Sehat"
        },
        {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.0, "Smoker": 1,
            "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0,
            "Veggies": 0, "HvyAlcoholConsump": 1, "AnyHealthcare": 0, "NoDocbcCost": 1,
            "GenHlth": 1, "MentHlth": 15, "PhysHlth": 20, "DiffWalk": 1, "Sex": 1,
            "Age": 12, "Education": 2, "Income": 1,
            "Description": "Pasien Berisiko Tinggi"
        },
        {
            "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22.0, "Smoker": 0,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 5, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
            "Age": 6, "Education": 6, "Income": 6,
            "Description": "Pasien Muda Sehat"
        }
    ]
    
    for i, sample in enumerate(sample_data):
        with st.expander(f"Contoh {i+1}: {sample['Description']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Parameter Pasien:**")
                for key, value in sample.items():
                    if key != "Description":
                        st.write(f"- {key}: {value}")
            
            with col2:
                # Quick prediction for sample
                sample_input = pd.DataFrame([{k: v for k, v in sample.items() if k != "Description"}])
                sample_scaled = scaler.transform(sample_input)
                sample_scaled_df = pd.DataFrame(sample_scaled, columns=[k for k in sample.keys() if k != "Description"])
                sample_pred = model.predict(sample_scaled_df)[0]
                sample_proba = model.predict_proba(sample_scaled_df)[0]
                
                st.write("**Hasil Prediksi:**")
                if sample_pred == 0:
                    st.success(f"✅ Tidak Diabetes: {sample_proba[0]:.2%}")
                elif sample_pred == 1:
                    st.error(f"🚨 Diabetes Tipe 1: {sample_proba[1]:.2%}")
                else:  # sample_pred == 2
                    st.warning(f"⚠️ Diabetes Tipe 2: {sample_proba[2]:.2%}")
    
    # Health tips
    st.subheader("💡 Tips Kesehatan")
    st.markdown("""
    **Untuk Mencegah Diabetes:**
    - 🏃‍♂️ Olahraga teratur minimal 30 menit sehari
    - 🥗 Konsumsi makanan sehat dan seimbang
    - 🚫 Hindari makanan tinggi gula dan lemak
    - 📏 Jaga berat badan ideal
    - 🚭 Hindari merokok dan alkohol
    - 🩺 Rutin cek kesehatan dan gula darah
    """)

else:
    st.error("❌ File model tidak ditemukan. Pastikan file 'diabetes_rf_model.joblib' dan 'diabetes_scaler.joblib' ada di direktori saat ini.")
    st.info("💡 Jalankan script 'diabetes_classification_model.py' terlebih dahulu untuk menghasilkan file model.")
    
    st.subheader("📋 Langkah-langkah untuk menjalankan aplikasi:")
    st.markdown("""
    1. **Jalankan Training Script**: 
       ```bash
       python diabetes_classification_model.py
       ```
    
    2. **Pastikan File Model Ada**:
       - `diabetes_rf_model.joblib`
       - `diabetes_scaler.joblib`
    
    3. **Jalankan Aplikasi Streamlit**:
       ```bash
       streamlit run diabetes_streamlit_app.py
       ```
    """)
