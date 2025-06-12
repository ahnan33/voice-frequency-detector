
# Voice Frequency Detector (VFD) – Biometric Voice Authentication

Aplikasi ini adalah implementasi dari sistem **Voice Frequency Detector** menggunakan arsitektur **CNN-BiLSTM dengan mekanisme anti-spoofing** berbasis Random Forest. Dibangun dengan Streamlit, aplikasi ini memungkinkan autentikasi suara berbasis file `.wav`.

## 🔧 Fitur Utama
- Ekstraksi fitur suara: MFCC, Chroma, Spectral Contrast, Formants
- Deteksi suara palsu dengan fitur spektral dan model Random Forest
- Klasifikasi suara asli/spoof dengan CNN-BiLSTM + Attention
- Antarmuka interaktif berbasis Streamlit
- Logging ke dalam file Excel `auth_log.xlsx`

## 🗂️ Struktur Proyek
```
├── vfd_app_integrated.py          # Aplikasi utama Streamlit (semua fungsi terintegrasi)
├── requirements.txt               # Dependensi Python
├── dataset/                       # Folder berisi file .wav untuk training (real dan spoof)
├── rf_spoof.pkl                   # Model anti-spoof (opsional jika sudah dilatih)
├── vfd_model.h5                   # Model CNN-BiLSTM (opsional jika sudah dilatih)
└── auth_log.xlsx                  # Log autentikasi otomatis (dibuat saat aplikasi berjalan)
```

## 🚀 Cara Menjalankan Lokal

1. **Siapkan virtual environment (opsional tapi disarankan)**:
```bash
python3.12 -m venv vfd-env
source vfd-env/bin/activate  # Linux/macOS
.fd-env\Scriptsctivate   # Windows
```

2. **Instal dependensi**:
```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi**:
```bash
streamlit run vfd_app_integrated.py
```

## 🌐 Deploy ke Streamlit Cloud

1. Upload file berikut ke GitHub:
    - `vfd_app_integrated.py`
    - `requirements.txt`
    - `dataset/` (jika tidak terlalu besar dan tidak rahasia)

2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud), pilih repo, dan deploy!

## 📝 Catatan
- Jika belum memiliki model `.h5` dan `.pkl`, jalankan fungsi pelatihan dari dalam aplikasi untuk menghasilkan keduanya.
