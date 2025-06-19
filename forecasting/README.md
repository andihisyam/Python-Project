# TUKUPOS Sales Analysis

Proyek ini berfokus pada analisis data penjualan dari perusahaan yang terdaftar di **TUKUPOS**, sebuah aplikasi manajemen UMKM yang dikembangkan oleh dosen **Telkom University Purwokerto**, **Novian Adi Prasetyo, M.Kom** dan **Arief Rais Bahtiar, M.Kom**. Aplikasi ini bertujuan membantu pelaku UMKM di Purbalingga dalam mengelola keuangan dan penjualan secara digital dengan antarmuka yang **user-friendly** dan **terjangkau**.

## 📊 Deskripsi Proyek

Analisis ini bertujuan untuk:
- Mengidentifikasi tren penjualan dari data perusahaan UMKM.
- Menentukan produk terlaris dan waktu penjualan terbaik.
- Memberikan insight bagi UMKM untuk mengoptimalkan strategi bisnis mereka.

## 📈 Analisis Deret Waktu
Analisis deret waktu dilakukan untuk memahami tren dan pola musiman dalam penjualan, dengan hasil utama sebagai berikut:

1. Memahami Pola Musiman Penjualan:
- Identifikasi pola musiman dari data historis penjualan.
- Analisis faktor-faktor yang mempengaruhi tren musiman, seperti event atau musim tertentu.

2. Fokus pada Penjualan Saat Musim Tinggi:
- Optimalisasi strategi pemasaran dan operasional saat permintaan tinggi.
- Memastikan ketersediaan stok dan kapasitas produksi yang memadai.
- Mengalokasikan sumber daya secara efektif untuk memenuhi lonjakan permintaan.

3. Menciptakan Permintaan di Luar Musim Tinggi:
- Menawarkan diskon dan promosi untuk mendorong penjualan di luar musim.
- Memanfaatkan media sosial dan digital marketing untuk menjangkau pelanggan potensial.
- Mengembangkan program loyalitas pelanggan untuk meningkatkan retensi.

4. Mengoptimalkan Operasional dan Efisiensi Biaya:
- Mengatur inventori dan rantai pasokan untuk meminimalkan biaya.
- Mengautomasi proses operasional guna meningkatkan efisiensi.

## 🔗 Analisis Apriori (Market Basket Analysis)
Analisis Apriori digunakan untuk menemukan pola asosiasi antar produk berdasarkan perilaku konsumen, dengan insight utama sebagai berikut:

1. Penataan Rak yang Efisien:
- Produk dengan hubungan kuat berdasarkan antecedent dan consequent dari analisis Apriori ditempatkan berdekatan untuk meningkatkan penjualan.
- Penataan ini membantu konsumen menemukan produk lebih mudah dan mendorong pembelian impulsif.

2. Strategi Bundling dan Promosi Berdasarkan Pola Pembelian:
- Produk yang sering dibeli bersama dapat dijadikan bundling atau paket promo.
- Strategi ini dapat meningkatkan transaksi dan memaksimalkan keuntungan melalui cross-selling.
  
## 🔒 Tentang Data
Data yang digunakan dalam analisis ini memiliki format sebagai berikut:

| ID  | Tanggal     | Trans ID  | Konsumen     | Nama Produk  | Kategori        | Qty  |
|-----|-------------|-----------|--------------|--------------|-----------------|------|
| 1   | 2023-01-01  | TRX001    | Konsumen A   | Produk A     | Minuman         | 2    |
| 2   | 2023-01-01  | TRX002    | Konsumen B   | Produk B     | Makanan Ringan  | 1    |
| 3   | 2023-01-02  | TRX003    | Konsumen C   | Produk A     | Minuman         | 3    |
| 4   | 2023-01-02  | TRX004    | Konsumen D   | Produk C     | Alat Rumah Tangga | 1  |
| 5   | 2023-01-03  | TRX005    | Konsumen E   | Produk B     | Makanan Ringan  | 4    |

### Penjelasan Kolom:
- **ID:** Nomor urut data.
- **Tanggal:** Tanggal transaksi dilakukan.
- **Trans ID:** ID unik untuk setiap transaksi.
- **Konsumen:** Nama atau identitas konsumen.
- **Nama Produk:** Produk yang dibeli dalam transaksi.
- **Kategori:** Kategori produk (misalnya Minuman, Makanan Ringan, Alat Rumah Tangga, dll).
- **Qty:** Jumlah produk yang dibeli.

Data asli **tidak disertakan** dalam repository ini karena alasan privasi. Anda dapat menggunakan data dengan format serupa untuk menjalankan analisis.

