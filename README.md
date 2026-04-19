# Analisis Sentimen Ulasan Kuburan Batu Lemo

Program ini menyediakan pipeline end-to-end untuk:

- preprocessing teks Bahasa Indonesia dan Inggris,
- analisis sentimen komparatif dengan `VADER`, `Naive Bayes`, dan `IndoBERT`,
- klasifikasi dimensi etika berbasis keyword,
- evaluasi model,
- visualisasi hasil,
- ekspor hasil ke CSV.

## Format Dataset

Minimal kolom:

```csv
review_text
"Tempatnya sangat indah dan budayanya terasa kuat."
```

Format yang direkomendasikan:

```csv
review_text,rating,date,sentiment_label
"Tempatnya sangat indah dan budayanya terasa kuat.",5,2025-01-14,positive
"The site is sacred, visitors should be respectful.",4,2025-01-15,positive
"Ticket price is expensive and the area was crowded.",2,2025-01-20,negative
```

Catatan:

- `review_text` wajib ada.
- `rating` opsional, tetapi sangat membantu untuk membangun label lemah jika `sentiment_label` tidak tersedia.
- `sentiment_label` opsional, namun paling direkomendasikan untuk evaluasi yang lebih valid.

## Instalasi

```bash
pip install -r requirements.txt
python3 -m nltk.downloader punkt punkt_tab stopwords vader_lexicon
```

## Menjalankan Aplikasi (Web)

```bash
streamlit run app.py
```

Lalu upload file CSV di UI.

## Generate Dataset 1000+ Baris (tanpa scraping)

Repo ini menyertakan generator data sintetis untuk demo / pengujian (bukan data ulasan asli dari internet):

```bash
python3 generate_reviews_csv.py --n 1500 --out reviews_1500.csv
```

Kolom yang dihasilkan sudah kompatibel dengan aplikasi: `review_text` (wajib), plus `rating`, `date`, `sentiment_label`.

## Berbagi ke Teman Beda Jaringan / Beda Pulau (Publish)

Agar bisa diakses dari perangkat & jaringan lain, host aplikasinya secara publik.

Opsi termudah:

- **Streamlit Community Cloud**: push repo ke GitHub → deploy `app.py` → dapat URL publik.
- Set secret/env `APP_PASSWORD` (disarankan) dan `GOOGLE_MAPS_API_KEY` (opsional).

Jika ingin membagikan dataset lewat aplikasi tanpa kirim file:

- Upload `reviews_1500.csv` ke tempat publik (contoh: GitHub raw) lalu pakai mode **URL CSV** di sidebar.

## Ambil Ulasan Otomatis (tanpa input manual)

Aplikasi mendukung sumber data:

- **Google Places API (resmi)**: ambil review via API key Google (butuh enable Places API + billing).
- **URL CSV**: baca dataset CSV publik dari URL (mis. hasil ekspor/scrape yang kamu sediakan sendiri).

Catatan penting:

- Kami **tidak menyediakan scraping langsung** Google Maps/TripAdvisor karena berpotensi melanggar Terms of Service.
- Google Places API biasanya hanya mengembalikan **sebagian kecil** review (bukan seluruh review di Google Maps).

Cara pakai Google Places API:

- Set env `GOOGLE_MAPS_API_KEY` (disarankan) atau isi di sidebar.
- Pilih mode `Google Places API (resmi)` lalu isi query tempat / place_id.

### Proteksi Password (disarankan untuk publik)

Set salah satu:

- `APP_PASSWORD`
- `STREAMLIT_APP_PASSWORD`

Untuk Google API key di deploy publik, simpan sebagai secret/env `GOOGLE_MAPS_API_KEY` (jangan hardcode di repo).

Atau di Streamlit Cloud, buat secret `APP_PASSWORD`.

Contoh lokal:

```bash
APP_PASSWORD="gantipassword" streamlit run app.py
```

### Opsi Deploy Cepat

- **Streamlit Community Cloud**: upload repo ini, set main file `app.py`, pastikan `requirements.txt` tersedia.
- **Docker**:

```bash
docker build -t kuburan-batu-lemo-sentiment .
docker run --rm -p 8501:8501 kuburan-batu-lemo-sentiment
```

### Catatan IndoBERT untuk Deploy Publik

Di `app.py`, IndoBERT berjalan **inference-only**. Untuk performa terbaik, gunakan model Hugging Face yang sudah fine-tuned untuk sentimen (bukan base model).

## Menjalankan Program

```bash
python3 sentiment_analysis_kuburan_batu.py \
  --input reviews.csv \
  --output-dir outputs \
  --bert-model indobenchmark/indobert-base-p1 \
  --show-plots
```

Untuk uji cepat, gunakan file contoh:

```bash
python3 sentiment_analysis_kuburan_batu.py --input sample_reviews.csv --output-dir outputs
```

## Output

Program akan menghasilkan:

- `reviews_with_sentiment_ethics.csv`
- `model_metrics.csv`
- `insights.txt`
- pie chart distribusi sentimen
- bar chart distribusi dimensi etika
- stacked bar sentimen per dimensi etika
- wordcloud review
- confusion matrix heatmap
- grafik perbandingan metode

## Catatan Akademik

- Jika dataset tidak memiliki `sentiment_label`, evaluasi ML akan menggunakan label dari `rating` atau pseudo-label berbasis lexicon.
- Untuk publikasi jurnal, lebih baik gunakan label sentimen manual atau rating yang tervalidasi.
- `IndoBERT` paling sesuai untuk data dominan Bahasa Indonesia; jika dataset bercampur banyak Bahasa Inggris, performanya perlu diinterpretasikan dengan hati-hati.
