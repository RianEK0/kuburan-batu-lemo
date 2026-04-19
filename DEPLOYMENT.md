# Publish (akses beda perangkat & beda jaringan)

Target: aplikasi bisa diakses teman di perangkat/jaringan lain (beda pulau) lewat URL publik.

## Opsi 1 — Streamlit Community Cloud (paling mudah)

1. Push repo ini ke GitHub.
2. Buka `https://share.streamlit.io` lalu pilih repo + branch + main file `app.py`.
3. (Disarankan) Tambah Secrets:
   - `APP_PASSWORD`: password untuk akses aplikasi.
   - `GOOGLE_MAPS_API_KEY`: kalau mau pakai mode **Google Places API (resmi)**.
4. Deploy → kamu dapat link publik untuk dibagikan.

Catatan:
- IndoBERT dimatikan default untuk demo publik (lebih cepat). Bisa dinyalakan di UI.

## Opsi 2 — Tunneling dari laptop (cepat, tapi bukan “hosting” permanen)

Kalau kamu ingin share sementara tanpa deploy:

- Jalankan lokal:
  - `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
- Buat URL publik via tunnel (pilih salah satu):
  - `ngrok http 8501`
  - `cloudflared tunnel --url http://127.0.0.1:8501`

## Dataset “di internet” (tanpa kirim file)

1. Buat CSV 1000+ baris:
   - `python3 generate_reviews_csv.py --n 1500 --out reviews_1500.csv`
2. Upload CSV ke tempat publik (contoh: GitHub raw).
3. Di aplikasi, pilih mode **URL CSV** lalu paste URL CSV publik tersebut.

## Penting: ulasan asli dari Google/TripAdvisor

Repo ini tidak menyediakan scraping langsung.
Untuk Google Maps, gunakan **Google Places API (resmi)** (biasanya hanya mengembalikan sebagian kecil review, bukan ribuan).

