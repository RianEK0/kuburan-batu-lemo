from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import torch

from data_sources import (
    DataSourceError,
    get_google_api_key,
    google_fetch_reviews_legacy,
    google_find_place_id_legacy,
    load_reviews_from_public_csv_url,
)
from sentiment_analysis_kuburan_batu import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    aggregate_lexicon_predictions,
    build_stopword_sets,
    classify_ethics,
    choose_best_method,
    compute_metrics,
    ensure_nltk_resources,
    generate_auto_insights,
    maybe_bootstrap_labels_from_lexicon,
    plot_confusion_matrix,
    prepare_labels,
    preprocess_text,
    set_seed,
    train_ml_model,
    validate_and_adjust_folds,
    visualize_results,
)

SAMPLE_500_PATH = Path(__file__).with_name("sample_reviews_500.csv")


def _read_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df


def _load_input_dataframe(source_mode: str, uploaded_file) -> pd.DataFrame | None:
    if source_mode == "Contoh (500 baris)":
        if not SAMPLE_500_PATH.exists():
            raise DataSourceError("File contoh `sample_reviews_500.csv` tidak ditemukan.")
        return pd.read_csv(SAMPLE_500_PATH)

    if source_mode == "Upload CSV":
        if not uploaded_file:
            return None
        return _read_csv(uploaded_file)

    if source_mode == "URL CSV":
        url = st.session_state.get("_csv_url", "")
        if not str(url).strip():
            return None
        return load_reviews_from_public_csv_url(str(url).strip())

    if source_mode == "Google Places API (resmi)":
        query = str(st.session_state.get("_google_query", "")).strip()
        place_id = str(st.session_state.get("_google_place_id", "")).strip()
        language = str(st.session_state.get("_google_lang", "id")).strip() or "id"
        api_key_input = str(st.session_state.get("_google_api_key", "")).strip()
        if not place_id and not query:
            return None
        api_key = get_google_api_key(api_key_input or None)
        if not place_id:
            place = google_find_place_id_legacy(api_key, query=query)
            place_id = place.place_id
            st.session_state["_google_place_id"] = place_id
        df = google_fetch_reviews_legacy(api_key, place_id=place_id, language=language)
        return df

    return None


def _get_app_password() -> str | None:
    try:
        secrets_pw = st.secrets.get("APP_PASSWORD")  # type: ignore[attr-defined]
        if secrets_pw:
            return str(secrets_pw)
    except Exception:  # noqa: BLE001
        pass
    env_pw = os.getenv("APP_PASSWORD") or os.getenv("STREAMLIT_APP_PASSWORD")
    return str(env_pw) if env_pw else None


def _require_password() -> bool:
    app_password = _get_app_password()
    if not app_password:
        return True

    st.sidebar.header("Akses")
    pw = st.sidebar.text_input("Password", type="password", placeholder="Masukkan password")
    if not pw:
        st.warning("Masukkan password untuk mengakses aplikasi.")
        return False
    if pw != app_password:
        st.error("Password salah.")
        return False
    return True


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1200px; }
          section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
          /* Hide Streamlit default footer/menu */
          footer { visibility: hidden; }
          #MainMenu { visibility: hidden; }
          header { visibility: hidden; }
          /* Headings */
          h2, h3 { letter-spacing: -0.02em; }
          /* Tabs */
          button[data-baseweb="tab"] { font-weight: 600; }
          /* Metric cards */
          div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(17,24,39,0.08);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          /* Dataframe container */
          div[data-testid="stDataFrame"] {
            border: 1px solid rgba(17,24,39,0.08);
            border-radius: 14px;
            overflow: hidden;
          }
          /* Buttons */
          .stButton > button, .stDownloadButton > button {
            border-radius: 12px !important;
            padding: 0.6rem 0.9rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_nlp_resources():
    set_seed(42)
    ensure_nltk_resources()
    stemmer = StemmerFactory().create_stemmer()
    id_stopwords, en_stopwords = build_stopword_sets()
    vader_analyzer = SentimentIntensityAnalyzer()
    return stemmer, id_stopwords, en_stopwords, vader_analyzer


@st.cache_resource(show_spinner=False)
def _load_transformers_model(model_name: str):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _map_to_supported_sentiment(label: str) -> str:
    label_lower = str(label).lower()
    mapping = {
        "neg": "negative",
        "negative": "negative",
        "negatif": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "netral": "neutral",
        "pos": "positive",
        "positive": "positive",
        "positif": "positive",
    }
    for key, value in mapping.items():
        if key in label_lower:
            return value
    return "neutral"


def _predict_indobert_inference(texts: list[str], model_name: str, max_length: int = 256) -> list[str]:
    tokenizer, model = _load_transformers_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 16
    predictions: list[str] = []
    id2label = getattr(model.config, "id2label", None) or {}

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            for pred_id in pred_ids:
                raw_label = id2label.get(int(pred_id), str(pred_id))
                predictions.append(_map_to_supported_sentiment(raw_label))
    return predictions


def _run_analysis(
    df_raw: pd.DataFrame,
    folds: int,
    max_length: int,
    bert_model_name: str,
    run_indobert: bool,
) -> dict:
    stemmer, id_stopwords, en_stopwords, vader_analyzer = _get_nlp_resources()

    df = df_raw.copy()
    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    df["ethical_category"] = df[TEXT_COLUMN].apply(classify_ethics)
    df["processed_text"] = df[TEXT_COLUMN].apply(
        lambda text: preprocess_text(
            text,
            stemmer=stemmer,
            id_stopwords=id_stopwords,
            en_stopwords=en_stopwords,
        )
    )

    labels = prepare_labels(df)
    labels = maybe_bootstrap_labels_from_lexicon(
        df,
        labels,
        vader_analyzer=vader_analyzer,
        stemmer=stemmer,
        id_stopwords=id_stopwords,
        en_stopwords=en_stopwords,
    )
    adjusted_folds = validate_and_adjust_folds(labels, int(folds))

    df["vader_sentiment"] = aggregate_lexicon_predictions(
        df,
        vader_analyzer=vader_analyzer,
        stemmer=stemmer,
        id_stopwords=id_stopwords,
        en_stopwords=en_stopwords,
    )
    vader_metrics = compute_metrics(labels, df["vader_sentiment"])

    nb_model, nb_metrics, nb_cv_predictions, nb_metadata = train_ml_model(df, labels, n_splits=adjusted_folds)
    df["naive_bayes_sentiment"] = nb_model.predict(df[["processed_text"]])

    bert_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if run_indobert:
        df["indobert_sentiment"] = _predict_indobert_inference(
            texts=df[TEXT_COLUMN].tolist(),
            model_name=bert_model_name,
            max_length=int(max_length),
        )
        bert_metrics = compute_metrics(labels, df["indobert_sentiment"])
    else:
        df["indobert_sentiment"] = "neutral"

    metrics_table = pd.DataFrame(
        [
            {"method": "VADER", **vader_metrics},
            {"method": "Naive Bayes", **nb_metrics},
            {"method": "IndoBERT", **bert_metrics},
        ]
    )

    best_method = choose_best_method(metrics_table)
    method_column_map = {
        "VADER": "vader_sentiment",
        "Naive Bayes": "naive_bayes_sentiment",
        "IndoBERT": "indobert_sentiment",
    }
    best_method_column = method_column_map[best_method]
    df["sentiment"] = df[best_method_column]

    # Create plots into a temporary directory, then keep them as bytes
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        visualize_results(
            df=df,
            metrics_table=metrics_table,
            best_method_name=best_method,
            best_method_column=best_method_column,
            output_dir=output_dir,
            show_plots=False,
        )
        plot_confusion_matrix(labels, df["vader_sentiment"], "Confusion Matrix - VADER", output_dir / "cm_vader.png")
        plot_confusion_matrix(
            labels,
            nb_cv_predictions,
            "Confusion Matrix - Naive Bayes (CV)",
            output_dir / "cm_naive_bayes.png",
        )
        if run_indobert:
            plot_confusion_matrix(
                labels,
                df["indobert_sentiment"],
                "Confusion Matrix - IndoBERT",
                output_dir / "cm_indobert.png",
            )

        viz_paths = [
            output_dir / "pie_sentiment_distribution.png",
            output_dir / "bar_ethics_distribution.png",
            output_dir / "stacked_sentiment_by_ethics.png",
            output_dir / "wordcloud_reviews.png",
            output_dir / "model_comparison.png",
            output_dir / "cm_vader.png",
            output_dir / "cm_naive_bayes.png",
        ]
        if run_indobert:
            viz_paths.append(output_dir / "cm_indobert.png")

        images = {p.name: p.read_bytes() for p in viz_paths if p.exists()}

    insights = generate_auto_insights(df, metrics_table, best_method, best_method_column)

    export_cols = [TEXT_COLUMN, "sentiment", "ethical_category"]
    for extra in ["rating", "date", LABEL_COLUMN, "vader_sentiment", "naive_bayes_sentiment", "indobert_sentiment"]:
        if extra in df.columns and extra not in export_cols:
            export_cols.append(extra)
    out_df = df[export_cols].copy()

    return {
        "df": out_df,
        "metrics_table": metrics_table,
        "best_method": best_method,
        "best_method_column": best_method_column,
        "insights": insights,
        "images": images,
        "naive_bayes_best_params": nb_metadata.get("best_params"),
    }


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis Kuburan Batu Lemo", layout="wide")
    _inject_css()
    st.markdown("## Analisis Sentimen Ulasan Wisata")
    st.markdown(
        "<div style='font-size:1.05rem; opacity:0.85; margin-top:-6px'>"
        "Kuburan Batu Lemo (Toraja) • VADER • Naive Bayes • IndoBERT • Dimensi Etika"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not _require_password():
        return

    with st.sidebar:
        st.header("Input Data")
        source_mode = st.radio(
            "Sumber data",
            ["Contoh (500 baris)", "Upload CSV", "Google Places API (resmi)", "URL CSV"],
            help=(
                "Untuk Google Maps gunakan Google Places API (resmi). "
                "Kami tidak menyediakan scraping karena berpotensi melanggar TOS."
            ),
        )

        uploaded = None
        if source_mode == "Contoh (500 baris)":
            st.caption("Cocok untuk demo cepat tanpa upload file.")
            if SAMPLE_500_PATH.exists():
                st.download_button(
                    "Download CSV contoh (500 baris)",
                    data=SAMPLE_500_PATH.read_bytes(),
                    file_name="sample_reviews_500.csv",
                    mime="text/csv",
                    width="stretch",
                )
        elif source_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
        elif source_mode == "URL CSV":
            st.text_input("URL CSV publik", key="_csv_url", placeholder="https://.../reviews.csv")
        else:
            st.text_input(
                "Google API key (atau set env `GOOGLE_MAPS_API_KEY`)",
                key="_google_api_key",
                type="password",
                placeholder="AIza...",
            )
            st.text_input(
                "Query tempat (contoh: Kuburan Batu Lemo Toraja)",
                key="_google_query",
                value="Kuburan Batu Lemo Toraja",
            )
            st.text_input("Place ID (opsional)", key="_google_place_id", placeholder="ChIJ...")
            st.selectbox("Language", options=["id", "en"], key="_google_lang", index=0)

            st.caption("Catatan: Places API biasanya hanya mengembalikan sejumlah kecil review (bukan semua).")

        st.markdown("Kolom wajib: `review_text`.")
        st.markdown("Opsional: `rating`, `date`, `sentiment_label`.")

        st.header("Pengaturan")
        folds = st.number_input("K-fold CV", min_value=2, max_value=10, value=5, step=1)
        max_length = st.number_input("Max length IndoBERT", min_value=64, max_value=512, value=256, step=32)
        bert_model_name = st.text_input(
            "HuggingFace sentiment model (IndoBERT)",
            value="indobenchmark/indobert-base-p1",
            help="Untuk mode publik, disarankan pakai model sentiment yang sudah fine-tuned (inference-only).",
        )
        run_indobert = st.checkbox(
            "Jalankan IndoBERT (inference-only)",
            value=False,
            help="Untuk demo publik, nonaktifkan dulu agar aplikasi lebih cepat dan tidak perlu download model.",
        )
        st.divider()
        if st.button("Reset hasil", width="stretch"):
            st.session_state.pop("result", None)

    try:
        df_raw = _load_input_dataframe(source_mode, uploaded)
        if df_raw is None:
            st.info("Isi sumber data di sidebar untuk mulai.")
            return
    except DataSourceError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat data: {exc}")
        return

    if TEXT_COLUMN not in df_raw.columns:
        st.error("CSV harus memiliki kolom `review_text`.")
        return

    max_rows = 20000
    if len(df_raw) > max_rows:
        st.warning(
            f"Dataset berisi {len(df_raw):,} baris. Demi stabilitas saat publik, "
            f"aplikasi hanya akan memproses {max_rows:,} baris pertama."
        )
        df_raw = df_raw.head(max_rows).copy()

    tab_data, tab_run, tab_results = st.tabs(["Data", "Jalankan", "Hasil"])

    with tab_data:
        st.subheader("Preview & Tabel Lengkap")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total baris", f"{len(df_raw):,}")
        if "language" in df_raw.columns:
            c2.metric("Bahasa", f"{df_raw['language'].nunique():,}")
        else:
            c2.metric("Bahasa", "-")
        if "rating" in df_raw.columns:
            c3.metric("Rata-rata rating", f"{pd.to_numeric(df_raw['rating'], errors='coerce').mean():.2f}")
        else:
            c3.metric("Rata-rata rating", "-")
        c4.metric("Kolom", f"{len(df_raw.columns):,}")

        with st.expander("Filter (opsional)", expanded=True):
            query = st.text_input(
                "Cari teks (contains)",
                placeholder="contoh: ramai, mahal, respectful...",
                key="_filter_query",
            )
            fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
            with fcol1:
                languages = None
                if "language" in df_raw.columns:
                    all_lang = sorted({str(x) for x in df_raw["language"].dropna().unique().tolist()})
                    languages = st.multiselect("Language", options=all_lang, default=all_lang, key="_filter_lang")
            with fcol2:
                sentiments = None
                if LABEL_COLUMN in df_raw.columns:
                    all_sent = sorted({str(x) for x in df_raw[LABEL_COLUMN].dropna().unique().tolist()})
                    sentiments = st.multiselect(
                        "Sentiment label",
                        options=all_sent,
                        default=all_sent,
                        key="_filter_sent",
                    )
            with fcol3:
                min_rating = None
                if "rating" in df_raw.columns:
                    min_rating = st.number_input("Min rating", min_value=1, max_value=5, value=1, step=1)

        df_view = df_raw.copy()
        if query:
            df_view = df_view[df_view[TEXT_COLUMN].astype(str).str.contains(str(query), case=False, na=False)].copy()
        if languages is not None and "language" in df_view.columns:
            df_view = df_view[df_view["language"].astype(str).isin([str(x) for x in languages])].copy()
        if sentiments is not None and LABEL_COLUMN in df_view.columns:
            df_view = df_view[df_view[LABEL_COLUMN].astype(str).isin([str(x) for x in sentiments])].copy()
        if min_rating is not None and "rating" in df_view.columns:
            df_view = df_view[pd.to_numeric(df_view["rating"], errors="coerce") >= float(min_rating)].copy()

        st.caption(f"Baris setelah filter: {len(df_view):,}")
        st.dataframe(df_view, width="stretch", height=560)

        st.download_button(
            "Download CSV (setelah filter)",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="reviews_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab_run:
        st.subheader("Jalankan Analisis")
        st.caption(
            "Klik Run untuk menjalankan pipeline. Untuk demo publik yang cepat, biarkan IndoBERT nonaktif."
        )
        run = st.button("Run", type="primary", width="stretch")
        if run:
            with st.spinner("Menjalankan pipeline..."):
                try:
                    st.session_state["result"] = _run_analysis(
                        df_raw=df_raw,
                        folds=int(folds),
                        max_length=int(max_length),
                        bert_model_name=str(bert_model_name).strip(),
                        run_indobert=bool(run_indobert),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    return

    with tab_results:
        result = st.session_state.get("result")
        if not result:
            st.info("Belum ada hasil. Buka tab **Jalankan** lalu klik **Run**.")
            return

        metrics_table = result["metrics_table"]
        best_method = result["best_method"]

        st.subheader("Ringkasan")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Metode terbaik (F1)", best_method)
        col2.metric("Accuracy", f"{metrics_table.set_index('method').loc[best_method, 'accuracy']:.3f}")
        col3.metric("Precision", f"{metrics_table.set_index('method').loc[best_method, 'precision']:.3f}")
        col4.metric("Recall", f"{metrics_table.set_index('method').loc[best_method, 'recall']:.3f}")

        st.markdown("**Tabel Metrik**")
        st.dataframe(metrics_table, width="stretch")

        st.subheader("Visualisasi")
        images = result.get("images", {}) or {}
        if images:
            preferred_order = [
                "pie_sentiment_distribution.png",
                "bar_ethics_distribution.png",
                "stacked_sentiment_by_ethics.png",
                "wordcloud_reviews.png",
                "model_comparison.png",
                "cm_vader.png",
                "cm_naive_bayes.png",
                "cm_indobert.png",
            ]
            ordered_items: list[tuple[str, bytes]] = []
            for key in preferred_order:
                if key in images:
                    ordered_items.append((key, images[key]))
            for key, value in images.items():
                if key not in {k for k, _ in ordered_items}:
                    ordered_items.append((key, value))

            left, right = st.columns(2)
            for i, (name, data) in enumerate(ordered_items):
                with (left if i % 2 == 0 else right):
                    st.image(data, caption=name, width="stretch")

        st.subheader("Insight Otomatis")
        insights = result.get("insights", []) or []
        if insights:
            st.info("\n".join([f"- {ins}" for ins in insights]))
        else:
            st.caption("Belum ada insight yang bisa digenerate dari data ini.")

        st.subheader("Download Output")
        out_df = result["df"]
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download reviews_with_sentiment_ethics.csv",
            data=csv_bytes,
            file_name="reviews_with_sentiment_ethics.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "Catatan: evaluasi ML bergantung pada `sentiment_label` atau label lemah dari `rating`/lexicon. "
            "Untuk publikasi jurnal, gunakan label manual dan laporkan skema anotasi."
        )


if __name__ == "__main__":
    main()
