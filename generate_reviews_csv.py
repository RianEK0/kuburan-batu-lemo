from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReviewTemplate:
    language: str
    sentiment: str  # negative|neutral|positive
    text: str


ID_TEMPLATES: list[ReviewTemplate] = [
    ReviewTemplate("id", "positive", "Tempatnya {adj_pos} dan {culture}. {respect_note}"),
    ReviewTemplate("id", "positive", "{culture_cap} terasa kuat. {staff_pos} {respect_note}"),
    ReviewTemplate("id", "positive", "Sangat {adj_pos2} untuk wisata edukasi. {guide_pos}"),
    ReviewTemplate("id", "neutral", "Lokasinya {adj_neu}, tapi {facility_neu}. {crowd_neu}"),
    ReviewTemplate("id", "neutral", "Menarik untuk dikunjungi. {facility_neu} dan {sign_neu}."),
    ReviewTemplate("id", "neutral", "Secara umum oke. {commercial_neu} {respect_note_short}"),
    ReviewTemplate("id", "negative", "Saya {adj_neg} karena {facility_neg}. {price_neg}"),
    ReviewTemplate("id", "negative", "Terlalu {crowd_neg} dan jadi {comfort_neg}. {behavior_neg}"),
    ReviewTemplate("id", "negative", "{commercial_neg}. Untuk tempat sakral, ini terasa {adj_neg2}."),
]

EN_TEMPLATES: list[ReviewTemplate] = [
    ReviewTemplate("en", "positive", "The place is {adj_pos} and the {culture} is strong. {respect_note}"),
    ReviewTemplate("en", "positive", "Amazing cultural value and {view_pos}. {guide_pos}"),
    ReviewTemplate("en", "positive", "A sacred site with {adj_pos2} history. {respect_note}"),
    ReviewTemplate("en", "neutral", "Interesting visit, but {facility_neu}. {crowd_neu}"),
    ReviewTemplate("en", "neutral", "Nice overall. {sign_neu} and {commercial_neu}"),
    ReviewTemplate("en", "neutral", "Worth a stop if you're nearby. {respect_note_short}"),
    ReviewTemplate("en", "negative", "Not worth it: {price_neg} and {facility_neg}."),
    ReviewTemplate("en", "negative", "Too {crowd_neg}, hard to enjoy. {behavior_neg}"),
    ReviewTemplate("en", "negative", "{commercial_neg}. It feels {adj_neg} for a sacred place."),
]


WORD_BANK = {
    "id": {
        "adj_pos": ["indah", "bagus", "unik", "luar biasa", "cantik"],
        "adj_pos2": ["rekomendasi", "keren", "menarik", "hebat"],
        "adj_neu": ["cukup menarik", "lumayan", "oke"],
        "adj_neg": ["kecewa", "kurang puas", "tidak nyaman"],
        "adj_neg2": ["kurang etis", "tidak pantas", "mengganggu"],
        "culture": ["budayanya terasa kuat", "nilai budaya Toraja terasa", "tradisi lokal terasa hidup"],
        "culture_cap": ["Nilai budaya", "Tradisi", "Budaya Toraja"],
        "staff_pos": ["Petugas ramah dan membantu.", "Warga lokal ramah.", "Pengelola informatif."],
        "guide_pos": ["Pemandu menjelaskan dengan baik.", "Guide-nya informatif.", "Penjelasan sejarahnya jelas."],
        "facility_neu": ["fasilitasnya standar", "area parkir sempit", "toiletnya biasa saja"],
        "facility_neg": ["toilet kotor", "fasilitas kurang terawat", "jalur akses rusak di beberapa titik"],
        "sign_neu": ["papan informasinya kurang", "informasi aturan belum jelas", "petunjuk arah minim"],
        "crowd_neu": ["cukup ramai saat jam tertentu.", "lebih enak datang pagi.", "tergantung hari kunjungan."],
        "crowd_neg": ["ramai sekali", "terlalu padat", "penuh rombongan besar"],
        "comfort_neg": ["kurang nyaman", "susah menikmati suasana", "bising"],
        "price_neg": ["Harga ticket mahal.", "Ticket terasa tidak sepadan.", "Biaya masuk cukup mahal."],
        "commercial_neu": ["ada unsur komersial, tapi masih wajar.", "ticketing cukup tertata.", "ada beberapa kios."],
        "commercial_neg": ["Terlalu komersial", "Banyak upsell dan kios", "Nuansa komersialisasi berlebihan"],
        "behavior_neg": [
            "Banyak pengunjung tidak sopan saat berfoto.",
            "Perilaku pengunjung mengganggu.",
            "Attitude beberapa rombongan buruk.",
        ],
        "respect_note": [
            "Pengunjung sebaiknya hormat dan sopan.",
            "Tolong jaga etika saat berkunjung.",
            "Mohon respect tradisi setempat.",
        ],
        "respect_note_short": ["Jaga sopan santun.", "Hormati tradisi.", "Respect ya."],
    },
    "en": {
        "adj_pos": ["beautiful", "unique", "amazing", "great"],
        "adj_pos2": ["meaningful", "fascinating", "wonderful"],
        "view_pos": ["beautiful views", "great scenery", "a peaceful atmosphere"],
        "adj_neu": ["okay", "decent", "fine"],
        "adj_neg": ["wrong", "disappointing", "unpleasant"],
        "culture": ["cultural heritage", "local traditions", "Toraja culture"],
        "guide_pos": ["The guide was helpful.", "Great explanations from the guide.", "Informative tour."],
        "facility_neu": ["facilities are average", "parking is limited", "restrooms are basic"],
        "facility_neg": ["poor facilities", "dirty restrooms", "unclear access path"],
        "sign_neu": ["signage is unclear", "rules are not well explained", "not enough information boards"],
        "crowd_neu": ["it can get busy at noon.", "go early for fewer crowds.", "depends on the day."],
        "crowd_neg": ["crowded", "packed", "overcrowded"],
        "price_neg": ["ticket price is expensive", "not worth the ticket price", "too pricey for what you get"],
        "commercial_neu": ["some commercialization, but manageable.", "ticketing is organized.", "a few stalls around."],
        "commercial_neg": ["Too commercial", "Upsells everywhere", "Over-commercialized"],
        "behavior_neg": [
            "Some visitors had bad behavior.",
            "Visitor behavior ruined the experience.",
            "A few groups were disrespectful.",
        ],
        "respect_note": [
            "Visitors should be respectful.",
            "Please respect local traditions.",
            "Be respectful and behave properly.",
        ],
        "respect_note_short": ["Be respectful.", "Respect traditions.", "Mind your behavior."],
    },
}


def _choice(rng: random.Random, items: list[str]) -> str:
    return str(rng.choice(items))


def _render_template(rng: random.Random, template: ReviewTemplate) -> str:
    bank = WORD_BANK[template.language]

    return (
        template.text.format(
            adj_pos=_choice(rng, bank.get("adj_pos", ["good"])),
            adj_pos2=_choice(rng, bank.get("adj_pos2", ["good"])),
            adj_neu=_choice(rng, bank.get("adj_neu", ["okay"])),
            adj_neg=_choice(rng, bank.get("adj_neg", ["bad"])),
            adj_neg2=_choice(rng, bank.get("adj_neg2", ["bad"])),
            culture=_choice(rng, bank.get("culture", ["culture"])),
            culture_cap=_choice(rng, bank.get("culture_cap", ["Culture"])),
            staff_pos=_choice(rng, bank.get("staff_pos", ["Staff are friendly."])),
            guide_pos=_choice(rng, bank.get("guide_pos", ["Good guide."])),
            view_pos=_choice(rng, bank.get("view_pos", ["nice views"])),
            facility_neu=_choice(rng, bank.get("facility_neu", ["facilities are okay"])),
            facility_neg=_choice(rng, bank.get("facility_neg", ["poor facilities"])),
            sign_neu=_choice(rng, bank.get("sign_neu", ["signage could be better"])),
            crowd_neu=_choice(rng, bank.get("crowd_neu", ["it can get busy"])),
            crowd_neg=_choice(rng, bank.get("crowd_neg", ["crowded"])),
            comfort_neg=_choice(rng, bank.get("comfort_neg", ["uncomfortable"])),
            price_neg=_choice(rng, bank.get("price_neg", ["ticket is expensive"])),
            commercial_neu=_choice(rng, bank.get("commercial_neu", ["some commercialization"])),
            commercial_neg=_choice(rng, bank.get("commercial_neg", ["too commercial"])),
            behavior_neg=_choice(rng, bank.get("behavior_neg", ["bad behavior"])),
            respect_note=_choice(rng, bank.get("respect_note", ["please be respectful"])),
            respect_note_short=_choice(rng, bank.get("respect_note_short", ["be respectful"])),
        )
        .replace("  ", " ")
        .strip()
    )


def _pick_sentiment(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.55:
        return "positive"
    if roll < 0.75:
        return "negative"
    return "neutral"


def _pick_rating(rng: random.Random, sentiment: str) -> int:
    if sentiment == "positive":
        return int(rng.choice([4, 5, 5]))
    if sentiment == "neutral":
        return 3
    return int(rng.choice([1, 2, 2]))


def _pick_date(rng: random.Random, today: date) -> str:
    days_back = int(rng.randint(0, 365 * 2))
    d = today - timedelta(days=days_back)
    return d.isoformat()


def generate_reviews(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(int(seed))
    today = date.today()

    rows: list[dict[str, object]] = []
    for _ in range(int(n)):
        language = "id" if rng.random() < 0.7 else "en"
        sentiment = _pick_sentiment(rng)
        rating = _pick_rating(rng, sentiment)
        template_pool = ID_TEMPLATES if language == "id" else EN_TEMPLATES
        candidates = [t for t in template_pool if t.sentiment == sentiment]
        template = rng.choice(candidates)
        text = _render_template(rng, template)
        rows.append(
            {
                "review_text": text,
                "rating": rating,
                "date": _pick_date(rng, today),
                "sentiment_label": sentiment,
                "source": "synthetic",
                "language": language,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tourism reviews CSV (1000+ rows).")
    parser.add_argument("--n", type=int, default=1200, help="Number of rows (default: 1200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--out", type=str, default="reviews_1200.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = generate_reviews(n=int(args.n), seed=int(args.seed))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
