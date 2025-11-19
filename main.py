import requests
import numpy as np
from flask import Flask, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# نموذج خفيف مناسب لـ Render CPU
model = SentenceTransformer("all-MiniLM-L6-v2")

ADS_URL = "https://yaznbook-mirosurvies.onrender.com/ads"


def fetch_ads():
    try:
        res = requests.get(ADS_URL, timeout=10)
        data = res.json()
        return data.get("ads", [])
    except Exception as e:
        print("Error fetching ads:", e)
        return []


def create_text(ad):
    parts = [
        ad.get("name", ""),
        ad.get("advertiser", ""),
        ad.get("url", "")
    ]
    return " ".join([p for p in parts if p])


def rank_ads(ads):
    if not ads:
        return []

    # النصوص اللي بنعمل لها emb
    texts = [create_text(a) for a in ads]

    embeddings = model.encode(texts, convert_to_tensor=True)

    # حساب score باستخدام متوسط التشابه (جودة الإعلان)
    similarity_matrix = util.cos_sim(embeddings, embeddings)

    scores = similarity_matrix.mean(dim=1).cpu().numpy()

    # ربط كل إعلان مع سكور
    ranked = []
    for idx, ad in enumerate(ads):
        ranked.append({
            "score": float(scores[idx]),
            "ad": ad
        })

    # ترتيب من الأفضل للأسوأ
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


@app.route("/")
def home():
    return jsonify({"message": "Yaznbook AI Ads Ranking Running"})


@app.route("/best-ads")
def best_ads():
    ads = fetch_ads()
    ranked = rank_ads(ads)

    # فقط الإعلانات بدون سكور
    clean_ads = [item["ad"] for item in ranked]

    return jsonify({
        "count": len(clean_ads),
        "bestAds": clean_ads
    })


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
