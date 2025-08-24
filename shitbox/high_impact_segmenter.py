
"""
High-Impact News Segmentation with FinBERT
------------------------------------------

Given a CSV with at least columns: title, url
this script:
  1) fetches the article body (best-effort)
  2) runs FinBERT on title and (optionally) on the article lead
  3) computes an "impact_score" using multiple criteria
  4) flags which articles deserve full-body processing

USAGE:
  python high_impact_segmenter.py --input news.csv --output tagged_news.csv \
      --tickers AAPL,TSLA,NVDA --high-cred reuters.com,bloomberg.com,wsj.com \
      --min-impact 0.55

Install deps (suggested):
  pip install transformers torch requests beautifulsoup4 readability-lxml newspaper3k tldextract pandas python-dateutil

CSV format expected (header names case-insensitive):
  title, url [, published_at]

Notes:
- FinBERT max length ~512 tokens. We run title + first-chunk of body (configurable).
- If article extraction fails, we still score using title + metadata.
"""

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# Model imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Optional: content extraction helpers
try:
    from newspaper import Article  # newspaper3k
    HAS_NEWSPAPER = True
except Exception:
    HAS_NEWSPAPER = False

try:
    from readability import Document
    HAS_READABILITY = True
except Exception:
    HAS_READABILITY = False


FINBERT_MODEL_NAME = "ProsusAI/finbert"  # 3-way: negative, neutral, positive (in that logit order)


HIGH_IMPACT_KEYWORDS = {
    # keyword : weight (0..1)
    "downgrade": 0.25,
    "upgrade": 0.2,
    "lawsuit": 0.35,
    "sues": 0.35,
    "sued": 0.35,
    "antitrust": 0.35,
    "investigation": 0.25,
    "probe": 0.25,
    "sec": 0.3,
    "earnings": 0.4,
    "guidance": 0.25,
    "merger": 0.45,
    "acquisition": 0.45,
    "m&a": 0.45,
    "buyback": 0.25,
    "dividend": 0.25,
    "bankruptcy": 0.5,
    "chapter 11": 0.5,
    "layoffs": 0.35,
    "strike": 0.3,
    "recall": 0.3,
    "forecast": 0.2,
    "outlook": 0.2,
    "rating cut": 0.3,
    "price target": 0.25,
    "ipo": 0.35,
    "sec filing": 0.25,
    "fda": 0.35,
    "guidance cut": 0.4,
    "guidance raise": 0.4,
}

DEFAULT_HIGH_CRED = {"reuters.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "cnbc.com"}


def load_finbert(device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_polarity(text: str, tokenizer, model, device: str, max_length: int = 512) -> Dict[str, float]:
    """Return FinBERT probs and a scalar polarity in [-1,1] = P(pos) - P(neg)."""
    if not text:
        return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "polarity": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    # ProsusAI/finbert order: [negative, neutral, positive]
    neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
    polarity = pos - neg
    return {"negative": neg, "neutral": neu, "positive": pos, "polarity": polarity}


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    return domain.lower()


def try_extract_with_newspaper(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    if not HAS_NEWSPAPER:
        return None, None
    try:
        art = Article(url)
        art.download()
        art.parse()
        txt = art.text or None
        pub = art.publish_date
        if isinstance(pub, datetime):
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
        else:
            pub = None
        return txt, pub
    except Exception:
        return None, None


def try_extract_with_readability(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    if not HAS_READABILITY:
        return None, None
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        doc = Document(resp.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n").strip() or None

        # Try meta for date
        soup_full = BeautifulSoup(resp.text, "html.parser")
        pub = None
        for tag in soup_full.find_all(["meta", "time"]):
            for attr in ["content", "datetime", "date", "pubdate"]:
                v = tag.get(attr)
                if v:
                    try:
                        pub = dateparser.parse(v)
                        break
                    except Exception:
                        pass
            if pub:
                break

        if isinstance(pub, datetime) and pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        return text, pub
    except Exception:
        return None, None


def get_article_text_and_date(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    txt, pub = try_extract_with_newspaper(url)
    if not txt:
        txt, pub = try_extract_with_readability(url)
    return txt, pub


def first_meaningful_chunk(text: str, tokenizer, max_tokens: int = 480) -> str:
    """Return up to ~max_tokens worth of content (rough approximation using words)."""
    if not text:
        return ""
    # Roughly limit by words; tokenizer will still enforce true token cap
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def keyword_signal(text: str, keywords: Dict[str, float]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw, w in keywords.items():
        if kw in t:
            score += w
    return min(score, 1.0)


def entity_signal(title: str, tickers: List[str]) -> float:
    if not tickers:
        return 0.0
    title_up = " " + re.sub(r"[^A-Za-z0-9]+", " ", title.upper()) + " "
    found = 0
    for t in tickers:
        t_clean = t.strip().upper()
        if not t_clean:
            continue
        if f" {t_clean} " in title_up:
            found += 1
    if found == 0:
        return 0.0
    # 0.3 base for one ticker, +0.1 per additional up to 0.6
    return min(0.3 + 0.1 * (found - 1), 0.6)


def recency_signal(published_at: Optional[datetime], now: Optional[datetime] = None) -> float:
    """1.0 if within 6 hours, decays to ~0 after ~7 days."""
    if not published_at:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)
    delta_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
    if delta_hours <= 6:
        return 1.0
    if delta_hours >= 24 * 7:
        return 0.0
    # linear decay between 6h and 7d
    span = 24 * 7 - 6
    return max(0.0, 1.0 - (delta_hours - 6) / span)


def source_signal(domain: str, high_cred: set) -> float:
    if not domain:
        return 0.0
    return 0.4 if domain in high_cred else 0.0


def compute_impact_score(
    title_polarity: float,
    title_kw: float,
    entity_sig: float,
    recency_sig: float,
    source_sig: float,
    divergence: float,
    weights: Dict[str, float],
) -> float:
    # All components are 0..1 except polarity [-1,1] -> use absolute
    apol = abs(title_polarity)  # clickbait could inflate; still informative
    score = (
        weights["w_title_sent"] * apol
        + weights["w_keywords"] * title_kw
        + weights["w_entity"] * entity_sig
        + weights["w_recency"] * recency_sig
        + weights["w_source"] * source_sig
        + weights["w_divergence"] * max(0.0, divergence - weights["divergence_tau"])
    )
    return max(0.0, min(score, 1.0))


def main():
    parser = argparse.ArgumentParser(description="High-impact article segmentation with FinBERT")
    parser.add_argument("--input", required=True, help="Input CSV with columns: title,url[,published_at]")
    parser.add_argument("--output", required=True, help="Output CSV with added columns")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers of interest (e.g., AAPL,TSLA)")
    parser.add_argument("--high-cred", default="", help="Comma-separated high-cred domains")
    parser.add_argument("--min-impact", type=float, default=0.55, help="Threshold for high_impact flag (0..1)")
    parser.add_argument("--analyze-lead", action="store_true", help="Also run FinBERT on first chunk of body")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout (s)")
    parser.add_argument("--user-agent", default="Mozilla/5.0", help="HTTP User-Agent")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else []
    high_cred = set([d.strip().lower() for d in args.high_cred.split(",") if d.strip()]) or set(DEFAULT_HIGH_CRED)

    # Load data
    df = pd.read_csv(args.input)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic checks
    if "title" not in df.columns or "url" not in df.columns:
        print("ERROR: input CSV must contain at least 'title' and 'url' columns.", file=sys.stderr)
        sys.exit(1)

    # Load FinBERT
    tokenizer, model, device = load_finbert()

    # Prepare outputs
    out_rows = []
    now_utc = datetime.now(timezone.utc)

    # Weights (tweak to taste)
    weights = dict(
        w_title_sent=0.15,
        w_keywords=0.25,
        w_entity=0.2,
        w_recency=0.2,
        w_source=0.1,
        w_divergence=0.25,
        divergence_tau=0.15,  # only count divergence above this
    )

    sess = requests.Session()
    sess.headers.update({"User-Agent": args.user_agent})

    for idx, row in df.iterrows():
        title = str(row.get("title", "") or "")
        url = str(row.get("url", "") or "")
        published_at_raw = row.get("published_at", None)

        # Parse provided date if exists
        published_at = None
        if isinstance(published_at_raw, str) and published_at_raw.strip():
            try:
                published_at = dateparser.parse(published_at_raw)
                if published_at and published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                published_at = None

        domain = extract_domain(url) if url else ""

        # Title sentiment
        title_sent = finbert_polarity(title, tokenizer, model, device)
        title_polarity = title_sent["polarity"]

        # Fetch article and extract body+date
        body_text = None
        body_pub = None
        try:
            body_text, body_pub = get_article_text_and_date(url)
        except Exception:
            body_text, body_pub = None, None

        if not published_at and body_pub:
            published_at = body_pub

        # Lead sentiment (optional)
        lead_sent = {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "polarity": 0.0}
        if args.analyze_lead and body_text:
            lead = first_meaningful_chunk(body_text, tokenizer, max_tokens=480)
            lead_sent = finbert_polarity(lead, tokenizer, model, device)

        divergence = abs(title_sent["polarity"] - lead_sent["polarity"])

        # Signals
        kw_sig = max(keyword_signal(title, HIGH_IMPACT_KEYWORDS), keyword_signal(body_text or "", HIGH_IMPACT_KEYWORDS))
        ent_sig = entity_signal(title, tickers)
        rec_sig = recency_signal(published_at, now_utc)
        src_sig = source_signal(domain, high_cred)

        impact = compute_impact_score(
            title_polarity=title_polarity,
            title_kw=kw_sig,
            entity_sig=ent_sig,
            recency_sig=rec_sig,
            source_sig=src_sig,
            divergence=divergence,
            weights=weights,
        )
        high_impact = impact >= args.min_impact

        out = {
            "title": title,
            "url": url,
            "domain": domain,
            "published_at": published_at.isoformat() if isinstance(published_at, datetime) else "",
            "title_neg": round(title_sent["negative"], 4),
            "title_neu": round(title_sent["neutral"], 4),
            "title_pos": round(title_sent["positive"], 4),
            "title_polarity": round(title_polarity, 4),
            "lead_neg": round(lead_sent["negative"], 4),
            "lead_neu": round(lead_sent["neutral"], 4),
            "lead_pos": round(lead_sent["positive"], 4),
            "lead_polarity": round(lead_sent["polarity"], 4),
            "divergence": round(divergence, 4),
            "keyword_sig": round(kw_sig, 4),
            "entity_sig": round(ent_sig, 4),
            "recency_sig": round(rec_sig, 4),
            "source_sig": round(src_sig, 4),
            "impact_score": round(impact, 4),
            "high_impact": bool(high_impact),
            "why": "; ".join(
                [s for s in [
                    f"kw={kw_sig:.2f}" if kw_sig > 0 else "",
                    f"ent={ent_sig:.2f}" if ent_sig > 0 else "",
                    f"rec={rec_sig:.2f}" if rec_sig > 0 else "",
                    f"src={src_sig:.2f}" if src_sig > 0 else "",
                    f"div={divergence:.2f}" if divergence > 0.2 else "",
                    f"|pol|={abs(title_polarity):.2f}" if abs(title_polarity) > 0.4 else "",
                ] if s]
            )
        }
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Columns: ", ", ".join(out_df.columns))


if __name__ == "__main__":
    main()
