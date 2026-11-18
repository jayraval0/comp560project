"""Synthetic financial news classification and t-SNE visualization.

Vectorize news semantics (upgrades, downgrades, positive, negative), reduce with
Truncated SVD, visualize with t-SNE at increasing sample sizes, and save the
reusable text embedding and classification models.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s - %(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT
RANDOM_STATE = 42
CLASSES = ["upgrade", "downgrade", "positive", "negative"]
CLASS_COLORS = {
    "upgrade": "#2ca02c",
    "downgrade": "#d62728",
    "positive": "#1f77b4",
    "negative": "#ff7f0e",
}
SAMPLE_SIZES = [60, 120, 180, 240]


def build_news_corpus() -> pd.DataFrame:
    """Labeled financial headlines: upgrades, downgrades, and broader pos/neg news."""
    tickers = [
        "Apple", "Microsoft", "Amazon", "Google", "Tesla", "Nvidia", "JPMorgan",
        "Johnson & Johnson", "Visa", "Procter & Gamble", "UnitedHealth", "Mastercard",
        "Disney", "Adobe", "PayPal", "Netflix", "Intel", "Verizon", "Salesforce",
        "Pfizer", "Coca-Cola", "PepsiCo", "Costco", "Walmart", "Broadcom", "Oracle",
        "Nike", "AbbVie", "Boeing", "Goldman Sachs", "Caterpillar", "Cisco",
    ]
    brokers = [
        "Goldman Sachs", "Morgan Stanley", "JPMorgan", "Bank of America", "Barclays",
        "Citi", "Wells Fargo", "UBS", "Deutsche Bank", "Jefferies",
    ]

    rows: list[dict] = []

    upgrade_templates = [
        "{broker} upgrades {ticker} to overweight and raises its price target",
        "{broker} lifts {ticker} to buy citing accelerating earnings momentum",
        "Analyst upgrade: {ticker} moved to outperform on stronger guidance",
        "{broker} initiates {ticker} at overweight with a Street-high target",
        "{ticker} receives a double upgrade to strong buy after beat-and-raise quarter",
        "{broker} upgrades {ticker} on improving free-cash-flow outlook",
        "Street upgrade hits {ticker} as {broker} turns constructive on the sector",
        "{broker} raises {ticker} to conviction buy after channel checks improve",
        "{ticker} upgraded to overweight; target price hiked on multiple expansion",
        "{broker} upgrades {ticker} from hold to buy following a product-cycle refresh",
    ]
    downgrade_templates = [
        "{broker} downgrades {ticker} to underweight and cuts its price target",
        "{broker} slashes {ticker} to sell citing fading earnings momentum",
        "Analyst downgrade: {ticker} moved to underperform on weaker guidance",
        "{broker} initiates {ticker} at underweight with a Street-low target",
        "{ticker} receives a double downgrade to strong sell after a miss",
        "{broker} downgrades {ticker} on deteriorating free-cash-flow outlook",
        "Street downgrade hits {ticker} as {broker} turns cautious on the sector",
        "{broker} cuts {ticker} to underperform after channel checks weaken",
        "{ticker} downgraded to equal-weight; target price reduced on multiple compression",
        "{broker} downgrades {ticker} from buy to hold following a product delay",
    ]
    positive_templates = [
        "{ticker} beats earnings estimates and raises full-year guidance",
        "{ticker} reports record revenue growth and expanding operating margins",
        "{ticker} announces a larger buyback and an increase in the dividend",
        "{ticker} wins a major multi-year contract, lifting near-term demand",
        "{ticker} shares jump after a stronger-than-expected product launch",
        "{ticker} posts upside on both EPS and sales, sending the stock higher",
        "Optimistic outlook: {ticker} sees accelerating cloud and AI demand",
        "{ticker} receives regulatory approval, clearing a key growth hurdle",
        "{ticker} same-store sales surprise to the upside in the latest quarter",
        "{ticker} management sounds confident on a call, highlighting operating leverage",
    ]
    negative_templates = [
        "{ticker} misses earnings estimates and cuts full-year guidance",
        "{ticker} reports shrinking revenue and contracting operating margins",
        "{ticker} suspends its buyback and trims the dividend",
        "{ticker} loses a major contract, pressuring near-term demand",
        "{ticker} shares slide after a weaker-than-expected product launch",
        "{ticker} posts a miss on both EPS and sales, sending the stock lower",
        "Cautious outlook: {ticker} warns of slowing cloud and consumer demand",
        "{ticker} faces a regulatory setback, raising risk to the growth plan",
        "{ticker} same-store sales disappoint in the latest quarter",
        "{ticker} management sounds defensive on a call, highlighting cost pressure",
    ]

    buckets = {
        "upgrade": upgrade_templates,
        "downgrade": downgrade_templates,
        "positive": positive_templates,
        "negative": negative_templates,
    }

    for label, templates in buckets.items():
        for i, template in enumerate(templates):
            for j in range(6):
                ticker = tickers[(i + j) % len(tickers)]
                broker = brokers[(i + 2 * j) % len(brokers)]
                headline = template.format(ticker=ticker, broker=broker)
                rows.append(
                    {
                        "headline": headline,
                        "label": label,
                        "ticker": ticker,
                        "source": "seed",
                        "template_group": f"{label}:{i}",
                    }
                )

    extra = [
        ("Microsoft cloud backlog hits a record as Azure growth re-accelerates", "positive"),
        ("Nvidia data-center revenue surges on hyperscaler AI spending", "positive"),
        ("Apple services revenue reaches a new high amid installed-base growth", "positive"),
        ("Amazon AWS margins expand as cost discipline takes hold", "positive"),
        ("Tesla deliveries beat the Street as factory utilization improves", "positive"),
        ("Costco membership renewals stay elevated, supporting fee income", "positive"),
        ("Pfizer vaccine sales collapse faster than expected", "negative"),
        ("Intel delays a process node and warns of further share losses", "negative"),
        ("Boeing production halt adds to delivery and cash-flow risk", "negative"),
        ("Disney streaming losses widen as subscriber growth stalls", "negative"),
        ("PayPal checkout volume slows with weaker consumer spending", "negative"),
        ("Verizon wireless net adds disappoint as price rises bite", "negative"),
        ("Morgan Stanley upgrades Nvidia to overweight on AI demand durability", "upgrade"),
        ("Bank of America upgrades Eli Lilly on obesity-franchise momentum", "upgrade"),
        ("Jefferies upgrades Broadcom to buy after custom-chip wins", "upgrade"),
        ("Citi upgrades Walmart to overweight on market-share gains", "upgrade"),
        ("Barclays downgrades Tesla to underweight on margin compression", "downgrade"),
        ("UBS downgrades Netflix to sell on password-sharing fatigue", "downgrade"),
        ("Deutsche Bank downgrades Intel to hold after guidance cut", "downgrade"),
        ("Wells Fargo downgrades Nike to underweight on China weakness", "downgrade"),
    ]
    for i, (headline, label) in enumerate(extra):
        rows.append(
            {"headline": headline, "label": label, "ticker": "", "source": "seed",
             "template_group": f"extra:{i}"}
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["headline"]).reset_index(drop=True)
    logging.info("Seed news corpus size: %s", len(df))
    return df


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    n = min(n, len(df))
    per_class = max(n // df["label"].nunique(), 1)
    parts = []
    for _, group in df.groupby("label"):
        take = min(len(group), per_class)
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])
    sample = pd.concat(parts).sample(frac=1.0, random_state=int(rng.integers(0, 10_000)))
    if len(sample) > n:
        sample = sample.iloc[:n]
    return sample.reset_index(drop=True)


def build_embedding_pipeline(n_components: int = 40) -> Pipeline:
    """Reusable embedding model: TF-IDF news vectors -> Truncated SVD."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
        ]
    )


def fit_tsne(embeddings: np.ndarray, perplexity: float) -> np.ndarray:
    perplexity = max(5.0, min(perplexity, (len(embeddings) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return tsne.fit_transform(embeddings)


def keyword_label(headline: str) -> str:
    """Apply a fixed keyword rule; default to positive when no rule matches."""
    words = set(re.findall(r"[a-z]+", headline.lower()))
    if words & {"downgrade", "downgrades", "downgraded", "underweight", "underperform"}:
        return "downgrade"
    if words & {"upgrade", "upgrades", "upgraded", "overweight", "outperform"}:
        return "upgrade"
    if words & {"miss", "misses", "loss", "losses", "cuts", "weak", "weaker", "slows"}:
        return "negative"
    return "positive"


def evaluate_classifiers(sample: pd.DataFrame, n_components: int) -> dict:
    headlines = sample["headline"]
    labels = sample["label"].to_numpy()
    keyword_predictions = headlines.map(keyword_label).to_numpy()
    model = build_embedding_pipeline(n_components=n_components)
    model.steps.append(("classifier", LogisticRegression(max_iter=400)))
    headline_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    template_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = {
        "headline_split": list(headline_cv.split(headlines, labels)),
        "template_split": list(template_cv.split(headlines, labels, sample["template_group"])),
    }
    scores = {}
    for name, folds in splits.items():
        classifier_scores = cross_val_score(
            model, headlines, labels, cv=folds, scoring="accuracy", error_score="raise"
        )
        keyword_scores = [(keyword_predictions[test] == labels[test]).mean() for _, test in folds]
        scores[name] = {
            "classifier": float(classifier_scores.mean()),
            "keyword_baseline": float(np.mean(keyword_scores)),
        }
    return scores


def plot_tsne_panels(results: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, result in zip(axes, results):
        coords = result["coords"]
        labels = result["labels"]
        for cls in CLASSES:
            mask = labels == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=22,
                alpha=0.85,
                c=CLASS_COLORS[cls],
                label=cls,
                edgecolors="none",
            )
        ax.set_title(f"n = {result['n']}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "t-SNE embeddings of financial news\n"
        "Upgrades / downgrades and positive / negative stories",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logging.info("Saved %s", path)


def plot_classification_accuracy(results: list[dict], path: Path) -> None:
    ns = [r["n"] for r in results]
    plt.figure(figsize=(8, 4.8))
    for split, label, color in (
        ("headline_split", "Headline split", "tab:blue"),
        ("template_split", "Template holdout", "tab:orange"),
    ):
        plt.plot(ns, [r[split]["classifier"] for r in results], "o-", color=color,
                 label=f"{label} · classifier")
        plt.plot(ns, [r[split]["keyword_baseline"] for r in results], "s--", color=color,
                 label=f"{label} · keyword rule")
    plt.xlabel("Sample size")
    plt.ylabel("Mean validation accuracy")
    plt.title("Synthetic headline classification")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    logging.info("Saved %s", path)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    news = build_news_corpus()
    news_path = MODEL_DIR / "news_corpus.csv"
    news.to_csv(news_path, index=False)
    logging.info("News corpus written to %s (%s rows)", news_path, len(news))

    encoder = LabelEncoder().fit(CLASSES)
    embedding_model = build_embedding_pipeline()
    high_dim = embedding_model.fit_transform(news["headline"])
    y_all = encoder.transform(news["label"])

    joblib.dump(embedding_model, MODEL_DIR / "news_embedding_svd.joblib")
    joblib.dump(encoder, MODEL_DIR / "news_label_encoder.joblib")
    logging.info("Saved SVD embedding pipeline")

    classifier = LogisticRegression(max_iter=400)
    classifier.fit(high_dim, y_all)
    joblib.dump(classifier, MODEL_DIR / "news_classifier.joblib")
    logging.info(
        "Saved reusable embedding + classifier (in-sample fit, not the Figure 3 accuracy curve)"
    )

    results = []
    for n in SAMPLE_SIZES:
        sample = stratified_sample(news, n, rng)
        # The plot uses all sampled headlines. Evaluation fits text features inside each fold.
        n_components = min(40, max(8, len(sample) // 3))
        local_model = build_embedding_pipeline(n_components=n_components)
        sample_embed = local_model.fit_transform(sample["headline"])
        perplexity = min(30, max(5, n // 4))
        coords = fit_tsne(sample_embed, perplexity)
        scores = evaluate_classifiers(sample, n_components)
        pd.DataFrame(
            {
                "tsne_x": coords[:, 0],
                "tsne_y": coords[:, 1],
                "label": sample["label"],
                "headline": sample["headline"],
            }
        ).to_csv(MODEL_DIR / f"tsne_n{n}_embeddings.csv", index=False)
        logging.info("n=%s | scores=%s", n, scores)
        results.append(
            {
                "n": len(sample),
                "coords": coords,
                "labels": sample["label"].to_numpy(),
                **scores,
            }
        )

    plot_tsne_panels(results, FIGURE_DIR / "tsne_news_embeddings.png")
    plot_classification_accuracy(results, FIGURE_DIR / "tsne_predictive_power.png")

    metrics = {
        "corpus_size": int(len(news)),
        "classes": CLASSES,
        "evaluation": "Mean accuracy across five folds with fold-fitted preprocessing. Template holdout keeps wording variants together.",
        "sample_sizes": [
            {key: r[key] for key in ("n", "headline_split", "template_split")}
            for r in results
        ],
        "models": {
            "embedding": "models/news_embedding_svd.joblib",
            "classifier": "models/news_classifier.joblib",
            "label_encoder": "models/news_label_encoder.joblib",
            "tsne_csv": [f"models/tsne_n{n}_embeddings.csv" for n in SAMPLE_SIZES],
        },
    }
    (MODEL_DIR / "tsne_metrics.json").write_text(json.dumps(metrics, indent=2))
    logging.info("t-SNE modeling complete")


if __name__ == "__main__":
    main()
