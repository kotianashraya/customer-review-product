
import io
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ================================================================
# CUSTOMER REVIEW INTELLIGENCE SYSTEM
# NLP + MACHINE LEARNING + BUSINESS ANALYTICS
# ================================================================

st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# PREMIUM DASHBOARD STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1d4ed8 55%, #38bdf8 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(30, 64, 175, .20);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        margin: .6rem 0 0;
        font-size: 1.05rem;
        opacity: .90;
    }

    .metric-card {
        background: white;
        padding: 1.2rem 1.3rem;
        border-radius: 18px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 7px 20px rgba(15, 23, 42, .06);
        min-height: 120px;
    }

    .metric-label {
        color: #64748b;
        font-size: .86rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 1.5rem 0 .7rem;
    }

    .insight-card {
        background: white;
        border-left: 5px solid #2563eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: .7rem;
        box-shadow: 0 5px 16px rgba(15, 23, 42, .05);
    }

    .review-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .05);
    }

    .tag {
        display: inline-block;
        padding: .25rem .65rem;
        margin: .15rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: .78rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .small-note {
        color: #64748b;
        font-size: .82rem;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# 1. SAMPLE DATA
# ================================================================

SAMPLE_REVIEWS = [
    ("Excellent camera quality and the battery lasts all day", "Positive"),
    ("Amazing sound quality and very comfortable headphones", "Positive"),
    ("The delivery was fast and packaging was excellent", "Positive"),
    ("I love the display quality, it is bright and sharp", "Positive"),
    ("Great product, the battery backup is fantastic", "Positive"),
    ("The laptop performance is excellent and very fast", "Positive"),
    ("Customer service was helpful and solved my issue quickly", "Positive"),
    ("Very good build quality and premium design", "Positive"),
    ("The camera takes beautiful photos in low light", "Positive"),
    ("Excellent value for money and smooth performance", "Positive"),
    ("The app is easy to use and works perfectly", "Positive"),
    ("The product arrived early and was safely packed", "Positive"),
    ("I am impressed with the quality and performance", "Positive"),
    ("The keyboard is comfortable and responsive", "Positive"),
    ("Battery life is much better than expected", "Positive"),
    ("Fantastic product, would definitely recommend it", "Positive"),
    ("The screen is beautiful and colors are accurate", "Positive"),
    ("Setup was simple and the instructions were clear", "Positive"),
    ("The device is lightweight and easy to carry", "Positive"),
    ("Support team was polite and very responsive", "Positive"),

    ("The battery drains very quickly and the phone gets hot", "Negative"),
    ("Terrible camera quality and blurry photos", "Negative"),
    ("The delivery was late and the package was damaged", "Negative"),
    ("Sound quality is poor and the headphones are uncomfortable", "Negative"),
    ("The laptop is extremely slow and freezes often", "Negative"),
    ("Customer service was rude and did not solve my problem", "Negative"),
    ("The screen has poor brightness and bad colors", "Negative"),
    ("Very disappointing product for this price", "Negative"),
    ("The app crashes frequently and is difficult to use", "Negative"),
    ("Build quality is cheap and the device feels fragile", "Negative"),
    ("Battery backup is horrible and needs charging twice a day", "Negative"),
    ("The product stopped working after one week", "Negative"),
    ("The packaging was damaged when it arrived", "Negative"),
    ("Performance is poor and applications take forever to open", "Negative"),
    ("The instructions are confusing and setup was difficult", "Negative"),
    ("I regret buying this product", "Negative"),
    ("The keyboard keys are stiff and uncomfortable", "Negative"),
    ("The camera is disappointing for the price", "Negative"),
    ("Delivery service was extremely slow", "Negative"),
    ("Support never responded to my complaint", "Negative"),

    ("The phone camera is average for this price", "Neutral"),
    ("Battery life is acceptable but nothing special", "Neutral"),
    ("The delivery arrived in five days as expected", "Neutral"),
    ("Sound quality is decent for casual listening", "Neutral"),
    ("The laptop performance is okay for basic tasks", "Neutral"),
    ("Customer service answered my question", "Neutral"),
    ("The display is fine for normal usage", "Neutral"),
    ("The product is similar to other products in this range", "Neutral"),
    ("The app provides the basic features I need", "Neutral"),
    ("Packaging was normal and the product arrived safely", "Neutral"),
    ("The design is simple and practical", "Neutral"),
    ("Setup took around fifteen minutes", "Neutral"),
    ("The keyboard works as expected", "Neutral"),
    ("Camera performance is reasonable in daylight", "Neutral"),
    ("The product is average overall", "Neutral"),
    ("Delivery timing was as mentioned on the website", "Neutral"),
    ("The device has standard features", "Neutral"),
    ("The quality is acceptable considering the price", "Neutral"),
    ("Support provided the requested information", "Neutral"),
    ("Performance is suitable for normal use", "Neutral")
]

DEMO_DF = pd.DataFrame(SAMPLE_REVIEWS, columns=["review", "sentiment"])


# ================================================================
# 2. TEXT PROCESSING
# ================================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text.lower())
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_sentiment(value):
    value = str(value).strip().lower()

    mapping = {
        "positive": "Positive",
        "pos": "Positive",
        "1": "Positive",
        "true": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "0": "Negative",
        "false": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "mixed": "Neutral"
    }

    return mapping.get(value, None)


# ================================================================
# 3. DATASET DETECTION
# ================================================================

TEXT_COLUMNS = [
    "review", "reviews", "review_text", "reviewtext", "text",
    "comment", "comments", "content", "feedback", "review_body",
    "reviewbody", "description"
]

SENTIMENT_COLUMNS = [
    "sentiment", "label", "polarity", "class", "emotion"
]

RATING_COLUMNS = [
    "rating", "ratings", "score", "stars", "star_rating",
    "overall", "review_rating"
]


def find_column(df, candidates):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for col in df.columns:
        normalized_col = str(col).strip().lower().replace(" ", "_")
        if any(candidate in normalized_col for candidate in candidates):
            return col

    return None


def prepare_uploaded_dataset(uploaded_file):
    raw = pd.read_csv(uploaded_file)

    if raw.empty:
        raise ValueError("The uploaded CSV is empty.")

    text_col = find_column(raw, TEXT_COLUMNS)
    sentiment_col = find_column(raw, SENTIMENT_COLUMNS)
    rating_col = find_column(raw, RATING_COLUMNS)

    if text_col is None:
        object_cols = raw.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            text_col = max(
                object_cols,
                key=lambda c: raw[c].astype(str).str.len().mean()
            )

    if text_col is None:
        raise ValueError(
            "Could not identify the review-text column. "
            "Use a column such as review, review_text, text, or comment."
        )

    result = pd.DataFrame()
    result["review"] = raw[text_col].astype(str).fillna("").str.strip()
    result = result[result["review"].str.len() >= 3].copy()

    # Existing sentiment labels
    if sentiment_col is not None:
        result["sentiment"] = raw.loc[result.index, sentiment_col].map(normalize_sentiment)

    # Rating-to-sentiment conversion
    elif rating_col is not None:
        ratings = pd.to_numeric(raw.loc[result.index, rating_col], errors="coerce")
        result["rating"] = ratings

        def rating_to_sentiment(x):
            if pd.isna(x):
                return np.nan
            if x <= 2:
                return "Negative"
            if x == 3:
                return "Neutral"
            return "Positive"

        result["sentiment"] = ratings.map(rating_to_sentiment)

    else:
        raise ValueError(
            "The dataset needs either a sentiment/label column or a rating column."
        )

    result = result.dropna(subset=["sentiment"])
    result["clean_review"] = result["review"].apply(clean_text)
    result = result[result["clean_review"].str.len() >= 3]
    result = result.drop_duplicates(subset=["clean_review"]).reset_index(drop=True)

    if len(result) < 30:
        raise ValueError("At least 30 valid reviews are required for model training.")

    if result["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    return result, text_col, sentiment_col, rating_col


# ================================================================
# 4. MODEL
# ================================================================

@st.cache_resource(show_spinner=False)
def train_model(reviews_tuple, labels_tuple):
    X = pd.Series(reviews_tuple)
    y = pd.Series(labels_tuple)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=50000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Negative", "Neutral", "Positive"]
    )

    return pipeline, accuracy, report, matrix, len(X_train), len(X_test)


# ================================================================
# 5. ASPECT INTELLIGENCE
# ================================================================

ASPECT_KEYWORDS = {
    "Battery": ["battery", "backup", "charging", "charge", "drain"],
    "Camera": ["camera", "photo", "photos", "picture", "pictures", "image", "images", "blurry"],
    "Display": ["display", "screen", "brightness", "colors", "colour"],
    "Performance": ["performance", "fast", "slow", "freezes", "speed", "applications", "crashes"],
    "Audio": ["sound", "audio", "headphones", "speaker", "music"],
    "Delivery": ["delivery", "delivered", "arrived", "shipping", "shipment"],
    "Packaging": ["packaging", "package", "packed", "box"],
    "Customer Service": ["customer service", "support", "complaint", "responded", "helpdesk"],
    "Build Quality": ["build", "quality", "fragile", "design", "material"],
    "Price / Value": ["price", "value", "cost", "money", "expensive", "worth"],
    "Usability": ["app", "setup", "instructions", "easy", "difficult", "keyboard", "interface"]
}

URGENT_WORDS = {
    "urgent", "immediately", "never", "broken", "stopped",
    "refund", "fraud", "dangerous", "damaged", "complaint",
    "rude", "horrible", "extremely", "terrible", "unsafe"
}

STOP_WORDS = {
    "the", "and", "was", "were", "is", "are", "this", "that",
    "for", "with", "very", "but", "not", "my", "it", "a", "an",
    "to", "of", "in", "on", "as", "i", "have", "has", "had",
    "be", "than", "from", "our", "your", "product", "phone"
}


def extract_aspects(text):
    lower = str(text).lower()
    found = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in keywords):
            found.append(aspect)

    return found if found else ["General"]


def aspect_sentiment(review, aspect, model):
    if aspect == "General":
        snippets = [review]
    else:
        keywords = ASPECT_KEYWORDS[aspect]
        sentences = re.split(r"[.!?]", str(review))
        snippets = [
            s.strip()
            for s in sentences
            if any(re.search(r"\b" + re.escape(k) + r"\b", s.lower()) for k in keywords)
        ]

        if not snippets:
            snippets = [review]

    predictions = model.predict([clean_text(s) for s in snippets])
    counts = Counter(predictions)

    return counts.most_common(1)[0][0]


def calculate_priority(review, sentiment):
    lower = str(review).lower()

    if sentiment == "Negative" and any(
        re.search(r"\b" + re.escape(word) + r"\b", lower)
        for word in URGENT_WORDS
    ):
        return "HIGH"

    if sentiment == "Negative":
        return "MEDIUM"

    return "LOW"


def extract_keywords(text, max_words=8):
    words = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
    words = [w for w in words if w not in STOP_WORDS]

    counts = Counter(words)
    return [word for word, _ in counts.most_common(max_words)]


# ================================================================
# 6. REVIEW ANALYSIS
# ================================================================

def analyze_review(review, model):
    cleaned = clean_text(review)

    probabilities = model.predict_proba([cleaned])[0]
    classes = model.classes_

    sentiment = classes[int(np.argmax(probabilities))]
    confidence = float(np.max(probabilities) * 100)

    aspects = extract_aspects(review)

    aspect_results = []
    for aspect in aspects:
        aspect_results.append({
            "Aspect": aspect,
            "Sentiment": aspect_sentiment(review, aspect, model)
        })

    priority = calculate_priority(review, sentiment)
    keywords = extract_keywords(review)

    if sentiment == "Positive":
        recommendation = "Maintain this strength and reinforce it in the customer experience."
    elif sentiment == "Negative":
        recommendation = "Investigate the reported issue and prioritize corrective action."
    else:
        recommendation = "Monitor this area for repeated feedback and emerging trends."

    return {
        "Review": review,
        "Sentiment": sentiment,
        "Confidence": round(confidence, 2),
        "Detected Aspects": ", ".join(aspects),
        "Priority": priority,
        "Keywords": ", ".join(keywords),
        "Recommendation": recommendation,
        "Aspect Results": aspect_results
    }


# ================================================================
# 7. BATCH ANALYSIS
# ================================================================

def analyze_dataset(df, model):
    output = df.copy()

    output["Predicted Sentiment"] = model.predict(output["clean_review"])

    probabilities = model.predict_proba(output["clean_review"])
    output["Confidence"] = np.max(probabilities, axis=1) * 100

    output["Detected Aspects"] = output["review"].apply(
        lambda x: ", ".join(extract_aspects(x))
    )

    output["Priority"] = [
        calculate_priority(review, sentiment)
        for review, sentiment in zip(
            output["review"],
            output["Predicted Sentiment"]
        )
    ]

    output["Keywords"] = output["review"].apply(
        lambda x: ", ".join(extract_keywords(x))
    )

    return output


# ================================================================
# 8. DATA PREPARATION
# ================================================================

st.sidebar.markdown("## 📁 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload a review CSV",
    type=["csv"],
    help="Use a CSV containing review text plus either sentiment labels or ratings."
)

max_training_rows = st.sidebar.slider(
    "Maximum training reviews",
    min_value=1000,
    max_value=50000,
    value=20000,
    step=1000
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Expected dataset")
st.sidebar.caption(
    "Accepted text columns include review, review_text, text, comment, "
    "content or feedback. A sentiment/label column or a 1–5 rating column "
    "can be used for supervised training."
)

# Load dataset
if uploaded_file is not None:
    try:
        df, text_col, sentiment_col, rating_col = prepare_uploaded_dataset(uploaded_file)
        source_name = uploaded_file.name
        source_type = "Uploaded real-world dataset"
    except Exception as exc:
        st.error(f"Dataset error: {exc}")
        st.stop()
else:
    df = DEMO_DF.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    text_col = "review"
    sentiment_col = "sentiment"
    rating_col = None
    source_name = "Built-in demonstration dataset"
    source_type = "Demonstration dataset"

# Limit training size while preserving class proportions
if len(df) > max_training_rows:
    training_df, _ = train_test_split(
        df,
        train_size=max_training_rows,
        random_state=42,
        stratify=df["sentiment"]
    )
else:
    training_df = df.copy()

# Train
with st.spinner("Training the review intelligence model..."):
    model, accuracy, report, matrix, train_count, test_count = train_model(
        tuple(training_df["clean_review"]),
        tuple(training_df["sentiment"])
    )

# Analyze the full loaded dataset
with st.spinner("Generating review insights..."):
    results_df = analyze_dataset(df, model)

# ================================================================
# 9. PRE-COMPUTED ANALYTICS
# ================================================================

total_reviews = len(results_df)
positive_count = int((results_df["Predicted Sentiment"] == "Positive").sum())
negative_count = int((results_df["Predicted Sentiment"] == "Negative").sum())
neutral_count = int((results_df["Predicted Sentiment"] == "Neutral").sum())
high_priority = int((results_df["Priority"] == "HIGH").sum())

positive_pct = positive_count / total_reviews * 100
negative_pct = negative_count / total_reviews * 100
neutral_pct = neutral_count / total_reviews * 100

aspect_records = []

for review, sentiment in zip(
    results_df["review"],
    results_df["Predicted Sentiment"]
):
    for aspect in extract_aspects(review):
        aspect_records.append({
            "Aspect": aspect,
            "Sentiment": sentiment
        })

aspect_df = pd.DataFrame(aspect_records)

if not aspect_df.empty:
    aspect_counts = aspect_df["Aspect"].value_counts().reset_index()
    aspect_counts.columns = ["Aspect", "Mentions"]

    aspect_pivot = pd.crosstab(
        aspect_df["Aspect"],
        aspect_df["Sentiment"]
    )
else:
    aspect_counts = pd.DataFrame(columns=["Aspect", "Mentions"])
    aspect_pivot = pd.DataFrame()

# ================================================================
# 10. HERO
# ================================================================

st.markdown("""
<div class="hero">
    <h1>Customer Review Intelligence</h1>
    <p>Turn large-scale customer feedback into clear sentiment, feature and business insights.</p>
</div>
""", unsafe_allow_html=True)

source_label = f"{source_type} • {len(df):,} reviews"

st.caption(source_label)

# ================================================================
# 11. KPI CARDS
# ================================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Reviews analysed</div>'
        f'<div class="metric-value">{total_reviews:,}</div></div>',
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Positive</div>'
        f'<div class="metric-value">{positive_pct:.1f}%</div></div>',
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Negative</div>'
        f'<div class="metric-value">{negative_pct:.1f}%</div></div>',
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">High priority</div>'
        f'<div class="metric-value">{high_priority:,}</div></div>',
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Model accuracy</div>'
        f'<div class="metric-value">{accuracy * 100:.1f}%</div></div>',
        unsafe_allow_html=True
    )

# ================================================================
# 12. NAVIGATION TABS
# ================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔎 Feature Insights",
    "💬 Review Analyzer",
    "📈 Model Performance",
    "📥 Data & Export"
])

# ================================================================
# TAB 1 — OVERVIEW
# ================================================================

with tab1:
    st.markdown('<div class="section-title">Customer sentiment overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])

    with c1:
        sentiment_counts = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Reviews": [positive_count, neutral_count, negative_count]
        })

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            hole=0.62,
            title="Overall sentiment"
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            legend_title="",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if not aspect_counts.empty:
            fig = px.bar(
                aspect_counts.head(10).sort_values("Mentions"),
                x="Mentions",
                y="Aspect",
                orientation="h",
                title="Most discussed product areas",
                text="Mentions"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=20, t=60, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Business signals</div>', unsafe_allow_html=True)

    top_positive_aspects = []
    top_negative_aspects = []

    if not aspect_df.empty:
        positive_aspects = aspect_df[aspect_df["Sentiment"] == "Positive"]["Aspect"].value_counts()
        negative_aspects = aspect_df[aspect_df["Sentiment"] == "Negative"]["Aspect"].value_counts()

        top_positive_aspects = positive_aspects.head(3).index.tolist()
        top_negative_aspects = negative_aspects.head(3).index.tolist()

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            '<div class="insight-card"><b>Customer sentiment</b><br>'
            f'{positive_pct:.1f}% of analysed reviews are positive, '
            f'while {negative_pct:.1f}% are negative.</div>',
            unsafe_allow_html=True
        )

    with i2:
        text = ", ".join(top_positive_aspects) if top_positive_aspects else "No clear aspect"
        st.markdown(
            f'<div class="insight-card"><b>Positive discussion areas</b><br>{text}</div>',
            unsafe_allow_html=True
        )

    with i3:
        text = ", ".join(top_negative_aspects) if top_negative_aspects else "No clear aspect"
        st.markdown(
            f'<div class="insight-card"><b>Areas needing attention</b><br>{text}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Priority distribution</div>', unsafe_allow_html=True)

    priority_counts = results_df["Priority"].value_counts().reindex(
        ["HIGH", "MEDIUM", "LOW"], fill_value=0
    ).reset_index()
    priority_counts.columns = ["Priority", "Reviews"]

    fig = px.bar(
        priority_counts,
        x="Priority",
        y="Reviews",
        text="Reviews",
        title="Review priority distribution"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# TAB 2 — FEATURE INSIGHTS
# ================================================================

with tab2:
    st.markdown('<div class="section-title">Aspect-level customer intelligence</div>', unsafe_allow_html=True)

    if not aspect_df.empty:
        left, right = st.columns([1.25, 1])

        with left:
            fig = px.bar(
                aspect_counts.head(12).sort_values("Mentions"),
                x="Mentions",
                y="Aspect",
                orientation="h",
                title="Feature mention frequency",
                text="Mentions"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=500,
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            if not aspect_pivot.empty:
                heatmap = aspect_pivot.reindex(
                    columns=["Negative", "Neutral", "Positive"],
                    fill_value=0
                )

                fig = px.imshow(
                    heatmap,
                    text_auto=True,
                    aspect="auto",
                    title="Sentiment by feature"
                )
                fig.update_layout(
                    height=500,
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Feature sentiment table</div>', unsafe_allow_html=True)

        display_table = aspect_pivot.reindex(
            columns=["Negative", "Neutral", "Positive"],
            fill_value=0
        ).reset_index()

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        # Word cloud
        st.markdown('<div class="section-title">Customer language</div>', unsafe_allow_html=True)

        all_text = " ".join(results_df["review"].astype(str))

        cloud = WordCloud(
            width=1400,
            height=500,
            background_color="white",
            stopwords=STOP_WORDS,
            min_font_size=10
        ).generate(all_text)

        buffer = io.BytesIO()
        cloud.to_image().save(buffer, format="PNG")

        st.image(
            buffer.getvalue(),
            caption="Frequently occurring customer terms",
            use_container_width=True
        )

# ================================================================
# TAB 3 — REVIEW ANALYZER
# ================================================================

with tab3:
    st.markdown('<div class="section-title">Analyse an individual customer review</div>', unsafe_allow_html=True)

    default_review = (
        "The camera quality is excellent and the screen looks beautiful, "
        "but the battery drains very quickly and customer support was slow."
    )

    review_input = st.text_area(
        "Customer review",
        value=default_review,
        height=140,
        placeholder="Enter a customer review..."
    )

    if st.button("Analyse Review", type="primary", use_container_width=True):
        if review_input.strip():
            result = analyze_review(review_input, model)

            st.markdown('<div class="section-title">Analysis result</div>', unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)

            with a1:
                st.metric("Sentiment", result["Sentiment"])

            with a2:
                st.metric("Confidence", f'{result["Confidence"]:.2f}%')

            with a3:
                st.metric("Priority", result["Priority"])

            st.markdown(
                '<div class="review-card">'
                f'<b>Review</b><br><br>{result["Review"]}'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            b1, b2 = st.columns(2)

            with b1:
                st.markdown("### Detected aspects")
                for aspect in result["Detected Aspects"].split(", "):
                    st.markdown(f'<span class="tag">{aspect}</span>', unsafe_allow_html=True)

            with b2:
                st.markdown("### Keywords")
                for keyword in result["Keywords"].split(", "):
                    st.markdown(f'<span class="tag">{keyword}</span>', unsafe_allow_html=True)

            st.markdown("### Aspect sentiment")

            aspect_display = pd.DataFrame(result["Aspect Results"])
            st.dataframe(
                aspect_display,
                use_container_width=True,
                hide_index=True
            )

            st.info(result["Recommendation"])

# ================================================================
# TAB 4 — MODEL PERFORMANCE
# ================================================================

with tab4:
    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    with m2:
        st.metric("Training reviews", f"{train_count:,}")

    with m3:
        st.metric("Testing reviews", f"{test_count:,}")

    st.markdown("### Classification metrics")

    metrics_rows = []

    for label in ["Negative", "Neutral", "Positive"]:
        if label in report:
            metrics_rows.append({
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1 Score": report[label]["f1-score"],
                "Support": int(report[label]["support"])
            })

    metrics_df = pd.DataFrame(metrics_rows)

    st.dataframe(
        metrics_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Confusion matrix")

    cm_df = pd.DataFrame(
        matrix,
        index=["Actual Negative", "Actual Neutral", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Classification confusion matrix"
    )
    fig.update_layout(
        height=450,
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="small-note">'
        'The evaluation metrics are calculated on a held-out test split that was not used during model training.'
        '</div>',
        unsafe_allow_html=True
    )

# ================================================================
# TAB 5 — DATA & EXPORT
# ================================================================

with tab5:
    st.markdown('<div class="section-title">Processed review dataset</div>', unsafe_allow_html=True)

    export_columns = [
        "review",
        "Predicted Sentiment",
        "Confidence",
        "Detected Aspects",
        "Priority",
        "Keywords"
    ]

    export_df = results_df[export_columns].copy()

    st.dataframe(
        export_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download analysed reviews CSV",
        data=csv_data,
        file_name="customer_review_intelligence_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### Dataset information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Rows", f"{len(df):,}")

    with info2:
        st.metric("Training rows", f"{len(training_df):,}")

    with info3:
        st.metric("Unique sentiment classes", df["sentiment"].nunique())

    st.markdown("### Dataset preview")
    st.dataframe(
        df.drop(columns=["clean_review"], errors="ignore").head(25),
        use_container_width=True,
        hide_index=True
    )

# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "Customer Review Intelligence System • NLP • TF-IDF • Logistic Regression • "
    "Aspect Analysis • Business Insights"
)
