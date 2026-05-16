
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "E-Commerce Analytics",
    page_icon  = "🛒",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Dark style ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #c8b8ff; }
    .metric-label { font-size: 13px; color: #888; margin-top: 5px; }
    .insight-box {
        background: #1a1a2e;
        border-left: 4px solid #7C3AED;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ["#7C3AED","#06B6D4","#F59E0B","#10B981","#EF4444",
          "#EC4899","#3B82F6","#84CC16"]

plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#0f1117",
    "axes.edgecolor":"#333","axes.labelcolor":"#fff",
    "xtick.color":"#aaa","ytick.color":"#aaa","text.color":"#fff",
    "grid.color":"#222","grid.linestyle":"--","grid.alpha":0.5,
})

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "👥 Customer Segments",
    "📈 Behavior Analysis",
    "🤖 Churn Predictor"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Oct–Nov 2019")
st.sidebar.markdown("**Records:** 599,721 events")
st.sidebar.markdown("**Users:** 111,878")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce_cleaned.csv")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df

@st.cache_data
def load_daily():
    return pd.read_csv("powerbi_daily_summary.csv")

@st.cache_resource
def load_model():
    model    = joblib.load("churn_model.pkl")
    scaler   = joblib.load("churn_scaler.pkl")
    le_cat   = joblib.load("churn_le_cat.pkl")
    le_brand = joblib.load("churn_le_brand.pkl")
    return model, scaler, le_cat, le_brand

try:
    df    = load_data()
    daily = load_daily()
    model, scaler, le_cat, le_brand = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load data: {e}")
    st.info("Upload your CSV files using the uploader below")
    data_loaded = False
    uploaded = st.file_uploader("Upload ecommerce_cleaned.csv",
                                 type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        data_loaded = True

if not data_loaded:
    st.stop()

# ═════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.markdown("**E-Commerce Customer Behavior — Oct & Nov 2019**")
    st.markdown("---")

    # KPI cards
    total_rev  = df[df["is_purchase"]==1]["price"].sum()
    total_pur  = df["is_purchase"].sum()
    conv_rate  = df["is_purchase"].mean() * 100
    aov        = total_rev / total_pur if total_pur > 0 else 0
    total_users= df["user_id"].nunique()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, value in zip(
        [c1,c2,c3,c4,c5],
        ["Total Revenue","Total Purchases","Conversion Rate",
         "Avg Order Value","Unique Users"],
        [f"${total_rev:,.0f}", f"{total_pur:,}",
         f"{conv_rate:.2f}%", f"${aov:.0f}", f"{total_users:,}"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Revenue by Category")
        cat_rev = (df[df["is_purchase"]==1]
                   .groupby("main_category")["price"]
                   .sum().sort_values(ascending=True).tail(8))
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(cat_rev.index, cat_rev.values,
                color=COLORS[:len(cat_rev)], edgecolor="none")
        ax.set_xlabel("Revenue ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🔽 Customer Funnel")
        funnel = df["event_type"].value_counts()
        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(funnel.index, funnel.values,
                      color=COLORS[:3], edgecolor="none", width=0.5)
        for bar, val in zip(bars, funnel.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+2000,
                    f"{val:,}", ha="center", fontsize=10)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("📈 Daily Revenue Trend")
    daily["date"] = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(14,4))
    ax.fill_between(daily["date"], daily["total_revenue"],
                    alpha=0.3, color="#7C3AED")
    ax.plot(daily["date"], daily["total_revenue"],
            color="#7C3AED", linewidth=2.5)
    ax.set_ylabel("Revenue ($)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    insights = [
        ("🔴 Conversion Crisis",
         "96.7% of users browse without purchasing. "
         "Retargeting campaigns could unlock $310K+ additional revenue."),
        ("🟣 Electronics Dominance",
         "Electronics drives 78% of revenue ($2.59M). "
         "Focus inventory and ads here."),
        ("🟡 Premium Brand Power",
         "Apple earns 2.4x Samsung revenue with fewer sales. "
         "Premium pricing strategy outperforms volume."),
    ]
    for title, text in insights:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{title}</strong><br>{text}
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ═════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.markdown("---")

    users = (df.groupby("user_id")
             .agg(purchases=("is_purchase","sum"),
                  events=("event_type","count"))
             .reset_index())
    users["segment"] = pd.cut(
        users["purchases"],
        bins=[-1,0,1,5,9999],
        labels=["Browsers","One-time","Regulars","Loyal"]
    )
    seg_counts = users["segment"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Segments")
        fig, ax = plt.subplots(figsize=(6,4))
        wedges,texts,autotexts = ax.pie(
            seg_counts.values,
            labels=seg_counts.index,
            autopct="%1.1f%%",
            colors=COLORS[:4],
            startangle=140,
            wedgeprops={"edgecolor":"#0f1117","linewidth":2}
        )
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Users per Segment")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(seg_counts.index, seg_counts.values,
                color=COLORS[:4], edgecolor="none")
        for i,val in enumerate(seg_counts.values):
            ax.text(val+200, i, f"{val:,}", va="center", fontsize=10)
        ax.set_xlabel("Number of Users")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("🏆 Top 10 Brands by Revenue")
    top_brands = (df[df["is_purchase"]==1]
                  .groupby("brand")["price"]
                  .sum()
                  .sort_values(ascending=False)
                  .head(10))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(top_brands.index, top_brands.values,
           color=COLORS, edgecolor="none")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════
# PAGE 3 — BEHAVIOR ANALYSIS
# ═════════════════════════════════════════════════════════
elif page == "📈 Behavior Analysis":
    st.title("📈 Behavior Analysis")

    # Category filter
    cats = ["All"] + sorted(df["main_category"].unique().tolist())
    selected = st.selectbox("Filter by Category", cats)
    filtered = df if selected=="All" else df[df["main_category"]==selected]
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕐 Purchases by Hour")
        hourly = filtered.groupby("hour")["is_purchase"].sum()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(hourly.index, hourly.values,
               color="#7C3AED", edgecolor="none")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Purchases")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("💳 Conversion by Price Range")
        pr = (filtered.groupby("price_range")
              .agg(events=("event_type","count"),
                   purchases=("is_purchase","sum"))
              .reset_index())
        pr["conversion"] = pr["purchases"]/pr["events"]*100
        pr = pr.dropna()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(pr["price_range"].astype(str),
               pr["conversion"],
               color=COLORS[:len(pr)], edgecolor="none")
        ax.set_ylabel("Conversion Rate (%)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("🗓️ Activity Heatmap — Hour vs Day of Week")
    pivot = (filtered.groupby(["day_of_week","hour"])["event_type"]
             .count()
             .unstack(fill_value=0))
    day_order = ["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig, ax = plt.subplots(figsize=(14,4))
    sns.heatmap(pivot, cmap="RdPu", ax=ax,
                linewidths=0.3, linecolor="#0f1117")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════
# PAGE 4 — CHURN PREDICTOR
# ═════════════════════════════════════════════════════════
elif page == "🤖 Churn Predictor":
    st.title("🤖 Customer Churn Predictor")
    st.markdown("Enter a customer\'s first session behaviour to predict "
                "if they will return for another purchase.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        sess_events  = st.slider("Total clicks in session",  1, 50, 10)
        sess_views   = st.slider("Products viewed",           1, 40,  8)
        sess_carts   = st.slider("Items added to cart",       0, 10,  1)
        sess_unique  = st.slider("Unique products browsed",   1, 20,  5)
    with col2:
        sess_avg_price  = st.number_input("Avg price viewed ($)",   0.0, 5000.0, 250.0)
        sess_max_price  = st.number_input("Max price viewed ($)",   0.0, 5000.0, 500.0)
        sess_price_span = st.number_input("Price range span ($)",   0.0, 5000.0, 200.0)
        sess_hour       = st.slider("Hour of visit (UTC)",          0, 23, 5)
    with col3:
        categories = le_cat.classes_.tolist()
        brands     = le_brand.classes_.tolist()
        category   = st.selectbox("Favourite Category", categories)
        brand      = st.selectbox("Favourite Brand",    brands)

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        cat_enc   = le_cat.transform([category])[0]
        brand_enc = le_brand.transform([brand])[0]
        features  = np.array([[
            sess_events, sess_views, sess_carts, sess_unique,
            sess_avg_price, sess_max_price, sess_price_span,
            sess_hour, cat_enc, brand_enc
        ]])
        prob  = model.predict_proba(features)[0][1]
        label = "🔴 HIGH churn risk" if prob>0.6 else (
                "🟡 MEDIUM churn risk" if prob>0.4 else
                "🟢 LOW churn risk — likely to return!")

        st.markdown("---")
        st.markdown(f"### Result: {label}")
        st.progress(float(prob))
        st.markdown(f"**Churn probability: {prob*100:.1f}%**")

        if prob > 0.6:
            st.error("⚠️ Action: Send retention email within 48 hours. "
                     "Offer 10% discount on viewed category.")
        elif prob > 0.4:
            st.warning("⚡ Action: Add to retargeting campaign. "
                       "Show personalised recommendations.")
        else:
            st.success("✅ This customer is likely to return. "
                       "Consider upsell opportunities.")
