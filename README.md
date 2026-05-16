# 🛒 E-Commerce Analytics Dashboard

A fully interactive **E-Commerce Customer Behavior Analytics** web application built with Python and Streamlit. Analyzes 599,721 customer events from October & November 2019 to uncover insights on revenue, customer segments, behavior patterns, and churn prediction.

🌐 **Live App:** [Click here to open](https://ecommerce-analytics-vtotvskozgzi9ew7yqhvyt.streamlit.app/)

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 **Overview** | Key metrics, revenue by category, customer funnel |
| 👥 **Customer Segments** | ML-based customer clustering and segmentation |
| 📋 **Behavior Analysis** | Deep dive into browsing and purchase behavior |
| 🔮 **Churn Predictor** | Machine learning model to predict customer churn |

---

## 🔢 Dataset Stats

| Metric | Value |
|---|---|
| Total Events | 599,721 |
| Total Revenue | $3,384,060 |
| Total Purchases | 11,214 |
| Conversion Rate | 1.87% |
| Avg Order Value | $302 |
| Unique Users | 111,878 |
| Period | Oct – Nov 2019 |

---

## 🛠️ Tech Stack

- **Python 3.14** — Core programming language
- **Streamlit** — Web application framework
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib & Seaborn** — Data visualizations
- **Scikit-learn** — Machine learning (clustering + churn prediction)
- **Joblib** — Model serialization
- **GitHub** — Version control and hosting
- **Streamlit Cloud** — Free deployment platform

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Gul-Andam/ecommerce-analytics.git
cd ecommerce-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run ecommerce-app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
ecommerce-analytics/
│
├── ecommerce-app.py        # Main Streamlit application
├── ecommerce_cleaned.csv   # Cleaned dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 📸 Screenshots

### Overview Dashboard
<img width="1354" height="729" alt="Website new" src="https://github.com/user-attachments/assets/a3458f0e-c810-4a08-adf5-dc2461c17919" />


---

## 🌐 Deployment

This app is deployed for fr<img width="1354" height="729" alt="Website new" src="https://github.com/user-attachments/assets/e587926f-7b5a-44f0-89c5-0339cf0b2389" />
ee on **Streamlit Community Cloud**.

To deploy your own version:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and `ecommerce-app.py`
5. Click **Deploy**

---

## 👨‍💻 Author

**Gul Andam**
- LinkedIn: https://www.linkedin.com/in/gull-andam-48a2a1331/
- GitHub: https://github.com/Gul-Andam
---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project helpful, please give it a star!
