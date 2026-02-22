# Fair AI Dashboard with Explainability & Energy Tracking

## Features
- 🤖 **Fair Model (Demographic Parity)** - Loaded by default
- 🧠 **SHAP Explainability** - Feature importance for each prediction
- ⚡ **Energy Tracking** - Real-time CO2 emissions, CPU, memory usage
- 📊 **Interactive Dashboard** - Beautiful UI with Plotly charts
- ⚖️ **Fairness Metrics** - Demographic parity difference

## Deployment on Streamlit Cloud
1. Upload this package to GitHub
2. Connect to Streamlit Cloud
3. Set main file path to `fair_ai_dashboard.py`
4. Deploy!

## Local Installation
```bash
pip install -r requirements_dashboard.txt
streamlit run fair_ai_dashboard.py
```

## Model Information
- **Fair Model**: LightGBM with Demographic Parity constraint
- **Features**: Age, Education, Hours/Week, Gender, Race
- **Protected Attribute**: Gender
