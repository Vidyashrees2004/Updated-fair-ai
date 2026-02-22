"""Fair AI Dashboard with error handling"""

# Try importing with error handling
try:
    import streamlit as st
except ImportError:
    print("Please install streamlit: pip install streamlit")
    raise

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    st.error("⚠️ Plotly not installed. Please install it with: pip install plotly")
    st.stop()

try:
    import shap
except ImportError:
    st.error("⚠️ SHAP not installed. Please install it with: pip install shap")
    st.stop()

try:
    import joblib
except ImportError:
    st.error("⚠️ joblib not installed. Please install it with: pip install joblib")
    st.stop()

try:
    from codecarbon import EmissionsTracker
except ImportError:
    st.error("⚠️ CodeCarbon not installed. Please install it with: pip install codecarbon")
    st.stop()

try:
    import psutil
except ImportError:
    st.warning("⚠️ psutil not installed. Energy tracking will be disabled.")
    psutil = None

import pandas as pd
import numpy as np
import time
import os

# Rest of your dashboard code continues here...
