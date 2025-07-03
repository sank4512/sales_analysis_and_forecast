import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st
from io import BytesIO
from datetime import datetime, date
from fpdf import FPDF
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="📊 Sales Forecast Dashboard", layout="wide")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding='ISO-8859-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df = df.dropna(subset=['Sales', 'Profit', 'Quantity'])
    return df

# --- Load Data ---
df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Filters")
region = st.sidebar.selectbox("Select Region", ['All'] + sorted(df['Region'].unique()))
product = st.sidebar.selectbox("Select Product", ['All'] + sorted(df['Product Name'].unique()))
start_date, end_date = st.sidebar.date_input("Select Date Range", [df['Order Date'].min(), df['Order Date'].max()])

filtered_df = df.copy()
if region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == region]
if product != 'All':
    filtered_df = filtered_df[filtered_df['Product Name'] == product]
filtered_df = filtered_df[(filtered_df['Order Date'] >= pd.to_datetime(start_date)) & (filtered_df['Order Date'] <= pd.to_datetime(end_date))]

# --- Title ---
st.title("📊 Sales Analysis and Forecasting Dashboard")

# --- KPIs ---
st.subheader("📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"INR {filtered_df['Sales'].sum():,.2f}")
col2.metric("Total Profit", f"INR {filtered_df['Profit'].sum():,.2f}")
col3.metric("Total Quantity", int(filtered_df['Quantity'].sum()))

# --- Monthly Trends ---
monthly = filtered_df.copy()
monthly.set_index('Order Date', inplace=True)
monthly = monthly.resample('M').sum(numeric_only=True)

prev_sales = monthly['Sales'].iloc[-2] if len(monthly) > 1 else 0
latest_sales = monthly['Sales'].iloc[-1] if len(monthly) > 0 else 0
change = ((latest_sales - prev_sales) / prev_sales) * 100 if prev_sales > 0 else 0
st.metric("Monthly Sales Change", f"{change:.2f}%", delta=f"{change:.2f}%")

# --- Monthly Sales ---
st.subheader("📆 Monthly Sales Trend")
fig1 = px.line(monthly, x=monthly.index, y='Sales', title='Monthly Sales')
st.plotly_chart(fig1)

# --- Monthly Profit ---
st.subheader("💰 Monthly Profit Trend")
fig2 = px.line(monthly, x=monthly.index, y='Profit', title='Monthly Profit')
st.plotly_chart(fig2)

# --- Quantity by Category ---
st.subheader("📦 Quantity by Category")
cat_qty = filtered_df.groupby('Category')['Quantity'].sum().reset_index()
fig3 = px.bar(cat_qty, x='Category', y='Quantity', color='Category')
st.plotly_chart(fig3)

# --- Top 5 Products ---
st.subheader("🏆 Top 5 Products by Sales")
top_products = filtered_df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(5)
st.dataframe(top_products)

# --- Profit Heatmap ---
st.subheader("🗺️ Region vs Category Profit Heatmap")
pivot = filtered_df.pivot_table(values='Profit', index='Region', columns='Category', aggfunc='sum')
fig4 = px.imshow(pivot, text_auto=True, title="Region vs Category Profit")
st.plotly_chart(fig4)

# --- Forecasting ---
st.subheader("📅 Forecast Sales (Next 90 Days)")
sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
sales.columns = ['ds', 'y']
model = Prophet()
model.fit(sales)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
fig5 = plot_plotly(model, forecast)
st.plotly_chart(fig5)

# --- Forecast Evaluation ---
if len(sales) >= 30:
    actual = sales['y'].iloc[-30:]
    predicted = forecast['yhat'].iloc[-30:]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    st.metric("📊 Forecast RMSE", f"{rmse:.2f}")

# --- Export Forecast CSV ---
st.download_button("📥 Download Forecast CSV", data=forecast.to_csv(index=False), file_name="sales_forecast.csv")

# --- Export Forecast PDF ---
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Sales Forecast Report", ln=True, align='C')
    pdf.ln(10)
    for i in range(min(len(df), 50)):
        row = df.iloc[i]
        line = f"{row['ds'].strftime('%Y-%m-%d')}: INR {round(row['yhat'], 2)}"
        pdf.cell(200, 6, txt=line, ln=True)
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

if st.button("📤 Export Forecast PDF"):
    pdf_bytes = create_pdf(forecast[['ds', 'yhat']])
    st.download_button("Download PDF", data=pdf_bytes, file_name="sales_forecast.pdf")
