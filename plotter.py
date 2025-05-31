import pandas as pd
import plotly.express as px
import streamlit as st

# Load your Excel file
df = pd.read_excel("./consolidated_results/statistics_vgg16.xlsx")

st.title("Interactive Data Filter and Scatter Plot")

# Sidebar filters for multiple columns
#columns_to_filter = ['Layer Name', 'Lambda Alignment', 'Accuracy Before']  # replace with your column names
columns_to_filter = df.columns.tolist()
print("Available columns for filtering:", columns_to_filter)
columns_to_filter = st.sidebar.multiselect("Select columns to filter", df.columns.tolist(), default=df.columns.tolist())
if not columns_to_filter:
    st.error("No valid columns found for filtering.")
    st.stop()
for col in columns_to_filter:
    unique_vals = df[col].dropna().unique()
    selected = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=unique_vals)
    df = df[df[col].isin(selected)]
    # Select columns for X, Y, and Z axes
    x_axis = st.selectbox("Select X-axis", df.columns, key="x_axis")
    y_axis = st.selectbox("Select Y-axis", df.columns, key="y_axis")
    #z_axis = st.selectbox("Select Z-axis", df.columns, key="z_axis")

    # 3D Scatter plot
    #fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=columns_to_filter[0] if columns_to_filter else None)
    fig = px.scatter(df, x=x_axis, y=y_axis, color=columns_to_filter[0] if columns_to_filter else None)
    st.plotly_chart(fig)

