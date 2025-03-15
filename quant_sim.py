import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Streamlit app title
st.title("Rigorous Trading History Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your trading history CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Convert columns to appropriate data types
    try:
        df['Time(UTC)'] = pd.to_datetime(df['Time(UTC)'])
        df['Price'] = pd.to_numeric(df['Price'])
        df['Quantity'] = pd.to_numeric(df['Quantity'])
        df['Amount'] = pd.to_numeric(df['Amount'])
        df['Fee'] = pd.to_numeric(df['Fee'].str.replace(' USDT', ''))
        df['Realized Profit'] = pd.to_numeric(df['Realized Profit'])
    except KeyError as e:
        st.error(f"Missing required column: {e}. Please ensure your file has the correct columns.")
        st.stop()

    # Add a cumulative profit column
    df['Cumulative Profit'] = df['Realized Profit'].cumsum()

    # Calculate key metrics
    total_trades = len(df)
    winning_trades = len(df[df['Realized Profit'] > 0])
    losing_trades = len(df[df['Realized Profit'] < 0])
    win_rate = winning_trades / total_trades * 100
    average_profit_per_trade = df['Realized Profit'].mean()
    average_loss_per_trade = df[df['Realized Profit'] < 0]['Realized Profit'].mean()
    profit_factor = abs(df[df['Realized Profit'] > 0]['Realized Profit'].sum() / df[df['Realized Profit'] < 0]['Realized Profit'].sum())
    total_fees = df['Fee'].sum()
    net_profit = df['Realized Profit'].sum() - total_fees
    max_drawdown = (df['Cumulative Profit'].cummax() - df['Cumulative Profit']).max()

    # Risk-adjusted metrics
    returns = df['Realized Profit'] / df['Amount']  # Normalized returns
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe ratio
    sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252)  # Annualized Sortino ratio
    calmar_ratio = returns.mean() / max_drawdown * np.sqrt(252)  # Annualized Calmar ratio

    # Higher-order moments of returns
    skewness = skew(returns)
    kurt = kurtosis(returns)

    # Outlier detection using Isolation Forest
    outlier_detector = IsolationForest(contamination=0.05, random_state=42)
    df['Outlier'] = outlier_detector.fit_predict(df[['Realized Profit']])
    outlier_trades = df[df['Outlier'] == -1]

    # Clustering trades based on profit and quantity
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Realized Profit', 'Quantity']])

    # Display key metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", total_trades)
        st.metric("Winning Trades", f"{winning_trades} ({win_rate:.2f}%)")
        st.metric("Average Profit per Trade", f"{average_profit_per_trade:.8f} USDT")
    with col2:
        st.metric("Losing Trades", losing_trades)
        st.metric("Average Loss per Trade", f"{average_loss_per_trade:.8f} USDT")
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    with col3:
        st.metric("Total Fees Paid", f"{total_fees:.8f} USDT")
        st.metric("Net Profit", f"{net_profit:.8f} USDT")
        st.metric("Maximum Drawdown", f"{max_drawdown:.8f} USDT")

    st.subheader("Risk-Adjusted Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.2f}")
    with col2:
        st.metric("Sortino Ratio (Annualized)", f"{sortino_ratio:.2f}")
    with col3:
        st.metric("Calmar Ratio (Annualized)", f"{calmar_ratio:.2f}")

    st.subheader("Higher-Order Moments of Returns")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Skewness of Returns", f"{skewness:.2f}")
    with col2:
        st.metric("Kurtosis of Returns", f"{kurt:.2f}")

    # Plot cumulative profit over time
    st.subheader("Cumulative Profit Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Time(UTC)'], df['Cumulative Profit'], label='Cumulative Profit')
    ax.set_title('Cumulative Profit Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Profit (USDT)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Plot distribution of normalized returns
    st.subheader("Distribution of Normalized Returns")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=50, color='blue', alpha=0.7, density=True, label='Returns')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', label='Normal Distribution')
    ax.set_title('Distribution of Normalized Returns')
    ax.set_xlabel('Normalized Returns')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Plot clustering results
    st.subheader("Clustering of Trades by Profit and Quantity")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data['Realized Profit'], cluster_data['Quantity'], label=f'Cluster {cluster}')
    ax.set_title('Clustering of Trades by Profit and Quantity')
    ax.set_xlabel('Realized Profit')
    ax.set_ylabel('Quantity')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Personalized Qualitative Analysis
    st.subheader("Personalized Qualitative Analysis")
    st.write("1. **Strategy Efficiency:**")
    if sharpe_ratio < 1:
        st.write("   - The strategy exhibits suboptimal risk-adjusted returns, as indicated by the low Sharpe ratio.")
        st.write("   - Recommendation: Focus on improving the consistency of returns and reducing volatility.")
    else:
        st.write("   - The strategy demonstrates strong risk-adjusted returns, as indicated by the high Sharpe ratio.")
        st.write("   - Recommendation: Consider scaling the strategy while maintaining risk controls.")

    st.write("2. **Risk Management:**")
    if max_drawdown > 0.1 * net_profit:
        st.write("   - The maximum drawdown is significant relative to net profit, indicating inadequate risk controls.")
        st.write("   - Recommendation: Implement stricter stop-loss mechanisms and dynamic position sizing.")
    else:
        st.write("   - The maximum drawdown is within acceptable limits, suggesting effective risk management.")
        st.write("   - Recommendation: Continue monitoring drawdowns and adjust risk controls as needed.")

    st.write("3. **Outlier Trades:**")
    if len(outlier_trades) > 0:
        st.write(f"   - {len(outlier_trades)} outlier trades were detected, which may indicate unusual market conditions or execution issues.")
        st.write("   - Recommendation: Investigate outlier trades to identify potential improvements in execution or strategy.")
    else:
        st.write("   - No significant outlier trades were detected, indicating consistent execution.")

    st.write("4. **Trade Clustering:**")
    st.write("   - The trades were clustered into three groups based on profit and quantity:")
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"     - Cluster {cluster}: {len(cluster_data)} trades with average profit {cluster_data['Realized Profit'].mean():.2f} and average quantity {cluster_data['Quantity'].mean():.2f}.")
    st.write("   - Recommendation: Focus on optimizing the most profitable clusters and reducing exposure to less profitable ones.")

    st.write("5. **Market Regime Adaptation:**")
    if skewness < 0 or kurt > 3:
        st.write("   - The negative skewness and high kurtosis of returns suggest exposure to tail risk and fat-tailed distributions.")
        st.write("   - Recommendation: Incorporate regime-switching models to adapt to changing market conditions.")
    else:
        st.write("   - The returns exhibit a relatively normal distribution, indicating stable performance across market conditions.")

    st.write("=== Final Assessment ===")
    st.write("The trading strategy demonstrates a mix of strengths and weaknesses. While the strategy shows potential, it requires refinement in risk management, execution, and adaptability to market conditions. Advanced techniques, such as machine learning and portfolio optimization, should be employed to enhance performance and scalability.")
