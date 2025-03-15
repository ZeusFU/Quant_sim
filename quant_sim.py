import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm, t
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf
from arch import arch_model
import warnings
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Streamlit app title
st.title("Advanced Quantitative Trading Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your trading history CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
        
        # Display the first few rows for verification
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
        
        # Data validation and conversion
        required_columns = ['Time(UTC)', 'Price', 'Quantity', 'Amount', 'Fee', 'Realized Profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your file has the correct format.")
            st.stop()
        
        # Convert columns to appropriate data types with robust error handling
        try:
            df['Time(UTC)'] = pd.to_datetime(df['Time(UTC)'])
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            
            # Handle different fee formats
            if df['Fee'].dtype == object:
                df['Fee'] = df['Fee'].str.replace(r'[^\d.-]', '', regex=True)
            df['Fee'] = pd.to_numeric(df['Fee'], errors='coerce')
            
            df['Realized Profit'] = pd.to_numeric(df['Realized Profit'], errors='coerce')
            
            # Check for and handle NaN values
            if df.isna().any().any():
                st.warning(f"Found {df.isna().sum().sum()} missing values in the data. These will be handled appropriately in the analysis.")
                # Fill NaN values where appropriate
                df['Fee'] = df['Fee'].fillna(0)
                df = df.dropna(subset=['Price', 'Quantity', 'Amount', 'Realized Profit'])
                
            # Verify data integrity
            if len(df) == 0:
                st.error("No valid data remains after cleaning. Please check your input file.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error converting data types: {e}. Please check your file format.")
            st.stop()

        # Add defensive calculation for returns
        df['Amount_Safe'] = df['Amount'].apply(lambda x: x if x != 0 else 1e-10)
        df['Return'] = df['Realized Profit'] / df['Amount_Safe']
        
        # Add a cumulative profit column
        df['Cumulative Profit'] = df['Realized Profit'].cumsum()
        
        # Sort by time for proper time series analysis
        df = df.sort_values('Time(UTC)')
        
        # Create daily returns for time series analysis
        df['Date'] = df['Time(UTC)'].dt.date
        daily_returns = df.groupby('Date')['Return'].sum().reset_index()
        daily_returns['Date'] = pd.to_datetime(daily_returns['Date'])
        
        # Detect data issues
        st.subheader("Data Quality Assessment")
        
        # Check for potential look-ahead bias
        time_diff = df['Time(UTC)'].diff().dropna()
        if (time_diff < pd.Timedelta(0)).any():
            st.warning("⚠️ Potential look-ahead bias detected: Trades are not in strict chronological order.")
        
        # Check for unrealistic returns
        extreme_returns = df[abs(df['Return']) > 1]
        if not extreme_returns.empty:
            st.warning(f"⚠️ Found {len(extreme_returns)} trades with returns exceeding 100%. This may indicate data quality issues.")
        
        # Calculate baseline metrics
        total_trades = len(df)
        winning_trades = len(df[df['Realized Profit'] > 0])
        losing_trades = len(df[df['Realized Profit'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Defensive calculation for average metrics
        average_profit_per_trade = df['Realized Profit'].mean() if len(df) > 0 else 0
        avg_winning_trade = df[df['Realized Profit'] > 0]['Realized Profit'].mean() if winning_trades > 0 else 0
        avg_losing_trade = df[df['Realized Profit'] < 0]['Realized Profit'].mean() if losing_trades > 0 else 0
        
        # Defensive calculation for profit factor
        total_profit = df[df['Realized Profit'] > 0]['Realized Profit'].sum()
        total_loss = df[df['Realized Profit'] < 0]['Realized Profit'].sum()
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        total_fees = df['Fee'].sum()
        net_profit = df['Realized Profit'].sum() - total_fees
        
        # Enhanced drawdown analysis
        df['Peak'] = df['Cumulative Profit'].cummax()
        df['Drawdown'] = df['Peak'] - df['Cumulative Profit']
        max_drawdown = df['Drawdown'].max()
        
        # Drawdown duration analysis
        df['In_Drawdown'] = df['Drawdown'] > 0
        df['Drawdown_Start'] = (df['In_Drawdown'] & ~df['In_Drawdown'].shift(1).fillna(False)).astype(int)
        df['Drawdown_End'] = (df['In_Drawdown'].shift(-1).fillna(False) & ~df['In_Drawdown']).astype(int)
        
        drawdown_periods = []
        current_start = None
        
        for i, row in df.iterrows():
            if row['Drawdown_Start'] == 1:
                current_start = row['Time(UTC)']
            if row['Drawdown_End'] == 1 and current_start is not None:
                drawdown_periods.append((current_start, row['Time(UTC)'], (row['Time(UTC)'] - current_start).total_seconds() / 86400))
                current_start = None
        
        # If still in drawdown at the end
        if current_start is not None:
            drawdown_periods.append((current_start, df['Time(UTC)'].iloc[-1], (df['Time(UTC)'].iloc[-1] - current_start).total_seconds() / 86400))
        
        avg_drawdown_duration = np.mean([d[2] for d in drawdown_periods]) if drawdown_periods else 0
        max_drawdown_duration = np.max([d[2] for d in drawdown_periods]) if drawdown_periods else 0
        
        # Risk-adjusted metrics with defensive calculations
        returns = df['Return']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
        
        # Calmar ratio calculation
        daily_returns_value = daily_returns['Return'].values if len(daily_returns) > 0 else np.array([0])
        annual_return = np.mean(daily_returns_value) * 252
        calmar_ratio = annual_return / (max_drawdown / net_profit) if max_drawdown > 0 and net_profit > 0 else 0
        
        # Higher-order moments of returns
        returns_skew = skew(returns)
        returns_kurt = kurtosis(returns)
        
        # Calculate Expected Shortfall (Conditional VaR)
        def calculate_ES(returns, alpha=0.05):
            sorted_returns = np.sort(returns)
            var_idx = int(alpha * len(returns))
            return -np.mean(sorted_returns[:var_idx]) if var_idx > 0 else 0
        
        expected_shortfall = calculate_ES(returns)
        
        # Calculate Omega Ratio
        def omega_ratio(returns, threshold=0):
            positive_sum = np.sum(returns[returns > threshold] - threshold)
            negative_sum = np.sum(threshold - returns[returns < threshold])
            return positive_sum / negative_sum if negative_sum > 0 else float('inf')
        
        omega = omega_ratio(returns)
        
        # Time series autocorrelation analysis
        if len(daily_returns) >= 10:
            try:
                autocorr = acf(daily_returns['Return'].fillna(0), nlags=10)
                has_autocorr_analysis = True
            except Exception as e:
                st.warning(f"Could not perform autocorrelation analysis: {e}")
                has_autocorr_analysis = False
        else:
            has_autocorr_analysis = False
            
        # Regime detection using Markov switching model
        if len(daily_returns) >= 30:  # Need sufficient data for regime detection
            try:
                # Prepare data for Markov model
                daily_returns['Return_Scaled'] = daily_returns['Return'] * 100  # Scale for numerical stability
                
                # Fit Markov switching model with 2 regimes
                mod = MarkovRegression(daily_returns['Return_Scaled'].values, k_regimes=2, trend='c', switching_variance=True)
                res = mod.fit(disp=False)
                
                # Get smoothed probabilities
                smoothed_probs = res.smoothed_marginal_probabilities
                regime_1_prob = smoothed_probs[0]
                
                # Assign regimes to dates
                daily_returns['Regime'] = np.where(regime_1_prob > 0.5, 0, 1)
                
                # Map regimes to the original trades
                regime_map = dict(zip(daily_returns['Date'], daily_returns['Regime']))
                df['Regime'] = df['Time(UTC)'].dt.date.map(lambda x: regime_map.get(pd.Timestamp(x), np.nan))
                
                # Calculate metrics by regime
                regime_stats = df.groupby('Regime').agg({
                    'Realized Profit': ['count', 'mean', 'sum'],
                    'Return': ['mean', 'std']
                })
                
                # Calculate Sharpe ratio by regime
                regime_stats['Sharpe'] = regime_stats[('Return', 'mean')] / regime_stats[('Return', 'std')] * np.sqrt(252)
                
                has_regime_analysis = True
            except Exception as e:
                st.warning(f"Could not perform regime analysis: {e}")
                has_regime_analysis = False
        else:
            has_regime_analysis = False
            
        # GARCH volatility modeling
        if len(daily_returns) >= 30:
            try:
                # Fit GARCH(1,1) model
                garch_model = arch_model(daily_returns['Return'].fillna(0).values, vol='GARCH', p=1, q=1)
                garch_result = garch_model.fit(disp='off')
                
                # Extract conditional volatilities
                conditional_vol = garch_result.conditional_volatility
                
                # Add to daily returns
                daily_returns['Conditional_Vol'] = conditional_vol
                
                # Calculate volatility adjusted returns
                daily_returns['Vol_Adjusted_Return'] = daily_returns['Return'] / daily_returns['Conditional_Vol'].replace(0, np.nan)
                
                has_garch_analysis = True
            except Exception as e:
                st.warning(f"Could not perform GARCH analysis: {e}")
                has_garch_analysis = False
        else:
            has_garch_analysis = False
            
        # Improved outlier detection using multiple features
        if len(df) >= 20:  # Need minimum data for outlier detection
            # Standardize features for outlier detection
            scaler = StandardScaler()
            features_for_outliers = ['Realized Profit', 'Quantity', 'Price']
            features_available = [f for f in features_for_outliers if f in df.columns]
            
            if len(features_available) >= 2:
                scaled_features = scaler.fit_transform(df[features_available])
                
                # Use Isolation Forest with auto contamination
                outlier_detector = IsolationForest(contamination='auto', random_state=42)
                df['Outlier'] = outlier_detector.fit_predict(scaled_features)
                outlier_trades = df[df['Outlier'] == -1]
                
                has_outlier_analysis = True
            else:
                has_outlier_analysis = False
                st.warning("Insufficient features for meaningful outlier detection")
        else:
            has_outlier_analysis = False
            
        # Improved clustering with DBSCAN
        if len(df) >= 30:  # Need minimum data for clustering
            # Standardize features for clustering
            scaler = StandardScaler()
            features_for_clustering = ['Realized Profit', 'Quantity', 'Price', 'Return']
            features_available = [f for f in features_for_clustering if f in df.columns]
            
            if len(features_available) >= 2:
                scaled_features = scaler.fit_transform(df[features_available])
                
                # Use DBSCAN for clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                df['Cluster'] = dbscan.fit_predict(scaled_features)
                
                # Get number of clusters (excluding noise which is labeled -1)
                n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
                
                has_cluster_analysis = True
            else:
                has_cluster_analysis = False
                st.warning("Insufficient features for meaningful clustering")
        else:
            has_cluster_analysis = False
        
        # Bootstrap analysis for confidence intervals
        def bootstrap_metric(data, metric_func, n_bootstrap=1000):
            bootstrap_results = []
            for _ in range(n_bootstrap):
                sample = data.sample(frac=1.0, replace=True)
                bootstrap_results.append(metric_func(sample))
            return np.percentile(bootstrap_results, [2.5, 50, 97.5])
        
        # Define metric functions
        def sharpe_func(data):
            ret = data['Return']
            if ret.std() > 0:
                return ret.mean() / ret.std() * np.sqrt(252)
            return 0
            
        def profit_factor_func(data):
            profit = data[data['Realized Profit'] > 0]['Realized Profit'].sum()
            loss = abs

def profit_factor_func(data):
    profit = data[data['Realized Profit'] > 0]['Realized Profit'].sum()
    loss = abs(data[data['Realized Profit'] < 0]['Realized Profit'].sum())
    return profit / loss if loss > 0 else float('inf')

def win_rate_func(data):
    winning_trades = len(data[data['Realized Profit'] > 0])
    total_trades = len(data)
    return winning_trades / total_trades * 100 if total_trades > 0 else 0

# Perform bootstrap analysis if enough data
if len(df) >= 30:
    try:
        sharpe_ci = bootstrap_metric(df, sharpe_func, n_bootstrap=500)
        profit_factor_ci = bootstrap_metric(df, profit_factor_func, n_bootstrap=500)
        win_rate_ci = bootstrap_metric(df, win_rate_func, n_bootstrap=500)
        has_bootstrap_analysis = True
    except Exception as e:
        st.warning(f"Could not perform bootstrap analysis: {e}")
        has_bootstrap_analysis = False
else:
    has_bootstrap_analysis = False
    st.warning("Insufficient data for bootstrap analysis (need at least 30 trades)")

# Monte Carlo simulation for strategy robustness
def monte_carlo_sim(returns, n_simulations=1000, n_periods=252):
    """Simulate future performance paths based on historical returns distribution"""
    simulations = np.zeros((n_simulations, n_periods))
    
    for i in range(n_simulations):
        # Generate random returns by sampling from historical returns with replacement
        rand_returns = np.random.choice(returns, size=n_periods, replace=True)
        
        # Calculate cumulative returns path
        cumulative_returns = np.cumprod(1 + rand_returns) - 1
        simulations[i] = cumulative_returns
    
    return simulations

if len(returns) >= 30:
    try:
        # Run Monte Carlo simulation
        mc_simulations = monte_carlo_sim(returns, n_simulations=500, n_periods=252)
        
        # Calculate statistics from simulations
        final_returns = mc_simulations[:, -1]  # Last period of each simulation
        mc_mean = np.mean(final_returns)
        mc_median = np.median(final_returns)
        mc_5th_percentile = np.percentile(final_returns, 5)
        mc_95th_percentile = np.percentile(final_returns, 95)
        
        has_monte_carlo = True
    except Exception as e:
        st.warning(f"Could not perform Monte Carlo simulation: {e}")
        has_monte_carlo = False
else:
    has_monte_carlo = False
    st.warning("Insufficient data for Monte Carlo simulation (need at least 30 trades)")

# Walk-forward analysis
def walk_forward_analysis(returns, window_size=30, step_size=10):
    """Perform walk-forward analysis to test strategy consistency"""
    if len(returns) < window_size * 2:
        return None, None
        
    n_windows = (len(returns) - window_size * 2) // step_size + 1
    if n_windows <= 0:
        return None, None
        
    train_sharpes = []
    test_sharpes = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        train_end = start_idx + window_size
        test_end = train_end + window_size
        
        train_returns = returns[start_idx:train_end]
        test_returns = returns[train_end:test_end]
        
        # Calculate Sharpe for train and test
        train_sharpe = train_returns.mean() / train_returns.std() * np.sqrt(252) if train_returns.std() > 0 else 0
        test_sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252) if test_returns.std() > 0 else 0
        
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
    
    return train_sharpes, test_sharpes

if len(returns) >= 60:  # Need at least 2 windows
    try:
        train_sharpes, test_sharpes = walk_forward_analysis(returns)
        
        if train_sharpes is not None and test_sharpes is not None:
            # Calculate correlation between train and test performance
            wf_correlation = np.corrcoef(train_sharpes, test_sharpes)[0, 1] if len(train_sharpes) > 1 else 0
            
            # Calculate average performance degradation
            wf_degradation = np.mean(np.array(test_sharpes) - np.array(train_sharpes))
            
            has_walk_forward = True
        else:
            has_walk_forward = False
    except Exception as e:
        st.warning(f"Could not perform walk-forward analysis: {e}")
        has_walk_forward = False
else:
    has_walk_forward = False
    st.warning("Insufficient data for walk-forward analysis (need at least 60 trades)")

# Display key metrics with confidence intervals
st.subheader("Key Performance Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trades", total_trades)
    st.metric("Winning Trades", f"{winning_trades} ({win_rate:.2f}%)")
    if has_bootstrap_analysis:
        st.write(f"Win Rate CI: [{win_rate_ci[0]:.2f}%, {win_rate_ci[2]:.2f}%]")
    st.metric("Average Profit per Trade", f"{average_profit_per_trade:.8f}")
with col2:
    st.metric("Net Profit", f"{net_profit:.8f}")
    st.metric("Profit Factor", f"{profit_factor:.2f}")
    if has_bootstrap_analysis:
        st.write(f"Profit Factor CI: [{profit_factor_ci[0]:.2f}, {profit_factor_ci[2]:.2f}]")
    st.metric("Total Fees Paid", f"{total_fees:.8f}")
with col3:
    st.metric("Max Drawdown", f"{max_drawdown:.8f}")
    st.metric("Avg Drawdown Duration", f"{avg_drawdown_duration:.2f} days")
    st.metric("Max Drawdown Duration", f"{max_drawdown_duration:.2f} days")

# Display advanced risk metrics
st.subheader("Advanced Risk Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.2f}")
    if has_bootstrap_analysis:
        st.write(f"Sharpe Ratio CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[2]:.2f}]")
with col2:
    st.metric("Sortino Ratio (Annualized)", f"{sortino_ratio:.2f}")
    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
with col3:
    st.metric("Expected Shortfall (5%)", f"{expected_shortfall:.4f}")
    st.metric("Omega Ratio", f"{omega:.2f}")

# Display higher-order moments
st.subheader("Distribution Characteristics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Skewness of Returns", f"{returns_skew:.2f}")
    skew_interpretation = "Positive skew (favorable)" if returns_skew > 0.2 else "Negative skew (unfavorable)" if returns_skew < -0.2 else "Approximately symmetric"
    st.write(f"Interpretation: {skew_interpretation}")
with col2:
    st.metric("Kurtosis of Returns", f"{returns_kurt:.2f}")
    kurt_interpretation = "Fat tails (high tail risk)" if returns_kurt > 3 else "Normal tails"
    st.write(f"Interpretation: {kurt_interpretation}")

# Visualizations
st.subheader("Time Series Analysis")

# Plot cumulative profit over time with drawdowns highlighted
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Time(UTC)'], df['Cumulative Profit'], label='Cumulative Profit')

# Highlight drawdown periods
for period in drawdown_periods:
    start, end, duration = period
    ax.axvspan(start, end, alpha=0.2, color='red')

ax.set_title('Cumulative Profit Over Time with Drawdown Periods')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Profit')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot returns distribution with normal and t-distribution fits
if len(returns) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of returns
    sns.histplot(returns, bins=30, kde=True, ax=ax)
    
    # Fit normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    mu, sigma = norm.fit(returns)
    y_norm = norm.pdf(x, mu, sigma)
    ax.plot(x, y_norm * len(returns) * (returns.max() - returns.min()) / 30, 'r-', 
            linewidth=2, label=f'Normal: μ={mu:.4f}, σ={sigma:.4f}')
    
    # Fit t-distribution
    df_t, loc_t, scale_t = t.fit(returns)
    y_t = t.pdf(x, df_t, loc=loc_t, scale=scale_t)
    ax.plot(x, y_t * len(returns) * (returns.max() - returns.min()) / 30, 'g--', 
            linewidth=2, label=f't-dist: ν={df_t:.2f}, loc={loc_t:.4f}, scale={scale_t:.4f}')
    
    ax.set_title('Returns Distribution with Fitted Distributions')
    ax.legend()
    st.pyplot(fig)

# Show autocorrelation if available
if has_autocorr_analysis:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(autocorr)), autocorr)
    ax.axhline(y=0, linestyle='-', color='black')
    ax.axhline(y=1.96/np.sqrt(len(returns)), linestyle='--', color='red')
    ax.axhline(y=-1.96/np.sqrt(len(returns)), linestyle='--', color='red')
    ax.set_title('Autocorrelation of Returns')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    st.pyplot(fig)

# Display regime analysis if available
if has_regime_analysis:
    st.subheader("Market Regime Analysis")
    
    # Display regime statistics
    st.write("Performance across different market regimes:")
    st.dataframe(regime_stats)
    
    # Plot regime probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_returns['Date'], regime_1_prob)
    ax.set_title('Probability of Regime 1 Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.grid(True)
    st.pyplot(fig)
    
    # Plot returns by regime
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in daily_returns['Regime'].unique():
        regime_data = daily_returns[daily_returns['Regime'] == regime]
        label = f"Regime {regime}: μ={regime_data['Return'].mean():.4f}, σ={regime_data['Return'].std():.4f}"
        sns.histplot(regime_data['Return'], label=label, alpha=0.6, ax=ax)
    ax.set_title('Returns Distribution by Market Regime')
    ax.legend()
    st.pyplot(fig)

# Display GARCH analysis if available
if has_garch_analysis:
    st.subheader("Volatility Analysis (GARCH)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_returns['Date'], conditional_vol)
    ax.set_title('Conditional Volatility Over Time (GARCH Model)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(True)
    st.pyplot(fig)
    
    # Plot volatility-adjusted returns vs raw returns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(daily_returns['Date'], daily_returns['Return'])
    ax1.set_title('Raw Returns')
    ax1.grid(True)
    
    ax2.plot(daily_returns['Date'], daily_returns['Vol_Adjusted_Return'])
    ax2.set_title('Volatility-Adjusted Returns')
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# Display outlier analysis if available
if has_outlier_analysis:
    st.subheader("Outlier Analysis")
    
    st.write(f"Detected {len(outlier_trades)} outlier trades out of {len(df)} total trades.")
    
    if len(outlier_trades) > 0:
        st.write("Outlier trades statistics:")
        outlier_stats = outlier_trades.describe()
        st.dataframe(outlier_stats)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[df['Outlier'] == 1]['Realized Profit'], df[df['Outlier'] == 1]['Quantity'], 
                 label='Normal', alpha=0.5)
        ax.scatter(outlier_trades['Realized Profit'], outlier_trades['Quantity'],
                 label='Outliers', color='red', alpha=0.5)
        ax.set_title('Outlier Trades Detection')
        ax.set_xlabel('Realized Profit')
        ax.set_ylabel('Quantity')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# Display clustering analysis if available
if has_cluster_analysis:
    st.subheader("Trade Clustering Analysis")
    
    st.write(f"Identified {n_clusters} distinct trade clusters plus noise points.")
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'Realized Profit': ['count', 'mean', 'sum', 'std'],
        'Return': ['mean', 'std'],
        'Quantity': 'mean',
        'Price': 'mean'
    })
    
    st.write("Cluster statistics:")
    st.dataframe(cluster_stats)
    
    # Visualize clusters
    if len(features_available) >= 2:
        # Use PCA for dimension reduction if more than 2 features
        if len(features_available) > 2:
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)
            x_label = 'PCA Component 1'
            y_label = 'PCA Component 2'
        else:
            reduced_features = scaled_features
            x_label = features_available[0]
            y_label = features_available[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each cluster
        for cluster in sorted(df['Cluster'].unique()):
