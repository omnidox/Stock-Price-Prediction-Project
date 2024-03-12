# %% [markdown]
# # Stock Price Prediction Project
# 
# ## 1. Project Concept and Scope
# ### Objective
# To predict future stock prices of specific companies representing different market sectors, using historical data.
# 
# ### Scope
# The project will focus on PLUG (Energy), NIO (Automotive), NTLA (Healthcare), SNAP (Communication Services), and CHPT (Industrials).
# 
# In the first part of our project, we will try to analyze the data. and in the second part, we will forecast the stock market.

# %% [markdown]
# ### Overview of Selected Stocks
# 
# This notebook provides an overview of five distinct stocks, each representing different sectors and industries. The stocks covered are:
# 
# 1. **PLUG (Plug Power Inc.)**: 
#    - Sector: Energy
#    - Industry: Electrical Equipment & Parts
#    - Description: Plug Power is an innovator in hydrogen and fuel cell technology, providing comprehensive hydrogen fuel cell turnkey solutions.
# 
# 2. **NIO (NIO Inc.)**:
#    - Sector: Automotive
#    - Industry: Auto Manufacturers
#    - Description: NIO is a pioneer in China's premium electric vehicle market, specializing in designing, manufacturing, and selling electric vehicles.
# 
# 3. **NTLA (Intellia Therapeutics Inc.)**:
#    - Sector: Healthcare
#    - Industry: Biotechnology
#    - Description: Intellia Therapeutics is a leading biotechnology company developing therapies using a CRISPR/Cas9 gene-editing system.
# 
# 4. **SNAP (Snap Inc.)**:
#    - Sector: Communication Services
#    - Industry: Internet Content & Information
#    - Description: Snap Inc. is the parent company of Snapchat, a popular social media platform known for its ephemeral messaging and multimedia features.
# 
# 5. **CHPT (ChargePoint Holdings Inc.)**:
#    - Sector: Industrials
#    - Industry: Specialty Industrial Machinery
#    - Description: ChargePoint Holdings is at the forefront of electric vehicle charging infrastructure, offering a comprehensive array of charging solutions.
# 
# Each of these companies represents a unique investment opportunity within its respective sector, reflecting different aspects of technological and industrial advancement.
# 

# %% [markdown]
# ## 2. Data Collection
# - Utilize Alpha Vantage API for historical stock price data.
# - Gather comprehensive data including prices, volumes, and market indicators.

# %%
"""
This script imports necessary libraries for stock price prediction.
"""
import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %%
symbols_list = ['PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT']

# %%
import time
import os
from alpha_vantage.timeseries import TimeSeries

def retrieve_stock_data(symbols):
    """
    Retrieve historical stock data for a given list of symbols using Alpha Vantage API.
    Deletes old CSV files if newer data is found and downloaded. If API limit is reached, it will print a message and continue with the next symbol.

    Parameters:
    symbols (list): A list of stock symbols to retrieve data for.

    Returns:
    None
    """
    # Read the API key from the file
    with open('AlphaVantage.txt', 'r') as file:
        api_key = file.read().strip()

    # Create a TimeSeries object with your API key
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Loop through the symbols and retrieve the historical data
    for symbol in symbols:
        try:
            # Get the historical data for the symbol
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            
            # Sort the data by index (date) just in case
            data.sort_index(inplace=True)

            # Get the first and last dates
            first_date = data.index[0].strftime('%Y-%m-%d')
            last_date = data.index[-1].strftime('%Y-%m-%d')

            # Generate the new file name
            new_file_name = f'{first_date}_{last_date}_{symbol}_historical_data.csv'

            # Check if a file for this symbol already exists
            existing_files = [f for f in os.listdir() if f.endswith(f'{symbol}_historical_data.csv')]
            if existing_files:
                # Sort files to find the most recent one
                existing_files.sort()
                most_recent_file = existing_files[-1]

                # Extract dates from the most recent file name
                existing_first_date, existing_last_date, *_ = most_recent_file.split('_')

                # Compare dates (strings comparison works because of the YYYY-MM-DD format)
                if existing_first_date <= first_date and existing_last_date >= last_date:
                    print(f"Data already up-to-date for {symbol}")
                    continue
                else:
                    # Remove older files
                    for file in existing_files:
                        os.remove(file)
                        print(f"Old file {file} deleted for {symbol}")

            # Save the new data to a CSV file
            data.to_csv(new_file_name)
            print(f"New data saved for {symbol}: {new_file_name}")

        except ValueError as e:
            print(f"Error retrieving data for {symbol}: {e}")
            # Optional: sleep for some time before continuing, or handle the error as needed
            # time.sleep(60)  # Sleep for 1 minute, for example

# Example usage
symbols_list = ['PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT']
retrieve_stock_data(symbols_list)


# %%
import glob
import pandas as pd

def load_stock_data(symbols):
    """
    Load the most recent, up-to-date historical data CSV files into variables.
    The 'Date' column in each CSV file is used as the DataFrame index and parsed as dates.

    Parameters:
    symbols (list): A list of stock symbols to load data for.

    Returns:
    dict: A dictionary containing the loaded data frames, with stock symbols as keys.
    """
    data_frames = {}

    for symbol in symbols:
        # Find the most recent CSV file for the symbol
        files = glob.glob(f'*{symbol}_historical_data.csv')
        if files:
            files.sort()
            most_recent_file = files[-1]

            # Load the CSV file into a data frame with 'Date' as the index column and parse dates
            data_frames[symbol] = pd.read_csv(most_recent_file, index_col='date', parse_dates=['date'])
            print(f"Data loaded for {symbol}: {most_recent_file}")
        else:
            print(f"No data found for {symbol}")

    return data_frames

# Example usage
# symbols_list should be defined earlier in your script
# e.g., symbols_list = ['AAPL', 'GOOGL', 'MSFT']
stock_data = load_stock_data(symbols_list)


# %% [markdown]
# ### Looking at the heads of our data

# %%
stock_data['PLUG'].head()

# %%
stock_data['NIO'].head()

# %%
stock_data['NTLA'].head()

# %%
stock_data['SNAP'].head()

# %%
stock_data['CHPT'].head()

# %% [markdown]
# ### Looking at summary of our data and checking for Null Values for all of our stocks

# %%
stock_data['PLUG']['2019':'2024'].describe()

# %%
stock_data['PLUG'].columns

# %%
stock_data['PLUG'].info()

# %%
# Check for missing values

stock_data['PLUG'].isna().sum()

# %%
stock_data['NIO']['2019':'2024'].describe()

# %%
stock_data['NIO'].info()

# %%
stock_data['NIO'].isna().sum()

# %%
stock_data['NTLA']['2019':'2024'].describe()

# %%
stock_data['NTLA'].info()

# %%
stock_data['NTLA'].isna().sum()

# %%
stock_data['SNAP']['2019':'2024'].describe()

# %%
stock_data['SNAP'].info()

# %%
stock_data['SNAP'].isna().sum()

# %%
stock_data['CHPT']['2019':'2024'].describe()

# %%
stock_data['CHPT'].info()

# %%
stock_data['CHPT'].isna().sum()

# %% [markdown]
# ### PLUG Closing Prices in its entirety. 

# %%
fig = px.line(stock_data['PLUG'], x=stock_data['PLUG'].index, y='4. close', title='PLUG Closing Prices')
fig.show()

# %% [markdown]
# ### PLUG Opening Prices from 2019 to 2024

# %%
fig = px.line(stock_data['PLUG']['2019':'2024'], x=stock_data['PLUG']['2019':'2024'].index, y='4. close', title='PLUG Closing Prices')
fig.show()

# %% [markdown]
# ## PLUG Closing Price Distribution Analysis
# 
# The provided composite plot offers a detailed distribution analysis of the closing prices for PLUG. The visualization combines both a histogram and a box plot to present a comprehensive view of the data.
# 
# ### Histogram Analysis
# - The histogram shows a **high frequency of closing and opening prices**, as evidenced by the tall bars on the left.
# - There is a **long right tail** in the distribution, indicating occasional spikes in the closing price to values significantly higher than the norm.
# - The presence of **right-skewness** in the histogram suggests that the majority of the data points are gathered on the lower end of the price spectrum, with fewer instances of high prices.
# 
# ### Box Plot Analysis
# - The **central box** represents the interquartile range (IQR), which holds the middle 50% of the closing price data. The **median** is denoted by the line within the box.
# - The **whiskers** extend from the IQR to display the full range of data, excluding outliers. The tips of the whiskers mark the lowest and highest non-outlier closing prices.
# - **Outliers** are plotted as individual points beyond the whiskers, representing closing prices that are significantly higher than typical values.
# 
# In conclusion, the plot indicates that while the closing prices of PLUG are most commonly lower, there have been several instances of significant price surges. The outliers highlight periods of atypical price behavior.
# 

# %%
"""
Create a histogram plot of the closing price distribution for the 'PLUG' stock.

Parameters:
- stock_data (DataFrame): The stock data containing the 'PLUG' stock information.
- nbins (int): The number of bins to use for the histogram.

Returns:
- None
"""
fig = px.histogram(
    stock_data['PLUG']['2019':'2024'], 
    x='4. close', 
    marginal='box',
    nbins=200,
    title='PLUG Closing Price Distribution'
)
fig.update_layout(bargap=0.1)
fig.show()

# %%
"""
Create a histogram plot of the Opening price distribution for the 'PLUG' stock.

Parameters:
- stock_data (DataFrame): The stock data containing the 'PLUG' stock information.
- nbins (int): The number of bins to use for the histogram.

Returns:
- None
"""
fig = px.histogram(
    stock_data['PLUG']['2019':'2024'], 
    x='1. open', 
    marginal='box',
    color_discrete_sequence=['red'],
    nbins=200,
    title='PLUG Opening Price Distribution'
)
fig.update_layout(bargap=0.1)
fig.show()

# %% [markdown]
# ### Correlation of opening and closing PLUG Prices

# %%
fig = px.scatter(stock_data['PLUG'], 
                 x='1. open', 
                 y='4. close', 
                 opacity=0.8,
                  
                 title='Open vs. Close')
fig.update_traces(marker_size=5)
fig.show()

# %%
stock_data['PLUG'].corr()

# %% [markdown]
# ## Graphical Summary of 'PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT' Stocks

# %%
stock_data['PLUG']['2019':'2024'].plot(subplots=True, figsize=(10,12))
plt.title('PLUG stock attributes from 2019 to 2024')
plt.show()

# %%
stock_data['NIO']['2019':'2024'].plot(subplots=True, figsize=(10,12))
plt.title('NIO stock attributes from 2019 to 2024')
plt.show()

# %%
stock_data['NIO'].corr()

# %%
stock_data['NTLA']['2019':'2024'].plot(subplots=True, figsize=(10,12))
plt.title('NTLA stock attributes from 2019 to 2024')
plt.show()

# %%
stock_data['SNAP']['2019':'2024'].plot(subplots=True, figsize=(10,12))
plt.title('SNAP stock attributes from 2019 to 2024')
plt.show()

# %%
stock_data['CHPT']['2019':'2024'].plot(subplots=True, figsize=(10,12))
plt.title('CHPT stock attributes from 2019 to 2024')
plt.show()

# %% [markdown]
# ## Plotting our data together prior normalization

# %%
# Plotting before normalization
# symbols_list = ['PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT']


stock_data['PLUG']['2019':'2024']['2. high'].plot()
stock_data['NIO']['2019':'2024']['2. high'].plot()
stock_data['NTLA']['2019':'2024']['2. high'].plot()
stock_data['SNAP']['2019':'2024']['2. high'].plot()
stock_data['CHPT']['2019':'2024']['2. high'].plot()


# Adding labels and title
plt.title('High Prices of Selected Stocks (2019-2024)')  # Title of the graph
plt.xlabel('Date')  # X-axis label
plt.ylabel('High Price Dollar Amount')  # Y-axis label

# Adding a legend
plt.legend(['PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT'])

# Display the plot
plt.show()


# %% [markdown]
# ## Normalization of Stock Data
# 
# In the analysis, we implement a normalization technique known as **Indexing to a Base Value**. This method allows us to compare the performance of different stocks over a common timescale and relative scale. By doing so, we can visualize and compare the relative returns of each stock as if they all started from the same point. Here's how the normalization is applied:
# 
# - We begin by selecting the daily high prices for each stock within the 2019 to 2024 period.
# - Each daily high price is then divided by the first recorded high price of the respective stock at the beginning of 2019 (`iloc[0]`).
# - The resulting ratio is multiplied by 100, effectively setting the base value to 100 for each stock.
# 
# This approach allows each stock's price series to start at a value of 100, and subsequent prices are adjusted to reflect the percentage change from this initial value. The resulting normalized series is plotted to show the growth trajectory of each stock relative to its starting point, making it straightforward to compare their performances over time.

# %%
# Normalizing and comparison
# Stocks start from 100

normalized_PLUG = stock_data['PLUG']['2019':'2024']['2. high'].div(stock_data['PLUG']['2019':'2024']['2. high'].iloc[0]).mul(100)
normalized_NIO = stock_data['NIO']['2019':'2024']['2. high'].div(stock_data['NIO']['2019':'2024']['2. high'].iloc[0]).mul(100)
normalized_NTLA = stock_data['NTLA']['2019':'2024']['2. high'].div(stock_data['NTLA']['2019':'2024']['2. high'].iloc[0]).mul(100)
normalized_SNAP = stock_data['SNAP']['2019':'2024']['2. high'].div(stock_data['SNAP']['2019':'2024']['2. high'].iloc[0]).mul(100)
normalized_CHPT = stock_data['CHPT']['2019':'2024']['2. high'].div(stock_data['CHPT']['2019':'2024']['2. high'].iloc[0]).mul(100)

normalized_PLUG.plot()
normalized_NIO.plot()
normalized_NTLA.plot()
normalized_SNAP.plot()
normalized_CHPT.plot()

# Adding labels and title
plt.title('Normalized High Prices of Selected Stocks (2019-2024)')  # Title of the graph
plt.xlabel('Date')  # X-axis label
plt.ylabel('Normalized High Price (Base 100)')  # Y-axis label

# Adding a legend
plt.legend(['PLUG', 'NIO', 'NTLA', 'SNAP', 'CHPT'])

# Display the plot
plt.show()

# %% [markdown]
# ## Value of Expanding Window Functions in Stock Analysis
# 
# The utilization of expanding window functions in the analysis of our stock data provides several benefits that are crucial for financial analysis and trading strategies:
# 
# ### Trend Identification
# - The **expanding mean** offers a visualization of the long-term trend, smoothing out short-term fluctuations to highlight the underlying direction of the stock's movement.
# 
# ### Volatility Analysis
# - The **expanding standard deviation** is indicative of the stock's volatility. An increasing trend suggests rising volatility, while a stable or decreasing trend points to reduced volatility.
# 
# ### Benchmarking
# - Serving as a benchmark, the expanding mean allows analysts to gauge whether the stock is performing above or below its historical average.
# 
# ### Support and Resistance Levels
# - Traders may consider the expanding mean as a dynamic support or resistance level, with the expectation that prices may revert to this mean over time.
# 
# ### Risk Management
# - Volatility analysis is integral to risk management. The expanding standard deviation can inform the setting of stop-loss orders and position sizing to align with an investor's risk tolerance.
# 
# ### Investment Decisions
# - Investors looking for long-term stability and growth can use the expanding mean to assess whether a stock fits their investment profile.
# 
# ### Valuation Models
# - Historical volatility is a key input in valuation models, such as options pricing, where the expanding standard deviation can provide the necessary data.
# 
# ### Anomaly Detection
# - Sudden and significant deviations from the expanding mean or standard deviation can signal anomalies that may require further investigation.
# 
# ### Market Comparisons
# - By normalizing stock data, these expanding metrics can be used to compare different stocks or to benchmark a stock against a market index, providing insights into relative performance.
# 
# Incorporating these metrics into stock analysis offers a dynamic and in-depth understanding of a stock's historical performance, informing a wide range of investment and trading decisions.
# 

# %%
# Expanding Window Functions PLUG

PLUG_expanding_mean = stock_data['PLUG']['2019':'2024']['2. high'].expanding().mean()
PLUG_std = stock_data['PLUG']['2019':'2024']['2. high'].expanding().std()
stock_data['PLUG']['2019':'2024']['2. high'].plot()
PLUG_expanding_mean.plot()
PLUG_std.plot()
plt.title('PLUG Expanding Window Functions')
plt.legend(['PLUG', 'PLUG Rolling Mean', 'PLUG Standard Deviation'])
plt.show()

# %%
# Expanding Window Functions NIO

NIO_expanding_mean = stock_data['NIO']['2019':'2024']['2. high'].expanding().mean()
NIO_std = stock_data['NIO']['2019':'2024']['2. high'].expanding().std()
stock_data['NIO']['2019':'2024']['2. high'].plot()
NIO_expanding_mean.plot()
NIO_std.plot()
plt.title('NIO Expanding Window Functions')
plt.legend(['NIO', 'NIO Rolling Mean', 'NIO Standard Deviation'])
plt.show()

# %%
# Expanding Window Functions NTLA

NTLA_expanding_mean = stock_data['NTLA']['2019':'2024']['2. high'].expanding().mean()
NTLA_std = stock_data['NTLA']['2019':'2024']['2. high'].expanding().std()
stock_data['NTLA']['2019':'2024']['2. high'].plot()
NTLA_expanding_mean.plot()
NTLA_std.plot()
plt.title('NTLA Expanding Window Functions')
plt.legend(['NTLA', 'NTLA Rolling Mean', 'NTLA Standard Deviation'])
plt.show()

# %%
# Expanding Window Functions SNAP

SNAP_expanding_mean = stock_data['SNAP']['2019':'2024']['2. high'].expanding().mean()
SNAP_std = stock_data['SNAP']['2019':'2024']['2. high'].expanding().std()
stock_data['SNAP']['2019':'2024']['2. high'].plot()
SNAP_expanding_mean.plot()
SNAP_std.plot()
plt.title('SNAP Expanding Window Functions')
plt.legend(['SNAP', 'SNAP Rolling Mean', 'SNAP Standard Deviation'])
plt.show()

# %%
# Expanding Window Functions CHPT

CHPT_expanding_mean = stock_data['CHPT']['2019':'2024']['2. high'].expanding().mean()
CHPT_std = stock_data['CHPT']['2019':'2024']['2. high'].expanding().std()
stock_data['CHPT']['2019':'2024']['2. high'].plot()
CHPT_expanding_mean.plot()
CHPT_std.plot()
plt.title('CHPT Expanding Window Functions')
plt.legend(['CHPT', 'CHPT Rolling Mean', 'CHPT Standard Deviation'])
plt.show()

# %% [markdown]
# ## Value of Rolling Window Functions in Stock Analysis
# 
# Applying rolling window functions to our stocks, like the rolling mean or moving average, to stock data over a specific period, such as 50 days, offers valuable insights for financial analysis and trading strategies:
# 
# ### Trend Identification
# - The **50-day rolling mean** smooths out daily price fluctuations, providing a clearer view of the short-term trend direction without the noise of daily volatility.
# 
# ### Momentum Indication
# - The rolling mean can serve as a momentum indicator. A stock price moving above its 50-day rolling mean may indicate increasing momentum and vice versa.
# 
# ### Trading Signals
# - Crossovers between the stock price and the 50-day rolling mean can be used as trading signals. When the stock price crosses above the rolling mean, it could be a buy signal, while a cross below might suggest a sell signal.
# 
# ### Support and Resistance Levels
# - The 50-day rolling mean may act as a temporary support or resistance level. Prices often bounce off this dynamic line during trends, providing potential entry and exit points.
# 
# ### Risk Management
# - Traders can use the 50-day rolling mean to set trailing stop-loss orders. If the price falls below this level, it could trigger a sale to prevent further losses.
# 
# ### Indicator of Sentiment
# - The position of the stock price relative to the 50-day rolling mean can indicate market sentiment. Prices consistently above this average may reflect bullish sentiment, while prices below may suggest bearish sentiment.
# 
# ### Technical Analysis
# - The rolling mean is a foundational component of various technical analysis indicators and strategies, such as Bollinger Bands and Moving Average Convergence Divergence (MACD).
# 
# ### Visualization of Price Stability
# - By analyzing how tightly the price follows the rolling mean, investors can get a sense of the stock's stability. A tight tracking suggests a stable stock, while wide divergence may indicate instability.
# 
# ### Comparison with Other Time Frames
# - Comparing the 50-day rolling mean with longer time frames, like a 50-day or 200-day moving average, can provide additional context on different market phases (short-term vs. long-term trends).
# 
# Implementing the 50-day rolling mean in stock analysis is a powerful method to decipher the market's movements, offering a dynamic view that is more responsive to recent price changes, thus informing a broad array of trading and investment decisions.
# 

# %%
# Rolling Window (50 day) Functions PLUG

PLUG_rolling_mean = stock_data['PLUG']['2019':'2024']['2. high'].rolling(50).mean()
PLUG_rolling_std = stock_data['PLUG']['2019':'2024']['2. high'].rolling(50).std()
stock_data['PLUG']['2019':'2024']['2. high'].plot()
PLUG_rolling_mean.plot()
PLUG_rolling_std.plot()
plt.title('PLUG Rolling Window Functions')
plt.legend(['PLUG', 'PLUG Rolling Mean', 'PLUG Standard Deviation'])
plt.show()

# %%
# Rolling Window (50 day) Functions NIO

NIO_rolling_mean = stock_data['NIO']['2019':'2024']['2. high'].rolling(50).mean()
NIO_rolling_std = stock_data['NIO']['2019':'2024']['2. high'].rolling(50).std()
stock_data['NIO']['2019':'2024']['2. high'].plot()
NIO_rolling_mean.plot()
NIO_rolling_std.plot()
plt.title('NIO Rolling Window Functions')
plt.legend(['NIO', 'NIO Rolling Mean', 'NIO Standard Deviation'])
plt.show()

# %%
# Rolling Window (50 day) Functions NTLA

NTLA_rolling_mean = stock_data['NTLA']['2019':'2024']['2. high'].rolling(50).mean()
NTLA_rolling_std = stock_data['NTLA']['2019':'2024']['2. high'].rolling(50).std()
stock_data['NTLA']['2019':'2024']['2. high'].plot()
NTLA_rolling_mean.plot()
NTLA_rolling_std.plot()
plt.title('NTLA Rolling Window Functions')
plt.legend(['NTLA', 'NTLA Rolling Mean', 'NTLA Standard Deviation'])
plt.show()

# %%
# Rolling Window (50 day) Functions SNAP

SNAP_rolling_mean = stock_data['SNAP']['2019':'2024']['2. high'].rolling(50).mean()
SNAP_rolling_std = stock_data['SNAP']['2019':'2024']['2. high'].rolling(50).std()
stock_data['SNAP']['2019':'2024']['2. high'].plot()
SNAP_rolling_mean.plot()
SNAP_rolling_std.plot()
plt.title('SNAP Rolling Window Functions')
plt.legend(['SNAP', 'SNAP Rolling Mean', 'SNAP Standard Deviation'])
plt.show()


# %%
# Rolling Window (50 day) Functions CHPT

CHPT_rolling_mean = stock_data['CHPT']['2019':'2024']['2. high'].rolling(50).mean()
CHPT_rolling_std = stock_data['CHPT']['2019':'2024']['2. high'].rolling(50).std()
stock_data['CHPT']['2019':'2024']['2. high'].plot()
CHPT_rolling_mean.plot()
CHPT_rolling_std.plot()
plt.title('CHPT Rolling Window Functions')
plt.legend(['CHPT', 'CHPT Rolling Mean', 'CHPT Standard Deviation'])
plt.show()

# %%
from pylab import rcParams
import statsmodels.api as sm

# %% [markdown]
# ## Trend and Seasonality
# 
# ## Decomposition Analysis of PLUG Stock Data
# 
# The code snippets below performs a time series decomposition on the 'high' prices of all our stocks from 2019 to 2024. This analysis is essential to understand underlying patterns in the stock price movements. Here's a breakdown of the decomposition and the insights we can gather:
# 
# ### Code Explanation
# - `rcParams['figure.figsize'] = 11, 9`: Sets the size of the plot to 11 inches wide and 9 inches tall for better visibility.
# - `decomposed_PLUG = sm.tsa.seasonal_decompose(stock_data['PLUG']['2019':'2024']['2. high'], period=252)`: This line of code utilizes the `seasonal_decompose` function from the `statsmodels` library to decompose the PLUG stock's high prices into three components:
#   - Trend
#   - Seasonality
#   - Residual
#   The decomposition uses a period of 252 days, corresponding to the approximate number of trading days in a year, suggesting an annual cycle analysis.
# - `figure = decomposed_PLUG.plot()`: Generates a plot of the decomposed time series.
# - `plt.show()`: Displays the plot with the decomposed components.
# 
# ### Insights from the Decomposition
# 
# 1. **Trend Component**:
#    - The trend line smooths out short-term fluctuations and shows the long-term direction of the stock's high prices. An upward trend may indicate a general increase in stock prices over the years, while a downward trend suggests a decline.
# 
# 2. **Seasonal Component**:
#    - This part of the decomposition reveals any regular patterns that repeat annually. For instance, specific times of the year might consistently show higher or lower stock prices, which could be linked to the company's business cycle, market sentiment, or external economic factors.
# 
# 3. **Residual Component**:
#    - The residuals represent the random variations in the stock price that cannot be explained by the trend or seasonality. Analyzing these residuals can help identify unusual events or anomalies that might require further investigation.
# 
# By performing this decomposition, we can gain a deeper understanding of the PLUG stock's behavior over time, separating systematic seasonal patterns and long-term trends from random, irregular movements. This information is crucial for investors and analysts in making informed decisions about future investments and understanding the stock's market dynamics.
# 

# %%
# Decomposition of PLUG

rcParams['figure.figsize'] = 11, 9
decomposed_PLUG = sm.tsa.seasonal_decompose(stock_data['PLUG']['2019':'2024']['2. high'], period=252)
figure = decomposed_PLUG.plot()

axes = figure.axes
axes[0].set_title('Time Series Decomposition of PLUG Stock High Prices', fontsize=12)

plt.show()

# %% [markdown]
# ## Time Series Decomposition of PLUG Stock High Prices
# 
# The attached figure represents the result of a time series decomposition of PLUG stock's high prices from 2019 to 2024. The decomposition separates the data into trend, seasonal, and residual components.
# 
# ### Observed Data
# - The initial plot represents the original observed data, showing significant volatility and a notable peak in early 2021. Following the peak, there's a visible decline, indicating a period of price correction or market adjustment.
# 
# ### Trend Component
# - The trend plot illustrates the long-term progression of the stock's high prices. There's an upward movement culminating in early 2021, signaling a period of growth, after which a downward trend is evident, suggesting a shift in the stock's momentum.
# 
# ### Seasonal Component
# - The seasonal component exhibits the cyclical patterns within the data, repeated over fixed intervals. The consistent amplitude of these cycles points to a stable seasonal influence over the observed time frame.
# 
# ### Residual Component
# - The residuals reflect the irregularities or noise after accounting for trend and seasonality. Around early 2021, the increased variability in residuals could indicate external factors or atypical events influencing stock prices that are not explained by seasonality or trend.
# 
# By analyzing these components, we can infer that the stock experienced growth leading up to 2021, followed by a downward adjustment. The seasonal fluctuations could be linked to regular market patterns or company-specific events, while the residuals may highlight anomalies or non-recurring events impacting the stock's performance.
# 
# This decomposition is instrumental in understanding the underlying behaviors of the stock's high prices, assisting investors and analysts in making data-driven decisions that account for trend, seasonality, and irregular occurrences in the market.
# 

# %%
# Decompostion of NIO 

rcParams['figure.figsize'] = 11, 9
decomposed_NIO = sm.tsa.seasonal_decompose(stock_data['NIO']['2019':'2024']['2. high'], period=252)
figure = decomposed_NIO.plot()

axes = figure.axes
axes[0].set_title('Time Series Decomposition of NIO Stock High Prices', fontsize=12)

plt.show()

# %%
# Decompostion of NTLA

rcParams['figure.figsize'] = 11, 9
decomposed_NTLA = sm.tsa.seasonal_decompose(stock_data['NTLA']['2019':'2024']['2. high'], period=252)
figure = decomposed_NTLA.plot()

axes = figure.axes
axes[0].set_title('Time Series Decomposition of NTLA Stock High Prices', fontsize=12)

plt.show()

# %%
# Decompostion of SNAP

rcParams['figure.figsize'] = 11, 9
decomposed_SNAP = sm.tsa.seasonal_decompose(stock_data['SNAP']['2019':'2024']['2. high'], period=252)
figure = decomposed_SNAP.plot()

axes = figure.axes
axes[0].set_title('Time Series Decomposition of SNAP Stock High Prices', fontsize=12)


plt.show()

# %%
# Decompostion of CHPT

rcParams['figure.figsize'] = 11, 9
decomposed_CHPT = sm.tsa.seasonal_decompose(stock_data['CHPT']['2019':'2024']['2. high'], period=252)
figure = decomposed_CHPT.plot()

axes = figure.axes
axes[0].set_title('Time Series Decomposition of CHPT Stock High Prices', fontsize=12)


plt.show()

# %% [markdown]
# ## Predictions

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(stock_data['PLUG']['2019':'2024']['4. close'])

# Use Matplotlib's date locators and formatters to handle dates on the x-axis
ax = plt.gca()  # Get the current Axes instance

# Set major ticks to the first month of each quarter
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
# Set major tick formatter to a formatter that prints the date as 'year-month'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("PLUG Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(stock_data['NIO']['2019':'2024']['4. close'])

# Use Matplotlib's date locators and formatters to handle dates on the x-axis
ax = plt.gca()  # Get the current Axes instance

# Set major ticks to the first month of each quarter
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
# Set major tick formatter to a formatter that prints the date as 'year-month'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("NIO Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(stock_data['NTLA']['2019':'2024']['4. close'])

# Use Matplotlib's date locators and formatters to handle dates on the x-axis
ax = plt.gca()  # Get the current Axes instance

# Set major ticks to the first month of each quarter
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
# Set major tick formatter to a formatter that prints the date as 'year-month'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("NTLA Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(stock_data['SNAP']['2019':'2024']['4. close'])

# Use Matplotlib's date locators and formatters to handle dates on the x-axis
ax = plt.gca()  # Get the current Axes instance

# Set major ticks to the first month of each quarter
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
# Set major tick formatter to a formatter that prints the date as 'year-month'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("SNAP Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(stock_data['CHPT']['2019':'2024']['4. close'])

# Use Matplotlib's date locators and formatters to handle dates on the x-axis
ax = plt.gca()  # Get the current Axes instance

# Set major ticks to the first month of each quarter
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
# Set major tick formatter to a formatter that prints the date as 'year-month'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("CHPT Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# %% [markdown]
# ## We start to Normalize or data for training

# %%
# Extract the '4. close' column of the PLUG stock from the 'stock_data' dataframe
price_PLUG = stock_data['PLUG']['2019':'2024']['4. close']

# Display information about the 'price_PLUG' series, such as data type and number of non-null values
price_PLUG.info()

# %%
# Print the 'price_PLUG' series to display the closing prices of PLUG stock
print(price_PLUG)

# %%
# Extract the index from the 'price_PLUG' series, which contains the dates
dates = price_PLUG.index


# %%
# Print the extracted dates
print(dates)

# %%
# Import the MinMaxScaler class from the sklearn.preprocessing module
from sklearn.preprocessing import MinMaxScaler


# Create an instance of the MinMaxScaler with the feature range set to (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

# Scale the 'price_PLUG' data, reshaping it to fit the scaler's expected input format and store the scaled values in a new variable
price_PLUG_scaled = scaler.fit_transform(price_PLUG.values.reshape(-1, 1))

# %%
# Print the scaled prices to see the transformed data
print(price_PLUG_scaled)

# %%
import pandas as pd

# Create a DataFrame from the scaled prices using the dates from 'price_PLUG' as the index and naming the column 'Scaled Price'
# This operation combines the scaled price data with their corresponding dates in a tabular format
scaled_prices_with_dates = pd.DataFrame(price_PLUG_scaled, index=price_PLUG.index, columns=['Scaled Price'])

# Print the newly created DataFrame to display the scaled prices alongside their dates
print(scaled_prices_with_dates)


# %%
# Print the index of the 'scaled_prices_with_dates' DataFrame, which contains the dates
print(scaled_prices_with_dates.index)


# %%
# Retrieve and print the specific date at index position 5 from the DataFrame's index, showing how to access individual dates
specific_date = scaled_prices_with_dates.index[5]  # Retrieves the date at position 5
print(specific_date)


# %%
# Print descriptive statistics for the 'scaled_prices_with_dates' DataFrame, providing insights into the distribution of scaled prices

print(scaled_prices_with_dates.describe())

# %% [markdown]
# ## Statistical Summary of Scaled PLUG Stock Prices (2019-2024)
# 
# The summary statistics of the scaled PLUG stock prices provide valuable insights into the distribution and variation of the stock prices over the period from 2019 to 2024. Below is a detailed analysis of these statistics:
# 
# - **Count**: The dataset contains **1,272 observations**, representing the total number of trading days or time points for which the PLUG stock prices were recorded and subsequently scaled.
# 
# - **Mean**: The average value of the scaled prices is approximately **-0.600**. This suggests that on average, the stock prices tend to be below the midpoint of the -1 to 1 scale, indicating a skew towards the lower end of the observed price range before scaling.
# 
# - **Standard Deviation (Std)**: With a standard deviation of **0.373**, there is a moderate level of variation or dispersion from the mean. This indicates that while there is variability in the scaled stock prices, it is not extremely high.
# 
# - **Minimum (Min)**: The minimum value of the scaled prices is **-1.000**, the lowest possible value on the scale, indicating that at least one trading day saw the stock price at its lowest observed level during the period analyzed.
# 
# - **First Quartile (25%)**: At the first quartile, 25% of the scaled prices are **less than or equal to -0.921**, suggesting a significant number of days with prices near the lower end of the observed range before scaling.
# 
# - **Median (50%)**: The median scaled price is **-0.691**, meaning that half of the trading days had scaled prices below this value. The median being lower than 0 but higher than the mean suggests a skew towards lower scaled prices.
# 
# - **Third Quartile (75%)**: Three quarters of the trading days had scaled prices that are **less than or equal to -0.337**, illustrating the distribution's skew towards lower values within the scaled range.
# 
# - **Maximum (Max)**: The maximum scaled price value is **1.000**, the highest possible value on the scale, indicating at least one trading day where the stock price reached the highest observed level during the analyzed period.
# 
# In summary, the scaled PLUG stock prices from 2019 to 2024 demonstrate a general skew towards the lower end of the scale, with a significant range between the minimum and maximum values. This analysis highlights the distribution's lean towards lower scaled prices, along with a moderate level of fluctuation as indicated by the standard deviation.
# 

# %%
# Retrieve and display the minimum value among the scaled prices, showing the lowest scaled price in the data set
scaled_prices_with_dates['Scaled Price'].values.min()

# %%
# Print the array of scaled prices without the dates, showing just the transformed price data
scaled_prices_with_dates['Scaled Price'].values

# %%
# This command returns only the numerical values from the DataFrame as a NumPy array, excluding the index (dates). 

scaled_prices_with_dates.values

# %%
# Convert the DataFrame to a records array including the index
records_with_dates = scaled_prices_with_dates.reset_index().to_records(index=False)

print(records_with_dates)


# %%
# Import the required libraries: matplotlib for plotting and pandas for data manipulation
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'price_PLUG' contains the original closing prices of the PLUG stock
# and 'scaled_prices_with_dates' is a DataFrame with dates as indexes and scaled prices

# Display the minimum and maximum values of the scaled prices to check the effectiveness of scaling
print("Min value (after scaling):", scaled_prices_with_dates['Scaled Price'].min())
print("Max value (after scaling):", scaled_prices_with_dates['Scaled Price'].max())

# Show the first few entries of the original and scaled data to observe the initial transformation effect
print("First few original values:\n", price_PLUG.head())
print("First few scaled values:\n", scaled_prices_with_dates.head())

# Begin plotting to compare the distribution of original and scaled data
plt.figure(figsize=(12, 6))  # Set the figure size for better visibility

# Plot the distribution of original data using a histogram
plt.subplot(1, 2, 1)  # Prepare a subplot environment with 1 row and 2 columns, this is the first plot
plt.hist(price_PLUG.values, bins=50)  # Plot histogram with 50 bins for better resolution
plt.title("Before Scaling")  # Title to indicate this plot shows the original data distribution

# Plot the distribution of scaled data using a histogram
plt.subplot(1, 2, 2)  # Second plot in the subplot environment
plt.hist(scaled_prices_with_dates['Scaled Price'].values, bins=50)  # Use scaled prices for histogram
plt.title("After Scaling")  # Title to differentiate from the pre-scaling plot

plt.show()  # Display the plots


# %% [markdown]
# ## Dataset Preparation for Week-Ahead Stock Price Prediction
# 
# Given a dataset with 1000 samples, the `split_data_week_ahead` function aims to prepare this dataset for a machine learning model to predict stock prices a week (7 days) ahead, using sequences of historical data defined by the `lookback` period. Here's how the function processes and splits the data, assuming `lookback = 20` and `forecast_horizon = 7`:
# 
# ### Step 1: Creating Sequences
# - The function creates sequences of length `lookback + forecast_horizon`. Each sequence includes 20 days of historical data followed by 7 days leading up to the target day.
# - Since each sequence must have a target value 7 days after the last day in the `lookback` period, the loop runs until `len(data_raw) - lookback - forecast_horizon + 1`. This ensures every sequence has enough future data to include the target value.
# - For 1000 samples, the maximum index we can start a sequence from is `1000 - 20 - 7 + 1 = 974`. This means the function creates sequences starting from index 0 to index 973.
# 
# ### Step 2: Splitting Data into Training and Testing Sets
# - The data is split into training and testing sets, with 80% used for training and 20% for testing. The `test_set_size` is calculated as `0.2 * total number of sequences`.
# - With the formula provided, if we have approximately 974 sequences, `test_set_size` would be `0.2 * 974 ≈ 195` (rounded).
# - Thus, `train_set_size` becomes `974 - 195 = 779`. This means the first 779 sequences are used for training, and the remaining 195 sequences are used for testing.
# 
# ### Step 3: Preparing `x_train`, `y_train`, `x_test`, and `y_test`
# - `x_train` and `x_test` are prepared by selecting all but the last 7 days of each sequence for `x` values, effectively using the first 20 days of each sequence as input features.
# - `y_train` and `y_test` extract the target value, which is the value 7 days after the last day in the `lookback` period for each sequence. This is achieved by using `-forecast_horizon` to select the target day as the output label.
# 
# ### Example Output Dimensions
# Given these parameters, let's estimate the shapes of the output arrays:
# - `x_train.shape`: With 779 training sequences, each containing 20 days of data, and assuming each day's data is represented by a single feature (due to `price_PLUG_scaled` being a 1D array transformed into 2D for the sequences), `x_train` would have a shape of `(779, 20, 1)`.
# - `y_train.shape`: Each sequence in `x_train` corresponds to a single target value (the price 7 days later), so `y_train` would have a shape of `(779, 1)`.
# - `x_test.shape`: There are 195 testing sequences, so `x_test` would have a shape of `(195, 20, 1)`.
# - `y_test.shape`: Similarly, `y_test` would have a shape of `(195, 1)`.
# 
# These steps ensure the model learns to predict the stock price 7 days ahead based on the preceding 20 days of data, with separate datasets for training and testing to evaluate its predictive performance.
# 

# %%
import numpy as np

# outputs all days outlined by the forecast horizon
def split_data_week_ahead_with_dates_multi(stock, lookback, forecast_horizon):
    data_raw = stock['Scaled Price'].values
    dates = stock.index
    
    data = []
    date_labels = []
    
    for index in range(len(data_raw) - lookback - forecast_horizon + 1):
        data.append(data_raw[index: index + lookback + forecast_horizon])
        # Collect a range of dates for each forecast horizon instead of a single date
        date_labels.append(dates[index + lookback: index + lookback + forecast_horizon])
    
    data = np.array(data)
    
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    
    x_train = data[:train_set_size, :-forecast_horizon]
    y_train = data[:train_set_size, -forecast_horizon:]
    x_test = data[train_set_size:, :-forecast_horizon]
    y_test = data[train_set_size:, -forecast_horizon:]
    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # Adjusting how dates are split to accommodate the change
    dates_train = date_labels[:train_set_size]
    dates_test = date_labels[train_set_size:]
    
    return x_train, y_train, x_test, y_test, dates_train, dates_test


# %%
# Outputs just the last date of each forecast horizon

import numpy as np

def split_data_week_ahead_with_dates_single(stock, lookback, forecast_horizon):
    data_raw = stock['Scaled Price'].values
    dates = stock.index
    
    data = []
    date_labels = []
    
    for index in range(len(data_raw) - lookback - forecast_horizon + 1):
        data.append(data_raw[index: index + lookback + forecast_horizon])
        # Now collecting only the date corresponding to the last day of each forecast horizon
        date_labels.append(dates[index + lookback + forecast_horizon - 1])  # Just the last day
    
    data = np.array(data)
    
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    
    # Extracting input sequences for training and testing
    x_train = data[:train_set_size, :-forecast_horizon]
    x_test = data[train_set_size:, :-forecast_horizon]
    
    # Modifying to focus only on the last day of the forecast horizon for y_train and y_test
    y_train = data[:train_set_size, -1].reshape(-1, 1)  # Reshaping to keep 2D structure
    y_test = data[train_set_size:, -1].reshape(-1, 1)   # Reshaping to keep 2D structure
    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # Adjusting how dates are handled to match the new y_train and y_test structure
    dates_train = [date_labels[i] for i in range(train_set_size)]
    dates_test = [date_labels[i] for i in range(train_set_size, len(date_labels))]
    
    return x_train, y_train, x_test, y_test, dates_train, dates_test


# %%
# outputs all days outlined by the forecast horizon

# Demonstrate how to use the 'split_data_week_ahead_with_dates' function to prepare data for time series forecasting.

# Set 'lookback' to 20 days, indicating that each input sequence will use the past 20 days of scaled stock prices to make predictions.
lookback = 20

# Set 'forecast_horizon' to 7 days, defining the prediction window as one week into the future from the last date of each input sequence.
forecast_horizon = 7

# Call the function with the scaled stock price DataFrame, 'lookback', and 'forecast_horizon' parameters.
# This splits the data into training and testing sets for both the input features (x) and the targets (y),
# along with the corresponding dates for the end of each sequence in both sets.
x_train_multi, y_train_multi, x_test_multi, y_test_multi, dates_train_multi, dates_test_multi = split_data_week_ahead_with_dates_multi(scaled_prices_with_dates, lookback, forecast_horizon)

# Print the shapes of the returned datasets to verify their dimensions and ensure they are correctly structured for model training and evaluation.
# 'x_train.shape' and 'x_test.shape' should reflect the number of sequences, the lookback period, and the dimensionality of the data (1 in this case, as we're dealing with univariate time series).
# 'y_train.shape' and 'y_test.shape' indicate the number of sequences and the forecast horizon, showing how many days ahead each sequence is aiming to predict.
print(f'x_train_multi.shape = {x_train_multi.shape}')
print(f'y_train_multi.shape = {y_train_multi.shape}')
print(f'x_test_multi.shape = {x_test_multi.shape}')
print(f'y_test_multi.shape = {y_test_multi.shape}')


# %%
# Outputs just the last date of each forecast horizon

# Demonstrate how to use the 'split_data_week_ahead_with_dates' function to prepare data for time series forecasting.

# Set 'lookback' to 20 days, indicating that each input sequence will use the past 20 days of scaled stock prices to make predictions.
lookback = 20

# Set 'forecast_horizon' to 7 days, defining the prediction window as one week into the future from the last date of each input sequence.
forecast_horizon = 7

# Call the function with the scaled stock price DataFrame, 'lookback', and 'forecast_horizon' parameters.
# This splits the data into training and testing sets for both the input features (x) and the targets (y),
# along with the corresponding dates for the end of each sequence in both sets.
x_train_single, y_train_single, x_test_single, y_test_single, dates_train_single, dates_test_single = split_data_week_ahead_with_dates_single(scaled_prices_with_dates, lookback, forecast_horizon)

# Print the shapes of the returned datasets to verify their dimensions and ensure they are correctly structured for model training and evaluation.
# 'x_train.shape' and 'x_test.shape' should reflect the number of sequences, the lookback period, and the dimensionality of the data (1 in this case, as we're dealing with univariate time series).
# 'y_train.shape' and 'y_test.shape' indicate the number of sequences and the forecast horizon, showing how many days ahead each sequence is aiming to predict.
print(f'x_train_single.shape = {x_train_single.shape}')
print(f'y_train_single.shape = {y_train_single.shape}')
print(f'x_test_single.shape = {x_test_single.shape}')
print(f'y_test_single.shape = {y_test_single.shape}')


# %% [markdown]
# ## Dataset Structure for Time Series Forecasting
# 
# The shapes of `x_train`, `y_train`, `x_test`, and `y_test` provide detailed insights into the dataset's structure prepared for training and testing in a machine learning model, specifically tailored for a time series forecasting task. Below is a breakdown of what each shape represents:
# 
# ### Training Dataset
# 
# - **`x_train.shape = (997, 20, 1)`**
#   - **997 training samples**: This dataset comprises 997 sequences, each intended for training the model.
#   - **20 time steps per sample**: Corresponds to the `lookback` period, indicating that each input sequence spans 20 days of historical data.
#   - **1 feature per time step**: Suggests that each day is represented by a single feature, such as the daily closing price of a stock, which the model uses to learn and make predictions.
# 
# The structure implies that the model is designed to learn from sequences of 20 consecutive days to predict future values, leveraging a single feature (likely the stock's closing price) from each day.
# 
# - **`y_train.shape = (997, 1)`**
#   - **997 target values**: Matches the number of training samples, ensuring each sequence has a corresponding prediction target.
#   - **Single-value target**: Indicates the prediction task is focused on forecasting a single outcome (e.g., the stock price) 7 days into the future from the last day of each 20-day input sequence.
# 
# This confirms the model's objective: to predict the outcome 7 days ahead for each 20-day sequence presented during training.
# 
# ### Testing Dataset
# 
# - **`x_test.shape = (249, 20, 1)`**
#   - **249 testing samples**: The model's performance will be evaluated on these separate sequences.
#   - **Identical sequence structure**: Like the training set, each testing sequence consists of 20 time steps with 1 feature per step, ensuring consistency in how the model applies its learned patterns to make predictions.
# 
# - **`y_test.shape = (249, 1)`**
#   - **249 target values for testing**: Each testing sequence has a corresponding target, aligning with the training set's structure.
#   - **Forecasting 7 days ahead**: Mirrors the training target setup, focusing on predicting the stock price (or a similar outcome) for a single day, 7 days into the future from each sequence.
# 
# ### Summary
# 
# The dataset is meticulously divided into training and testing sets, with 997 sequences allocated for training and 249 for testing. Each sequence leverages 20 days of historical data to forecast a single outcome, ensuring the model is trained and evaluated on its ability to predict future values based on a fixed historical window. The singular feature dimension likely represents a specific data aspect, such as the daily closing price, which is critical for time series forecasting models. This structured approach facilitates learning from historical data to predict future values (e.g., stock prices 7 days ahead), allowing the model to be assessed on unseen data to gauge its predictive generalization capabilities.
# 

# %%
# Import necessary libraries from PyTorch for neural network construction and operations
import torch
import torch.nn as nn

# Convert the training and testing input datasets (x_train and x_test) from NumPy arrays to PyTorch tensors.
# This conversion is necessary because PyTorch models require inputs to be in the form of tensors.
# The '.type(torch.Tensor)' ensures the data type is set to the default tensor type, suitable for model input.
x_train_gru_multi = torch.from_numpy(x_train_multi).type(torch.Tensor)
x_test_gru_multi = torch.from_numpy(x_test_multi).type(torch.Tensor)

# Similarly, convert the training and testing target datasets (y_train and y_test) from NumPy arrays to PyTorch tensors,
# allowing these datasets to be used as labels in the training and evaluation of the PyTorch model.
y_train_gru_multi = torch.from_numpy(y_train_multi).type(torch.Tensor)
y_test_gru_multi = torch.from_numpy(y_test_multi).type(torch.Tensor)


# %%
# Import necessary libraries from PyTorch for neural network construction and operations
import torch
import torch.nn as nn

# Convert the training and testing input datasets (x_train and x_test) from NumPy arrays to PyTorch tensors.
# This conversion is necessary because PyTorch models require inputs to be in the form of tensors.
# The '.type(torch.Tensor)' ensures the data type is set to the default tensor type, suitable for model input.
x_train_gru_single = torch.from_numpy(x_train_single).type(torch.Tensor)
x_test_gru_single = torch.from_numpy(x_test_single).type(torch.Tensor)

# Similarly, convert the training and testing target datasets (y_train and y_test) from NumPy arrays to PyTorch tensors,
# allowing these datasets to be used as labels in the training and evaluation of the PyTorch model.
y_train_gru_single = torch.from_numpy(y_train_single).type(torch.Tensor)
y_test_gru_single = torch.from_numpy(y_test_single).type(torch.Tensor)


# %% [markdown]
# # Training Parameters for RNN Model
# 
# The training parameters listed are typical hyperparameters for a Recurrent Neural Network (RNN) model. The appropriateness of these parameters depends on the specific details the dataset and the problem at hand.
# 
# ## Parameters Breakdown
# 
# 1. **Input Dimension (`input_dim`)**:
#    - `input_dim = 1`
#    - Indicates the number of input features per time step.
#    - For a single feature like closing stock price, this is set to 1.
# 
# 2. **Hidden Dimension (`hidden_dim`)**:
#    - `hidden_dim = 32`
#    - Represents the size of the hidden state in the RNN.
#    - Larger `hidden_dim` can capture more complex patterns but increases computational load and overfitting risk.
# 
# 3. **Number of Layers (`num_layers`)**:
#    - `num_layers = 2`
#    - Indicates two stacked RNN layers.
#    - More layers can help learn complex patterns but increase model complexity and overfitting risk.
# 
# 4. **Output Dimension (`output_dim`)**:
#    - `output_dim = 1`
#    - Dimensionality of the output, typically 1 for predicting a single continuous value.
# 
# 5. **Number of Epochs (`num_epochs`)**:
#    - `num_epochs = 105`
#    - Number of times the entire training dataset passes through the network.
#    - Requires monitoring for overfitting or underfitting.
# 
# ## Considerations
# 
# - **Overfitting vs. Underfitting**: Monitor training and validation loss. Overfitting is indicated by decreasing training loss and increasing validation loss. Use regularization techniques like dropout to mitigate.
# - **Experimentation**: Adjust hyperparameters based on model performance during training.
# - **Validation**: Use a validation set to monitor performance on unseen data and help tune hyperparameters.
# 
# In summary, these parameters are a reasonable starting point, but they may need to be adjusted based on the model's performance during training.
# 

# %%
# Define the dimensions and configuration for a neural network model for seven outputs, specifically for a time series forecasting task.

# input_dim: Specifies the number of input features per timestep in the input sequence. 
# For univariate time series forecasting (predicting one variable using itself), this is set to 1.
input_dim = 1

# hidden_dim: Defines the size of the hidden layer(s). Here, 32 units are chosen for the hidden layers, 
# which determines the model's capacity to learn representations from the data.
hidden_dim = 32

# num_layers: Sets the number of recurrent layers in the network. Using 2 layers here suggests a deeper model 
# for capturing more complex patterns in the data.
num_layers = 2

# output_dim: Specifies the number of output features. For seven day forecasting (predicting seven variables),
# this is set to 7.
output_dim_multi = 7

# num_epochs: Indicates the total number of training cycles the model will go through. Setting this to 105 
# epochs means the entire training dataset will be passed forward and backward through the network 105 times.
num_epochs = 105


# %%
# Define the dimensions and configuration for a neural network model for a single output, specifically for a time series forecasting task.

# input_dim: Specifies the number of input features per timestep in the input sequence. 
# For univariate time series forecasting (predicting one variable using itself), this is set to 1.
input_dim = 1

# hidden_dim: Defines the size of the hidden layer(s). Here, 32 units are chosen for the hidden layers, 
# which determines the model's capacity to learn representations from the data.
hidden_dim = 32

# num_layers: Sets the number of recurrent layers in the network. Using 2 layers here suggests a deeper model 
# for capturing more complex patterns in the data.
num_layers = 2

# output_dim: Specifies the number of output features. For seven day forecasting (predicting seven variables),
# this is set to 7.
output_dim_single = 1

# num_epochs: Indicates the total number of training cycles the model will go through. Setting this to 105 
# epochs means the entire training dataset will be passed forward and backward through the network 105 times.
num_epochs = 105


# %%
import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    GRU Neural Network for time series forecasting.
    
    Attributes:
    - input_dim: The number of input features per timestep.
    - hidden_dim: The number of features in the hidden state h.
    - num_layers: The number of stacked GRU layers.
    - output_dim: The number of output features (forecast horizon).
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initializes the GRU model with the specified parameters and layers.
        
        Parameters:
        - input_dim (int): Number of input features.
        - hidden_dim (int): Size of GRU hidden layers.
        - num_layers (int): Number of GRU layers.
        - output_dim (int): Number of output features.
        """
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim  # Size of the hidden layer
        self.num_layers = num_layers  # Number of GRU layers
        
        # The GRU layer; batch_first=True means the input tensors will be of shape (batch_size, seq_length, features)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer that maps the GRU layer output to the desired output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - x (Tensor): The input sequence to the GRU model.
        
        Returns:
        - Tensor: The output of the model.
        """
        # Initialize hidden state with zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Forward propagate the GRU
        # out: tensor containing the output features (h_t) from the last layer of the GRU, for each t.
        out, (hn) = self.gru(x, (h0.detach()))  # detach h0 to prevent backprop through the initial hidden state
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) 
        return out


# %%
# Instantiate the GRU model with specified architecture parameters.
# input_dim: Number of features per timestep in the input sequence.
# hidden_dim: Size of the hidden layers within GRU cells.
# output_dim: Number of output features (e.g., forecasted values).
# num_layers: Number of recurrent layers in the GRU.
model_multi = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim_multi, num_layers=num_layers)

# Define the loss function for the model's output. The Mean Squared Error (MSE) Loss is used here,
# which is common for regression problems, including time series forecasting.
# 'reduction=mean' specifies that the losses are averaged over all observations for each minibatch.
criterion = torch.nn.MSELoss(reduction='mean')

# Choose the Adam optimizer for updating model parameters. Adam is an adaptive learning rate optimization algorithm
# designed to handle sparse gradients on noisy problems, making it suitable for many types of neural network training.
# lr=0.01 sets the initial learning rate, determining the step size at each iteration while moving toward a minimum of the loss function.
optimiser_multi = torch.optim.Adam(model_multi.parameters(), lr=0.01)


# %%
# Instantiate the GRU model with specified architecture parameters.
# input_dim: Number of features per timestep in the input sequence.
# hidden_dim: Size of the hidden layers within GRU cells.
# output_dim: Number of output features (e.g., forecasted values).
# num_layers: Number of recurrent layers in the GRU.
model_single = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim_single, num_layers=num_layers)

# Define the loss function for the model's output. The Mean Squared Error (MSE) Loss is used here,
# which is common for regression problems, including time series forecasting.
# 'reduction=mean' specifies that the losses are averaged over all observations for each minibatch.
criterion = torch.nn.MSELoss(reduction='mean')

# Choose the Adam optimizer for updating model parameters. Adam is an adaptive learning rate optimization algorithm
# designed to handle sparse gradients on noisy problems, making it suitable for many types of neural network training.
# lr=0.01 sets the initial learning rate, determining the step size at each iteration while moving toward a minimum of the loss function.
optimiser_single = torch.optim.Adam(model_single.parameters(), lr=0.01)


# %%
import time  # Import the time module to track the duration of the training process.

# Initialize an array to record the loss at each epoch.
hist_multi = np.zeros(num_epochs)

# Record the start time to calculate the total training duration.
start_time = time.time()

# This list 'gru' appears unused in the provided snippet. 
# If it's intended for storing model states or outputs at each epoch, consider adding relevant code or removing the declaration.
gru = []

# Begin training over the specified number of epochs.
for t in range(num_epochs):
    # Forward pass: Compute the predicted y by passing x to the model.
    y_train_pred_multi = model_multi(x_train_gru_multi)

    # Compute and print the loss using the Mean Squared Error between predicted and actual y values in the training set.
    loss_multi = criterion(y_train_pred_multi, y_train_gru_multi)
    print(f"Epoch {t} MSE: {loss_multi.item()}")
    hist_multi[t] = loss_multi.item()  # Store the loss in 'hist' for later analysis or plotting.

    # Zero the gradients before running the backward pass.
    optimiser_multi.zero_grad()
    
    # Backward pass: Compute the gradient of the loss with respect to model parameters.
    loss_multi.backward()
    
    # Perform a single optimization step (parameter update).
    optimiser_multi.step()

# Calculate and print the total training time.
training_time = time.time() - start_time    
print(f"Training time: {training_time}")


# %%
import time  # Import the time module to track the duration of the training process.

# Initialize an array to record the loss at each epoch.
hist_single = np.zeros(num_epochs)

# Record the start time to calculate the total training duration.
start_time = time.time()

# This list 'gru' appears unused in the provided snippet. 
# If it's intended for storing model states or outputs at each epoch, consider adding relevant code or removing the declaration.
gru = []

# Begin training over the specified number of epochs.
for t in range(num_epochs):
    # Forward pass: Compute the predicted y by passing x to the model.
    y_train_pred_single = model_single(x_train_gru_single)

    # Compute and print the loss using the Mean Squared Error between predicted and actual y values in the training set.
    loss_single = criterion(y_train_pred_single, y_train_gru_single)
    print(f"Epoch {t} MSE: {loss_single.item()}")
    hist_single[t] = loss_single.item()  # Store the loss in 'hist' for later analysis or plotting.

    # Zero the gradients before running the backward pass.
    optimiser_single.zero_grad()
    
    # Backward pass: Compute the gradient of the loss with respect to model parameters.
    loss_single.backward()
    
    # Perform a single optimization step (parameter update).
    optimiser_single.step()

# Calculate and print the total training time.
training_time = time.time() - start_time    
print(f"Training time: {training_time}")


# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

# Example data and variables for demonstration
# Assuming y_train_pred, y_train_gru are PyTorch tensors and dates_train is modified as per the previous explanation
# y_train_pred = model_output_for_y_train_pred
# y_train_gru = actual_output_for_y_train_gru
# dates_train = modified_dates_train_with_lists_of_dates

# Make predictions using the trained model on both the training and testing datasets.
y_test_pred_multi = model_multi(x_test_gru_multi)

# Inverse transform predictions and actuals
y_train_pred_inv_multi = scaler.inverse_transform(y_train_pred_multi.detach().numpy())
y_train_inv_multi = scaler.inverse_transform(y_train_gru_multi.detach().numpy())

y_test_pred_inv_multi = scaler.inverse_transform(y_test_pred_multi.detach().numpy())  # Inverse transform for testing predictions
y_test_inv_multi = scaler.inverse_transform(y_test_gru_multi.detach().numpy())  # Inverse transform for actual testing values

# Initialize dictionaries to hold DataFrames for each day's predictions and actuals
train_predict_multi = {}
train_original_multi = {}
test_predict_multi = {}
test_original_multi = {}


for day in range(forecast_horizon):
    # Extract predictions and actuals for the current day
    train_day_predictions = y_train_pred_inv_multi[:, day]
    train_day_actuals = y_train_inv_multi[:, day]
    
    # Extract dates for the current day from each sequence's dates list
    train_day_dates = [dates[day] for dates in dates_train_multi]  # Assuming dates_train is a list of lists
    
    # Create DataFrames for the current day's predictions and actuals
    train_predict_multi[f"Day {day + 1}"] = pd.DataFrame(train_day_predictions, columns=[f"Predicted Day {day + 1}"])
    train_original_multi[f"Day {day + 1}"] = pd.DataFrame(train_day_actuals, columns=[f"Actual Day {day + 1}"])
    
    # Set the index of each DataFrame to the corresponding dates
    train_predict_multi[f"Day {day + 1}"].index = pd.to_datetime(train_day_dates)
    train_original_multi[f"Day {day + 1}"].index = pd.to_datetime(train_day_dates)

    # Extract test predictions and actuals for the current day
    test_day_predictions = y_test_pred_inv_multi[:, day]
    test_day_actuals = y_test_inv_multi[:, day]
    
    # Extract dates for the current day from each sequence's dates list in the test dataset
    # This assumes 'dates_test_multi' is structured similarly to 'dates_train_multi', as a list of lists
    test_day_dates = [dates[day] for dates in dates_test_multi]  # Update this variable as per your data structure
    
    # Create DataFrames for the current day's test predictions and actuals
    test_predict_multi[f"Day {day + 1}"] = pd.DataFrame(test_day_predictions, columns=[f"Predicted Day {day + 1}"])
    test_original_multi[f"Day {day + 1}"] = pd.DataFrame(test_day_actuals, columns=[f"Actual Day {day + 1}"])
    
    # Set the index of each DataFrame to the corresponding test dates
    test_predict_multi[f"Day {day + 1}"].index = pd.to_datetime(test_day_dates)
    test_original_multi[f"Day {day + 1}"].index = pd.to_datetime(test_day_dates)


# %%
# Access the DataFrame for day 1's predictions
day_7_predictions = train_predict_multi["Day 7"]
# Access the DataFrame for day 1's actuals
day_7_actuals = train_original_multi["Day 7"]

# Print or analyze the DataFrames as needed
print(day_7_predictions.head())
print(day_7_actuals.head())


# %%
print(test_predict_multi["Day 7"].head())

print(test_original_multi["Day 7"].head())

# %%
import pandas as pd

# Assuming 'scaler', 'y_train_pred', and 'y_train_gru' are already defined
# Assuming 'dates_train' is already prepared by the 'split_data_week_ahead_with_dates_2' function


# Make predictions using the trained model on both the training and testing datasets.
y_test_pred_single = model_single(x_test_gru_single)

# Invert predictions to transform them back to the original data scale, undoing the earlier normalization.
# This step is necessary to make the error metrics comparable to the original data values.

# Inverse transform the predictions and actual values for the last day
y_train_pred_inv_single = scaler.inverse_transform(y_train_pred_single.detach().numpy())
y_train_inv_single = scaler.inverse_transform(y_train_gru_single.detach().numpy())
y_test_pred_inv_single = scaler.inverse_transform(y_test_pred_single.detach().numpy())  # Inverse transform for testing predictions
y_test_inv_single = scaler.inverse_transform(y_test_gru_single.detach().numpy())  # Inverse transform for actual testing values

# Create DataFrames for the predictions and actual values
train_predict_single = pd.DataFrame(y_train_pred_inv_single, columns=['Predicted'])
train_original_single = pd.DataFrame(y_train_inv_single, columns=['Actual'])
test_predict_single = pd.DataFrame(y_test_pred_inv_single, columns=['Predicted'])
test_original_single = pd.DataFrame(y_test_inv_single, columns=['Actual'])

# Convert 'dates_train' to datetime format if not already
dates_train_datetime_single = pd.to_datetime(dates_train_single)
dates_test_datetime_single = pd.to_datetime(dates_test_single)

# Set the index of each DataFrame to the corresponding dates for the last day's forecast
train_predict_single.index = dates_train_datetime_single
train_original_single.index = dates_train_datetime_single
test_predict_single.index = dates_test_datetime_single
test_original_single.index = dates_test_datetime_single

# Now, both 'predict_df' and 'original_df' DataFrames have dates as their index


# %%
print(train_predict_single.head())
print(train_original_single.head())
print(test_predict_single.head())
print(test_original_single.head())


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set the visual style of the plots to 'darkgrid' for better readability and aesthetics.
sns.set_style("darkgrid")

# Assuming 'hist' contains the training loss history
# Create a figure object to hold the subplots, with adjusted size for clarity.
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# First subplot for the original data and the training predictions.
# Note: Use 'original2.index' and 'predict2.index' directly for the x-axis values
# to ensure dates are correctly plotted. Since 'original2' and 'predict2' share the same index, 
# you can use either for plotting both sets of data.
sns.lineplot(ax=axs[0], x=train_original_single.index, y=train_original_single['Actual'], label="Actual", color='royalblue')
sns.lineplot(ax=axs[0], x=train_predict_single.index, y=train_predict_single['Predicted'], label="Training Prediction (GRU)", color='tomato')

axs[0].set_title('PLUG closing stock price', size=14, fontweight='bold')
axs[0].set_xlabel("Date", size=14)
axs[0].set_ylabel("Cost (USD)", size=14)
# Optionally, rotate the x-axis date labels for better readability
axs[0].tick_params(axis='x', rotation=45)

# Second subplot for the training loss over epochs.
sns.lineplot(ax=axs[1], data=hist_single, color='royalblue')
axs[1].set_xlabel("Epoch", size=14)
axs[1].set_ylabel("Loss", size=14)
axs[1].set_title("Training Loss", size=14, fontweight='bold')

plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Choose which day to plot
day_to_plot = "Day 7"

# Set the visual style of the plots to 'darkgrid'
sns.set_style("darkgrid")

# Create a figure object to hold the subplots, with adjusted size for clarity
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# Plotting predictions vs. actual values for the chosen day
sns.lineplot(ax=axs[0], x=train_predict_multi[day_to_plot].index, y=train_predict_multi[day_to_plot]['Predicted ' + day_to_plot], label="Prediction", color='tomato')
sns.lineplot(ax=axs[0], x=train_original_multi[day_to_plot].index, y=train_original_multi[day_to_plot]['Actual ' + day_to_plot], label="Actual", color='royalblue')

axs[0].set_title(f'{day_to_plot} Predictions vs Actual', size=14, fontweight='bold')
axs[0].set_xlabel("Date", size=14)
axs[0].set_ylabel("Price (USD)", size=14)
axs[0].tick_params(axis='x', rotation=45)  # Rotate the x-axis labels for better readability

# Assuming 'hist' contains the training loss history
# Plotting training loss
sns.lineplot(ax=axs[1], data=hist, color='royalblue')
axs[1].set_title("Training Loss", size=14, fontweight='bold')
axs[1].set_xlabel("Epoch", size=14)
axs[1].set_ylabel("Loss", size=14)

plt.tight_layout()  # Automatically adjust subplot params to give specified padding
plt.show()


# %%
import math, time
from sklearn.metrics import mean_squared_error

# Make predictions using the trained model on both the training and testing datasets.
y_test_pred_multi = model_multi(x_test_gru_multi)

# Invert predictions to transform them back to the original data scale, undoing the earlier normalization.
# This step is necessary to make the error metrics comparable to the original data values.
y_train_pred_inv_multi = scaler.inverse_transform(y_train_pred_multi.detach().numpy())  # Inverse transform for training predictions
y_train_inv_multi = scaler.inverse_transform(y_train_gru_multi.detach().numpy())  # Inverse transform for actual training values
y_test_pred_inv_multi = scaler.inverse_transform(y_test_pred_multi.detach().numpy())  # Inverse transform for testing predictions
y_test_inv_multi = scaler.inverse_transform(y_test_gru_multi.detach().numpy())  # Inverse transform for actual testing values

# # Calculate the root mean squared error (RMSE) for both training and testing datasets.
# # RMSE is a standard way to measure the error of a model in predicting quantitative data.
# # Squaring the errors, averaging them, and taking the square root gives us the RMSE.
# trainScore = math.sqrt(mean_squared_error(y_train_inv[:,6], y_train_pred_inv[:,6]))  # RMSE for the training data
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test_inv[:,6], y_test_pred_inv[:,6]))  # RMSE for the testing data
# print('Test Score: %.2f RMSE' % (testScore))

for i in range(7):  # Assuming there are 7 columns
    trainScore = math.sqrt(mean_squared_error(y_train_inv_multi[:, i], y_train_pred_inv_multi[:, i]))
    testScore = math.sqrt(mean_squared_error(y_test_inv_multi[:, i], y_test_pred_inv_multi[:, i]))
    print(f'Column {i+1} Train Score: {trainScore:.2f} RMSE')
    print(f'Column {i+1} Test Score: {testScore:.2f} RMSE')


# Append the training and testing scores, along with the training time, to the 'gru' list.
# This could be useful for later analysis or comparison with other models.
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)


# %%
import plotly.graph_objects as go

# Placeholder data for demonstration
days = list(range(1, 8))
# Placeholder lists for RMSE values
train_rmse = []
test_rmse = []

# Calculate RMSE for each day and append to lists
for i in range(7):  # Assuming there are 7 days in the forecast horizon
    train_rmse.append(math.sqrt(mean_squared_error(y_train_inv_multi[:, i], y_train_pred_inv_multi[:, i])))
    test_rmse.append(math.sqrt(mean_squared_error(y_test_inv_multi[:, i], y_test_pred_inv_multi[:, i])))

# Create a Plotly graph object for plotting
fig = go.Figure()

# Add traces for train and test RMSE
fig.add_trace(go.Scatter(x=days, y=train_rmse, mode='lines+markers', name='Train RMSE'))
fig.add_trace(go.Scatter(x=days, y=test_rmse, mode='lines+markers', name='Test RMSE'))

# Update layout for aesthetics
fig.update_layout(
    title='RMSE over 7 Days',
    xaxis_title='Day',
    yaxis_title='RMSE',
    template='plotly_dark'
)

# Showing the figure
fig.show()


# %%
import math, time
from sklearn.metrics import mean_squared_error

# Make predictions using the trained model on both the training and testing datasets.
y_test_pred_single = model_single(x_test_gru_single)

# Invert predictions to transform them back to the original data scale, undoing the earlier normalization.
# This step is necessary to make the error metrics comparable to the original data values.
y_train_pred_inv_single = scaler.inverse_transform(y_train_pred_single.detach().numpy())  # Inverse transform for training predictions
y_train_inv_single = scaler.inverse_transform(y_train_gru_single.detach().numpy())  # Inverse transform for actual training values
y_test_pred_inv_single = scaler.inverse_transform(y_test_pred_single.detach().numpy())  # Inverse transform for testing predictions
y_test_inv_single = scaler.inverse_transform(y_test_gru_single.detach().numpy())  # Inverse transform for actual testing values

# Calculate the root mean squared error (RMSE) for both training and testing datasets.
# RMSE is a standard way to measure the error of a model in predicting quantitative data.
# Squaring the errors, averaging them, and taking the square root gives us the RMSE.
trainScore = math.sqrt(mean_squared_error(y_train_inv_single[:,0], y_train_pred_inv_single[:,0]))  # RMSE for the training data
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_inv_single[:,0], y_test_pred_inv_single[:,0]))  # RMSE for the testing data
print('Test Score: %.2f RMSE' % (testScore))

# %%
import plotly.graph_objects as go

# Extracting Day 7 data for plotting
day_7_train_pred = train_predict_single
day_7_train_orig = train_original_single
day_7_test_pred = test_predict_single
day_7_test_orig = test_original_single

# Combining train and test data for a continuous plot
full_pred = pd.concat([day_7_train_pred, day_7_test_pred])
full_orig = pd.concat([day_7_train_orig, day_7_test_orig])

# Creating the figure object
fig = go.Figure()

# Adding traces for train prediction, test prediction, and actual values
fig.add_trace(go.Scatter(x=day_7_train_pred.index, y=day_7_train_pred['Predicted'], mode='lines', name='Train Prediction'))
fig.add_trace(go.Scatter(x=day_7_test_pred.index, y=day_7_test_pred['Predicted'], mode='lines', name='Test Prediction'))
fig.add_trace(go.Scatter(x=full_orig.index, y=full_orig['Actual'], mode='lines', name='Actual Value'))

# Updating layout for aesthetics
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,  # Show tick labels for dates
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        title_text='Close (USD)',
        titlefont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
    ),
    showlegend=True,
    template='plotly_dark',
)

# Adding a title to the plot
fig.update_layout(title_text='PLUG Stock Prediction for Day 7', title_x=0.5)

# Showing the figure
fig.show()


# %%
def plot_prediction_for_day(train_predict, train_original, test_predict, test_original, day_number):
    import plotly.graph_objects as go
    import pandas as pd

    # Validating day_number
    if day_number < 1 or day_number > 7:
        print("Day number must be between 1 and 7.")
        return

    day_key = f"Day {day_number}"

    # Extracting data for the specified day
    day_train_pred = train_predict[day_key]
    day_train_orig = train_original[day_key]
    day_test_pred = test_predict[day_key]
    day_test_orig = test_original[day_key]

    # Combining train and test data for a continuous plot
    full_pred = pd.concat([day_train_pred, day_test_pred])
    full_orig = pd.concat([day_train_orig, day_test_orig])

    # Creating the figure object
    fig = go.Figure()

    # Adding traces for train prediction, test prediction, and actual values
    fig.add_trace(go.Scatter(x=day_train_pred.index, y=day_train_pred[f'Predicted {day_key}'], mode='lines', name='Train Prediction'))
    fig.add_trace(go.Scatter(x=day_test_pred.index, y=day_test_pred[f'Predicted {day_key}'], mode='lines', name='Test Prediction'))
    fig.add_trace(go.Scatter(x=full_orig.index, y=full_orig[f'Actual {day_key}'], mode='lines', name='Actual Value'))

    # Updating layout for aesthetics
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,  # Show tick labels for dates
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
        ),
        showlegend=True,
        template='plotly_dark',
    )

    # Adding a title to the plot
    fig.update_layout(title_text=f'PLUG Stock Prediction for {day_key}', title_x=0.5)

    # Showing the figure
    fig.show()

# Example usage (assuming the dictionaries train_predict_multi, train_original_multi, test_predict_multi, test_original_multi are defined as per your data)
# plot_prediction_for_day(train_predict_multi, train_original_multi, test_predict_multi, test_original_multi, 3)


# %%
plot_prediction_for_day(train_predict_multi, train_original_multi, test_predict_multi, test_original_multi, 7)


