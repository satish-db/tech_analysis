import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.gridspec import GridSpec
from point_figure import PointAndFigureChart


# Set some plotting defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# ## 1. Download Historical Data

# Define the ticker and date range
ticker = 'SPY'  # S&P 500 ETF
start_date = '2020-01-01'
end_date = '2023-12-31'

# Download data
data = yf.download(ticker, start=start_date, end=end_date)
print(f"Downloaded {len(data)} days of data for {ticker}")
#rename columns to remove substring in column names
data.rename(
    columns= lambda x:x.replace(ticker,'').strip().lower(),inplace= True
)
print(data.columns.to_list())
print(data.head())


# plt.figure(figsize=(14, 7))
# plt.plot(data.index, data['Close'])
# plt.title(f'{ticker} Price History')

# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.grid(True)
# plt.show()


# ## 2. Create a Point and Figure Chart

# Define parameters
box_size = 1.0  # $1 box size
reversal_size = 3  # 3-box reversal

# Create PnF chart
pnf = PointAndFigureChart(box_size=box_size, reversal_size=reversal_size)
pnf.build_chart(data)

fig = pnf.plot(figsize=(15, 10), show_signals=False)
plt.tight_layout()
plt.show()