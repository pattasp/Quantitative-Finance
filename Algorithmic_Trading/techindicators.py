# ======= Library :TECHNICAL INDICATORS FOR TRADING ============

#Again we will import some usefull libraries in order to construct the 
#Techincal Indicators we are about to use later on the ML project.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#=== Simple Moving Average(SMA) Indicator=========

# SMA_n(t) is a rolling window function of n time periods that 
# tracks the unweighthed arithmetic mean of the asset price over those 
# last n prices.

#So the inputs should be the Price which is in type of pd.Series & n int
#The output will also be pd.Series object

#================= NOTE ======================
# As a price when downloading the ticker get and as volume :
# price = df['Adj Close'] #extracts the df Adj Close column as a pd.Series!
# volume =df['Volume']


def sma(price : pd.Series , n: int) -> pd.Series:
    """
    The formula is written in LaTeX form :
    SMA_n(t) = (1/n) * sum_{i=0}^{n-1} P(t-i)
    """
    
    if n <= 0:
        raise ValueError("n of the time window must be positive.")
    
    return price.rolling(window = n).mean()

#====== Exponential Moving Average (EMA) ========

#EMA_n(t) is a recursion method that assign exponentially decreasing 
# weights to the older prices.

def ema(price: pd.Series, n:int) -> pd.Series :
    """
    EMA = a*P(t) + (1-a)*EMA_n(t-1)
    
    with a = 2/(n+1)
    
    The pandas library allows the recursion be using the :
        ewm(span =..., adjust = False)
    """
    
    if n <= 0:
        raise ValueError("n must be positive.")
    
    return price.ewm(span=n,adjust =False).mean()

#====== SIGNAL GENERATION FROM Moving Average (MA) Crossovers ========

#Based on those we can construst 2 states : Bullish | Bearish 

#We put the MA signals == time series windows into a unique function:
#and then we store those 2 states onto a DataFrame
    
def ma_signals_crossovers(ma_short : pd.Series,ma_long: pd.Series) -> pd.DataFrame :
    """
    SIGNAL GENERATION : Difference between MA_short and MA_long 
    Here we always have that n_short < n_long :
        
    1) Bullish Crossover (Golden Cross) is the state where :
    
    At time t -> if MA_short(t) > MA_long(t) & MA_short(t-1) <= MA_long(t-1)
    This signal denotes a potential upward momentum -> candidate for Long Entry
    
    2) Bearish Crossover(Death Cross) is the state where :
    
    At time t -> if MA_short(t) < MA_long(t) & MA_short(t-1) >= MA_long(t-1)
    Then we get a signal that refers to downward momentum -> candidate for Short Entry 
    OR LONG EXIT.
    """
    
    diff_t = ma_short - ma_long    
    
    # From pandas we can access the previous time step with the .shift(1) command 
    # Practicly we move our whole pd.Series 1 row down thus meaning we are accessing 
    # the previous time step. This helps us with obtaining the conditions needed to 
    # secure the Bullish | Bearish states.
    
    bullish_signal = (diff_t > 0) & (diff_t.shift(1) <= 0)
    bearish_signal = (diff_t < 0) & (diff_t.shift(1) >= 0)
    
    # Now we are returning the above feautures as DataFrames :
    # just to be more clean with the output.
    
    return pd.DataFrame({
        "Bullish Signal" : bullish_signal,
        "Bearish Signal" : bearish_signal 
        })

#======== Relative Strenght Index (RSI) ==================

# We know if RSI is a bounded momentum oscillator that mesaures the speed
# and the magnitude of recent price changes. So it is really imporant to keep 
# track on movements of the market !

# The RSI formula is also working with rolling averages based on the price changes.

def rsi(price : pd.Series , n :int) -> pd.Series :
    """
    Formula of the RSI(t) is :
        RSI(t) = 100 - (100/1+RS(t))
    with RS(t) being the relative strenght based on the average gain / average loss
    
    RS(t) = Avg_Gain(t) / Avg_Loss(t)    
    
    """
    # So we understand that before computing the RSI we need RS
    # and before RS we must firstly find the Avg Gain/Loss 
    
    """
    The formulas are give by our instructor and are the following :
        
        Avg_Gain = (1/n) * sum(G_i) with G_i = max(P(i)-P(i-1),0.0)
        Avg_Loss = (1/n) * sum(L_i) with L_i = max(P(i-1) -P(i),0.0)

    """
    
    if n <= 0:
        raise ValueError("n must be positive so the Avg Gain/Loss dont go to infty")
    
    # we see that there are time differences of the prices : pd.Series objects :
        
    price_diff = price.diff()
    
    # Because price_diff is pd.Series object we cant use the standard .max() function.
    # Instead we need to apply a clip or a np.maximum 
    gains  = np.maximum(price_diff,0)
    losses = np.maximum(-price_diff,0)

    avg_gain = gains.rolling(window = n).mean()
    avg_loss = losses.rolling(window = n).mean()
    
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1+RS))

    return RSI


#====== RSI values to Signals ==========

def rsi_signal(rsi_series : pd.Series, low: float =30.0 ,high: float =70.0) -> pd.Series: 
    """
    We put as input(formal) parameter in the function the rsi_series because:
        When we will apply the .rsi() function to a pd.Series object it will return 
        a pd.Series object !
        
        
    Core values for the RSI Signals Series are :
        
    1) RSI < 30.0 => oversold in the market => buy signal
    
    2) RSI > 30.0 => overbought in the market => sell signal
    
    Return the results as a DataFrame of Boolean Values 
    """
    
    if low >= high :
        raise ValueError("Low signal value cant be bigger than High.")
    
    # Compare the RSI_series obj. value with the thresholds of low,high !
    buy_signal = rsi_series < low
    sell_signal = rsi_series > high
    
    return pd.DataFrame({
        "RSI Buy" : buy_signal,
        "RSI Sell" : sell_signal
        })
    
def macd(price : pd.Series, short : int=12, long: int=26,signal :int=9) -> pd.Series :
    """
    macd stands for Moving Average Convergence Divergence and it represents
    a Momemntum Indicator that tracks the difference between 2 diff. periods
    of EMA of an asset. Mainly these 2 are the 12-period and 26-period:
        MACD(t) = EMA_12(t) -EMA_26(t)
    
    We obtain the trading signal by :        
    Signal(t) = EMA_9(MACD(t))
    """
    
    # Calculating the base EMA : short =12 period, long=26 period
    ema_short = ema(price,short)
    ema_long  = ema(price,long)
    
    # MACD Calculation:
    MACD = ema_short - ema_long
    
    # Calculate the Signal Line (9-period EMA of MACD)
    signal_line = ema(MACD, signal)
    
    
    # Return a DataFrame with both MACD and Signal Line
    return pd.DataFrame({
        "MACD": MACD,
        "Signal Line": signal_line
    })


def macd_signals(macd_df : pd.DataFrame) -> pd.DataFrame :
    """
    We are using a separate function to capture the desired signals
    for trading give by the MACD fucntion. It is easier to find the 
    MACD(t-1) when working with a DataFrame. We just need the shift(1) 
    command.
    
    So :
        Bullish : MACD(t) > Signal(t) && MACD(t-1) <= Signal(t-1)
        Bearish : MACD(t) < Signal(t) && MACD(t-1) >= Signal(t-1)
    """
    
    # We take each column from the MACD DataFrame returned before individually:
        
    macd_col = macd_df['MACD']
    signal_col = macd_df['Signal Line']
    
    # We generate the Bullish = Buy signal:
        
    bullish_signal = (macd_col > signal_col) & (macd_col.shift(1) <= signal_col.shift(1))
    bearish_signal = (macd_col < signal_col) & (macd_col.shift(1) >= signal_col.shift(1))
    
    return pd.DataFrame({
        "Bullish Signal": bullish_signal,
        "Bearish Signal": bearish_signal
        })

def boolinger_bands(price :pd.Series,n:int, k:float) -> pd.DataFrame :
    """
    The Bollinger Bands are a useful tool that provides a dynamic range for the 
    price movements based on its recent volatility.
    They consist of 3 bands ; Lower,Middle,Upper:
        Lower => The SMA minus k times the volatility σ_n(t)
        Middle => The SMA plus k times the >> σ_n(t)
        Upper => an SMA of the asset price.
    """
    #By the SMA function quick calculation of the Middle Band:
        
    middle = sma(price,n)
    
    # Define the volatility :
    sigma_n = price.rolling(window = n).std()
    
    # Now we just define the Upper and Lower bands:
        
    upper = middle + k*sigma_n
    lower = middle - k*sigma_n
    
    # Return the results into a DataFrame object : 
    return pd.DataFrame({
        "Lower Band":lower,
        "Middle Band": middle,
        "Upper Band": upper
        })

def obv(price :pd.Series,volume:pd.Series) -> pd.Series:
    """
    Now we define the On-Balance-Volume (OBV) indicator. It is a cumulative
    measure that uses trading volume to infer buying and selling pressure.
    It's formula is :
        OBV = OBV(t-1) + sgn(P(t)-P(t-1))*volume(t)
    """
    # We start by constructing step by step the function:
    
    # We need to calculate the price changess : P_t - P_t-1:
    price_change = price.diff()
    
    # Apply the sng function on that price change:
    
    sgn = price_change.apply(lambda x: 1 if x>0 else(-1 if x<0 else 0))
    
    # Multiply the sgn by the volume:
    
    obv_changes = sgn * volume
    
    # To take the OBV(t-1) + sgn... part we want to use cummulative sum:
    # So we have :
    obv_final = obv_changes.cumsum()
    
    return obv_final

