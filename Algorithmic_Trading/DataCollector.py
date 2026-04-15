#======== TIME SERIES DATA COLLECTOR MODULE ==================

# We install some required libraries to download the proper stock 
# market data

import os 
import datetime
import pandas as pd
import time 
import pickle 
import requests 

# We gather our data from online web scrapping using APIs
# We either go with TWELVE DATA or Yahoo Finance

# Note : When working with third party APIs there is a window limitation
# so we cannot use the API too many times. Thus there is a need for a function
# that will be checking the rate of our usage:
    

# ============ MAIN FUNCTIONS CORE =================== 

# The following function manages these constraints:
    

def api_limit_checker(request_timestamps,max_requests = 8,window_sec = 60):
    # The input formal parameters declare :
        # i) request_timestams : It's a list of time.time() timestamps
        # for prior requests 
        
        # ii) max_requests : the maximum allowed requests per time window o
        # of 60sec. The default is set at 8
        
        # iii) the size of the rolling window time be default is set to 60
        
    # Outputs :
        # pruned_timestamps : An updated list of timestamps, with the older than
        # 'window_sec' being erased 
        
        # used : how many requests have been made in the last "window_sec"
        # left : how many requests remain available for the user to capitalize on
        # into gaining data before running into the maximum
        
        # So we will need RunTime Errors to catch possible breaks 
        
        # Obtain the current time
        current = time.time()
        
        # Keep the Timestamps within the last "window_sec"
        
        pruned_timestamps = [t for t in request_timestamps if (current-t)< window_sec]
        
        used= len(pruned_timestamps)
        
        left= max_requests - used 
        
        # Catch Limit :
        
        if used >= max_requests:
            oldest_in_timewindow = pruned_timestamps[0] #1st index is the oldest request 
            
            # Set a waiting time window until you are able to retry requesting tickers from the API
            wait_time = int(window_sec -(current - oldest_in_timewindow))+1
            
            raise RuntimeError(f"Limit is exceed :{max_requests} in the last {window_sec} s \n"
                  f"Please wait: {wait_time} seconds before retrying.")
            
        return pruned_timestamps,used,left
    
# -----------------------------------------------------------------------------------------------------   
    
# Now we need another function. With it we will be able to return any historical data 
# we want from API for a given ticker/symbol and within our desired date range.
# They will be in the form of pandas DataFrame. Mainly to be more able for data handling & wrangling
# plus our utilities libraries are based on pd.Series & pd.DataFrames
     

#  API_KEY  put down your API Key. You find it on your Dashboard at the Twelve Data Web on your profile
#  click this url for more information : https://twelvedata.com/docs#overview

API_KEY = "b10e7ad39a0043c38dbc24d0b8f83bfe"

def get_api_data(ticker,start_time,end_time,interval = '1day',apikey = API_KEY,timeout_sec =30) -> pd.DataFrame :
    
    # Input Parameters 
    # Ticker of the desired stocks we want to grab.
    # start_time,end_time : Time Range . time_interval represents the band of each time.
    # Can be 1day==Daily, 1week = Weekly etc..
    
    # apikey = string of the API_key that we fetch the encapsulated web data
    
    # Outputs : A DataFrame i.e df that will hold valeus such as :
        # OPEN,HIGH,LOW,ADJ CLOSE(CLOSE),VOLUME
    # Note : We are working in the TimeSeries format so we need them to be DateIndexed.

    # Error Catchers :If expected output values are missing or if API is expired.

     
    # Sanity checker for invalid API Keys.
    if apikey is None or str(apikey).strip() == "":
        raise ValueError("The provided API Key {API_KEY} is invalid.")
     

   # Setting the base root of the TWELVE DATA API endpoint that handles our requests for historical data.
    base_url = "https://api.twelvedata.com/time_series"
    
    # We create a dictionary so we could navigate the data more quickly and avoid possible duplicates.
    # when applying the API Request.
    
    params = {
        "symbol": ticker,
        "start_date":start_time.strftime("%Y-%m-%d"), #we convert a datetime object to a formatted string
        "end_date":end_time.strftime("%Y-%m-%d"),
        "interval":interval,
        "apikey":apikey
        }
    
    # Here we make the API true request and then store it as "data" variable.
    # We used the timeout = timeout_sec to set up the maximum time in seconds to wait 
    # for the server tto respond.If that time exceeds the setup we will raise a runtime error.
    # If that happens just change the default waiting time for the server to respond.
    # Mainly used to see if our requests are really large or to always keep in mind that the server
    # might not be available at the moment.
    
    
    response = requests.get(base_url, params = params, timeout = timeout_sec)
    data = response.json()

    # DATA CHECKERS TO ENSURE THAT WE ARE NOT MISSING ANY OR THAT THE API IS RUNNING:
    
    if "status" in data and data["status"] == "error":
        msg = data.get("message","Unknown Error")
        raise ValueError(f"There is an error regarding the Twelve Data API for ticker:{ticker}:{msg}")
    
    # If everything so far went without errors parse the collected data into a pd.DataFrame:
    
    df = pd.DataFrame(data["values"])
    
    # Now we need to do a quick Clean and Data Convertion inside the DataFrame.
    
    # Again a Sanity Check to see if the "datetime" is in the df.columns :
    
    if "datetime" not in df.columns: 
        raise ValueError(f" Missing 'datetime' in response for the ticker: {ticker}.")
    
    # Convert any possible string columns to numerical ones to be able to do real maths:
    for col in ["open","high","low","close","volume"] :
        if col in df.columns:
            df[col]= pd.to_numeric(df[col],errors = "coerce")
    
    # Convert to the desired DateTimeIndex :
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(by = "datetime",inplace = True)
    df.set_index("datetime",inplace = True)
    
    return df

# ---------------------------------------------------------------------------------------------------------------------

# Due to the fact that every other technical library created works with pd.Series
# Here there is an added function that will Standardize prehand the data to ensure
# that the columns : Open High Low Close, Adj Close),Volume -> "OHLCV" will follow
# from the TWELVE DATA API the preinstalled libraries format:
    

def standardize_ohlcv_columns(df):
    """ 
    In this function the following procedure is applied:
        
    Convert TwelveData output columns to your project standard:
    Open, High, Low, Close, Volume, Adj Close (fallback = Close)
    
    We need this to ensure that our data will be in the proper format 
    So our files wont "break" the pipeline
    """  
    out = df.copy() # We take as the variable out a copy of our df 
    # so we wont temper with the raw one. Mainly backup mode.
     
    rename_map = {
        "open":"Open",
        "high": "High",
        "low": "Low",
        "close":"Close",
        "volume":"Volume"
        }
    
    out = out.rename(columns={k : v for k,v in rename_map.items() if k in out.columns})
    
    # Enusre now the required OHLC:
    
    necessary = {"Open","High","Low","Close"} # These are the NEEDED columns we will work on.
    
    # find if any one is missing by checking the diff of the necessary - iteration of out.columns
    
    missing = necessary - set(out.columns)  # set is a quick function that iterates all the columns in out
    
    if missing: # if missing is True 
       raise ValueError(f"Missing the necessary OHLC columsn after the standardization :{missing}")

    
    # Check now the Volume Columns. It's important added feauture inserted on the OBV Technical Indicator:
    if "Volume" not in out.columns :
        # Set it as 0 so there is a later on functionality:
        out["Volume"] = 0.0 
    
    
    # Provide the Adj Close as Close :
    #Note to msfl : Recheck this Close if it stands for Adj Close. Really important to work with Adj Close
    # because it encapsulates all the available info about the corporations actions.

    if "Adj Close" not in out.columns:
        # Just set the Close 
        out["Adj Close"] = out["Close"]
    
    # If not Brute Force the DateTimeIndex: Important aspect when working with Time Series Objects :
    
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
        
    # Resort the index :
    out = out.sort_index()
    
    return out
  

# We constructed our api getter function hence,
# we need to create a function that properly downloads the dataset

def download_dataset(td_symbol_groups,start_date,end_date,interval="1day",apikey=API_KEY,max_requests=8,window_sec=60,
                     sleep_sec = 0.0, verbose =True):
    
    # sleep_sec is the desired time delay we might wish when downlaoding our data.
    # verbose is a keyword argument that helps us control printing/logging to the console 
    # we used it in case we might have a vast amount of symbols/tickers so we keep a better 
    # track on which symbol is being download | how many rows | whether API errors occur |
    # how close we are to the limit.
    
    # Mainly this function is a barrier to stop any further downloads 
    # until the user waits.
    
    # The main approach is try| except:
    
    collected_data = {}
    request_timestamps = []
        
    for group_name, symbols in td_symbol_groups.items():
        if verbose:
            print(f"\n --- Download the group of tickers :{group_name}----")
        
        group_data = {}
            
        for symbol in symbols:
            # Prior is to check limits first:
            request_timestamps,used,left = api_limit_checker(
                    request_timestamps,max_requests=max_requests,window_sec=window_sec)
                
            if verbose:
                print(f" Request usage in last 60s: Used ={used},Left = {left}")
                
                # Grab the data :
                try :
                # We firstly get the Raw Twelve Data 
                        raw = get_api_data(symbol,start_time = start_date,end_time = end_date,interval = interval,apikey=apikey)
                        
                # Now we standardize them to our preset:
                        df =  standardize_ohlcv_columns(raw)
                        group_data[symbol] = df
                     
                        if verbose :
                            print(f"Downloaded {symbol},rows = {len(df)}")
                            
                except Exception as ex:
                    if verbose :
                        print(f"Error downloading {symbol}:{ex}")
                    
                request_timestamps.append(time.time())
                
                collected_data[group_name] = group_data
                
    if verbose:
        print("\n All Downloads are finished.\n")
        
    return collected_data


# ====================== DOWNLOADING THE DESIRED DATASET ===========================

# Set the time range we wish our historical data to be from
start_date  = datetime.datetime(2015,1,1)
end_date = datetime.datetime(2026,1,1)
interval = '1day'

# Defining the symbol groups to be downloaded straight from the Twelve Data website

td_symbol_groups = {
    "stocks":[
        "INTC", # Intel Corporation Stock
        "AMD"  # Advanced Micro Devices 
        ],
    "currencies": [
        "USD/EUR", # USD vs EUR
        "JPY/USD"  #JPY vs USD
    ],
    "indices": [
        "SPY",     # S&P 500 ETF
        "DIA",     # Dow Jones ETF
        "NADAQ",   # NASDAQ ETF
    ]
}
    
# Now we will perfom the actual downloading:

collected_data = download_dataset(td_symbol_groups,start_date,end_date,interval,apikey=API_KEY)


# ================================ CACHING (PICKLE) =================================================


def save_collected_data(collected_data,directory = "./data",filename = "collected_data.pkl"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory,filename)
    with open(path,"wb") as f:
        pickle.dump(collected_data,f)
    return path

def load_collected_data(directory="./data", filename="collected_data.pkl"):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached dataset found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)     
    
# Ensuring the Data are being stored in the Correct File:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")    
    
# Save the downloaded dataset into a pickle file so DataPreparator can load it later
saved_path = save_collected_data(collected_data, directory="./data", filename="collected_data.pkl")
print(f"Saved collected dataset to: {saved_path}")