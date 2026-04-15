# =========== FUNDAMENTAL INDICATORS MODULE==================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Now we will create another module where we will have our fundamental
# indicators that we are going to use in our trading ML algorithm.

# We will be starting with the Price/Earnings ratio with a time window :

# It will thus be a Trailing P/E ratio :
    
def pe_ratio(price :pd.Series, eps_dates: pd.Series, eps_values: pd.Series) -> pd.Series:
    """
    We proceed now with calculating our P/E with the following formula:
        P/E = P(t)/E(t) where E(t) stands for the Earnings Per Share 
        Note : E(t) should always be at discrete time points T_E subset of T
    
    T is the whole {t_min...t_max} and denotes the total trading horizon.
    Earnings may be obtained from historical market data or predicted via time
    series models (e.g LSTM)
    
    Now for any time t belongs T,we must define t_last(t) = max(t belong T_eps)
    as our most recent earnings report date available at time t.
    So here we will need a mechanism that will 
    """
    
    # Sanity checking statements so our prices and eps are on the same 
    # time intervals and none is forwarded 
    #if len(price) != len(eps_dates):
        #raise ValueError("Price and EPS Date time series object must have same length")
    
    #if len(eps_dates) != len(eps_values):
        #raise ValueError("EPS Dates and EPS Values series must be of the same length")
        
    # A nice way to keep track of the most recent earnings is to sort data by date 
    # Thus making sure we always get the most recent EPS 
    
    # Convert the pd.Series into a DataFrame 
    eps_df = pd.DataFrame({"Date":eps_dates,"EPS":eps_values})
    
    # Sorting :
    eps_df = eps_df.sort_values(by="Date")
    
    # We need to take those 2 time series object and merge them.
    # They not always have the same datetime stamps :
    # Prices are daily while Earnings are on specific dates !
    
    # With plain merging we would have NaN values between pricing and EPS
    # So we can use the : merge_asof(direction = "backwards" )
    
    price_df = pd.DataFrame({"Date":price.index,"Price":price})
    
    merged_df = pd.merge_asof(price_df.sort_values("Date"),eps_df.sort_values("Date"))
    
    # Now we have a robust DataFrame that has a solid matching between Prices and EPS 
    
    # Calculate the P/E ratio
    
    p_e = merged_df["Price"]/merged_df["EPS"]
    
    return p_e 

def earnings_suprise(eps_actual : pd.Series, eps_expected: pd.Series) -> pd.Series:
    """
    We construct now another fundamental indicator named Earnings Suprise
    It quantifies the deviation between actual and expected earnings
    
    Suprise(t) = [E_actual(t) - E_expected(t)] / E_actual(t)
    where the E_expected is prediected by a Time Series Model Trained on Historical Earnings
    
    Practicly this means :
    E_exp(t) = function of (E(t-1),E(t-2),...E(t-k) ; Θ)
    
    k : lookback window 
    θ : theta is the model's learning parameters 
    f : forecasting model such as LSTM,Regression Tree etc.

    """
    
    if not eps_actual.index.equals(eps_expected.index):
        raise ValueError("Actual and expected EPS must have the same index")
        
    if(eps_expected ==0 ).any():
        raise ValueError("Expected EPS obtained by the model has zero values.Cannot divide with 0.")
        
    suprise = (eps_actual - eps_expected)/ eps_expected
    
    return suprise
