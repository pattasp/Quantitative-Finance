# ================= LSTM MODEL FOR STOCK PREDICTION ===========================

import os 
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F


# -----------------------------------------------
# Now I will be importing the necessary modules from
# DataCollector and DataPreparator 

from DataCollector import load_collected_data  
from DataPreparator import build_combined_price_df

# -------------------------------------------------
# Importing my technical indicators
import techindicators as ti
from utilspositions import LongPosition,ShortPosition


# Most important part when working with PyTorch is to always be sure
# about our Tensors shapes in order for the operations to be fully 
# functional. So we need to always Standardize our Input Feautures.

#--------------------------------------------------------------------

# Prior we can first examine the working device:

# If someone works either on cpu or gpu(CUDA) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:",device)
print()

# Additional info if using cuda:
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    
# EXTRA ------------ The below part is for Apple Macbook Users !!! Uncomment 
# if you work in SILICON GPU

#================== Check for APPLE SILICON GPU (M1/M2/M3) ====================

# torch.backends.mps.is_available() # false if you are not running on mac

# If true set the device type:
# device = "mps" if torch.backends.is_available() else "cpu"

#if torch.cuda.is_available():
    #device = "cuda" # Use NVIDIA GPU (if available)
#elif torch.backends.mps.is_available():
    #device = "mps" # Use Apple Silicon GPU (if available)
#else:
    #device = "cpu" # Default to CPU if no GPU is available

#=========================================================


# Device
def get_execution_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_execution_device()
print("Using device:", device)


# TradingDataset :
    
class TradingDataset(Dataset):
    def __init__(self, X, indicator_bank, returns):
        """
        X: torch.FloatTensor (N, seq_len, d_input)
        indicator_bank: list of torch.FloatTensor, each (N, M)
        returns: torch.FloatTensor (N,)
        """
        self.X = X
        self.indicator_bank = indicator_bank
        self.returns = returns

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        indicators = [bank[idx] for bank in self.indicator_bank]
        y = self.returns[idx]
        return x, indicators, y
    

    
# Metrics == Losses in Tensor format which are ensured to be 
# differentiable for the gradient descent procedure. 


def sharpe_loss(returns: torch.Tensor) -> torch.Tensor:
    mean = returns.mean()
    std = returns.std(unbiased=False) + 1e-6
    return -mean / std


def return_loss(returns: torch.Tensor) -> torch.Tensor:
    return -returns.sum()


def hybrid_loss(returns: torch.Tensor, mu_sharpe: float = 0.5, mu_ret: float = 0.5) -> torch.Tensor:
    return mu_sharpe * sharpe_loss(returns) + mu_ret * return_loss(returns)


# Hybrid loss with the regularization term.
def hybrid_loss_with_reg(
    strat_returns: torch.Tensor,
    model: nn.Module,
    mu_sharpe: float = 0.5,
    mu_ret: float = 0.5,
    mu_reg: float = 1e-4,
) -> torch.Tensor:
    base = mu_sharpe * sharpe_loss(strat_returns) + mu_ret * return_loss(strat_returns)
    reg = torch.zeros((), device=strat_returns.device)
    for p in model.parameters():
        reg = reg + (p ** 2).sum()
    return base + mu_reg * reg


# Differntiable portfolio simulator returns function 

def differentiable_portfolio_returns(
    D: torch.Tensor,
    asset_returns: torch.Tensor,
    cost: float = 0.0005,
    kappa: float = 5.0,
    already_position : bool = False,
):
    """
    Differentiable 1-asset portfolio model.

    D: (T,) or (B,) continuous decision in [-1,1] (from tanh)
    asset_returns: (T,) or (B,) next-step returns of the asset

    We convert decisions to a smooth position weight w_t in [-1,1],
    then apply transaction costs on turnover |w_t - w_{t-1}|.
    
    Returns:
      strat_returns: (T,) differentiable strategy returns
      w: (T,) differentiable position weights
    """
    # Smooth position mapping (still differentiable)
    
    # Bypassing tanh when D is already a position
    
    if already_position:
        w = D.clamp(-1.0,1.0)
    else:
        w = torch.tanh(kappa*D) # in [-1,1]
    
    # Turnover / trading cost (first step has no previous position)
    w_prev = torch.cat([torch.zeros_like(w[:1]), w[:-1]], dim=0)
    turnover = torch.abs(w - w_prev)
    
    strat_returns = w * asset_returns - cost * turnover
    return strat_returns, w


def equity_curve_from_returns(strat_returns: torch.Tensor, C0: float = 1.0):
    """
    Differentiable equity curve:
      C_t = C0 * Π_{i<=t} (1 + r_i)
    """
    return C0 * torch.cumprod(1.0 + strat_returns, dim=0)


# LSTM Trading Agent

class LSTMTradingAgent(nn.Module):
    """
    Shared LSTM encoder -> context h_t -> per-indicator heads:
    - window logits -> Gumbel-Softmax weights -> indicator_hat
    - thresholds theta_plus, theta_minus
    - signal S_i(t)
    Aggregate: D(t) = tanh( sum_i beta_i S_i(t) + b )

    forward(..., return_details=True) returns extra tensors needed for deliverable plots:
    weights, thresholds, indicator_hat, signals, S, D
    """
    def __init__(self, d_input, d_hidden, num_indicators, window_sizes, tau=1.0, eps_sigmoid=0.1):
       
        super().__init__()
        self.num_indicators = num_indicators
        self.window_sizes = window_sizes
        self.M = len(window_sizes)
        self.tau = float(tau)
        self.eps_sigmoid = float(eps_sigmoid)

        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)
        self.window_heads = nn.ModuleList([nn.Linear(d_hidden, self.M) for _ in range(num_indicators)])
        self.threshold_heads = nn.ModuleList([nn.Linear(d_hidden, 2) for _ in range(num_indicators)])

        self.beta = nn.Parameter(torch.randn(num_indicators))
        self.bias = nn.Parameter(torch.zeros(1))
        

    def forward(self, x, indicator_bank, return_details: bool = False):
        _, (h_n, _) = self.lstm(x)
        h_t = h_n[-1]

        signals = []
        details = {
            "weights": [],
            "theta_plus": [],
            "theta_minus": [],
            "indicator_hat": [],
            "signal": [],
        }

        for i in range(self.num_indicators):
            alpha_logits = self.window_heads[i](h_t)

            
            # Gumbel Softmax noise must be applied on training set
            # due to its stohastic nature
            
            if self.training:
                u = torch.rand_like(alpha_logits).clamp(1e-6,1-1e-6)
                gumbel = -torch.log(-torch.log(u))
                logits = alpha_logits + gumbel
            else:
                logits = alpha_logits  #deterministc at eval/test
            
            w = torch.softmax(logits / self.tau, dim=-1)

            I_i = indicator_bank[i] #shape of [B,M]
            I_hat = (w * I_i).sum(dim=-1) # shape of [B]
            
            # Enforced order on theta- <= theta+ to avoid overtrading

            th = self.threshold_heads[i](h_t)
            
            center = th[:, 0]
            width = F.softplus(th[:,1]) + 1e-6 # always positive
            
            theta_minus = center - 0.5*width
            theta_plus  = center + 0.5*width
            
            # Signal generation function 
            s_i = torch.sigmoid((I_hat - theta_plus) / self.eps_sigmoid) - \
                  torch.sigmoid((theta_minus - I_hat) / self.eps_sigmoid)

            signals.append(s_i)

            if return_details:
                details["weights"].append(w)
                details["theta_plus"].append(theta_plus)
                details["theta_minus"].append(theta_minus)
                details["indicator_hat"].append(I_hat)
                details["signal"].append(s_i)
                
        # Global Signal Aggregations 
        S = torch.stack(signals, dim=-1) #[B,num_indicators]
        
        D = torch.tanh(S @ self.beta + self.bias)  # [B]
        # raw decision score. Avoid double 
        # clampping by tanh function !

        if return_details:
            details["S"] = S
            details["D"] = D
            return D, details
        return D
                
# ----------------------------------------------------------------
# Building X,indicator_bank,y from combined_df
# ----------------------------------------------------------------

# Helper functions ====================================================

def _standardize_2d_train_only(arr: np.ndarray, train_T: int, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize columns of (T, F) array using stats computed on first train_T rows.
    """
    mu = arr[:train_T].mean(axis=0, keepdims=True)
    sd = arr[:train_T].std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (arr - mu) / sd

def _ensure_datetime_index(df:pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    if "Date" in df.columns:
        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    
    else :
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
    return df
#================================================================================

# CORE PART : PyTorch == Tensor but we have DataFrame from combined_df
# So we need to convert the df to torch format


def build_tensors_from_combined_df(
        combined_df : pd.DataFrame= build_combined_price_df,
        target_col : str = "INTC", # you can switch it any time to your preference
        seq_len : int = 30,
        window_sizes =(5,10,15,20,30),
        scale_X = True,
        scale_indicators = True,
        train_split = 0.8,
) :
    """
    Creates:
      X : sequences of raw features (here: multivariate returns)  (N, seq_len, d_input)
      y : next-day return of target_col                          (N,)
      indicator_bank : list length N_ind, each (N, M)
    Notes:
      - X contains raw market features only (prices/returns/volumes). Here we use returns.
      - Indicators are computed separately on price_target and stacked into indicator_bank.
      - Rolling indicators create NaNs; we drop any row where ANY indicator has NaN.
    """
    
    df = _ensure_datetime_index(combined_df)
    
    df = df.apply(pd.to_numeric,errors="coerce").dropna()
    df = df.sort_index()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column : {target_col} not in df columns: : {list(df.columns)}")
    
    
    # Raw market features for LSTM: multivariate returns (acceptable per spec)
    
    rets = df.pct_change().dropna()
    feature_cols = list(rets.columns)
    d_input  = len(feature_cols)
    
    # Precomputing indicators    
    # Indicator bank from the Target Price series (NOT RETURNS)
    
    price_target = pd.to_numeric(df[target_col],errors='coerce').astype(float)
    
    # PART OF CHOOSING OUR INDICATORS FROM ti Module 
    
    ws = list(window_sizes)
    ind_list = []
    
    # Indicator 0 : RSI(window)
    rsi_cols = []
    for w in ws:
        
        rsi = ti.rsi(price_target,n=int(w)).reindex(df.index) # aling on price index
        rsi_cols.append(rsi)
    
    ind_list.append(pd.concat(rsi_cols,axis=1))
    
    #Indicator 1: SMA(window)(use price)
    
    sma_cols = []
    for w in ws:
        
        sma = ti.sma(price_target,n=int(w)).reindex(df.index)
        
        sma_cols.append(sma)
    ind_list.append(pd.concat(sma_cols,axis=1))
    
    # Indicator 3 : EMA(window)(use price)
    ema_cols = []
    for w in ws:
        
        ema = ti.ema(price_target,n=int(w)).reindex(df.index)
        
        ema_cols.append(ema)
    ind_list.append(pd.concat(ema_cols,axis=1))
    
    # Indicator 3: Bollinger bandwidth(window) = (upper-lower)/mid
    bb_cols = []
    for w in ws:
        
        bb = ti.boolinger_bands(price_target, n=int(w), k=2)

        
        mid = pd.to_numeric(bb["Middle Band"],errors='coerce')
        upper = pd.to_numeric(bb["Upper Band"],errors='coerce')
        lower = pd.to_numeric(bb["Lower Band"],errors='coerce')

        bw = (upper - lower) / (mid.replace(0, np.nan))
        
        bw = bw.replace([np.inf,-np.inf],np.nan)
        bw = bw.clip(lower=0, upper=bw.quantile(0.995))  # simple outlier cap
        
        bb_cols.append(bw.reindex(df.index))
    ind_list.append(pd.concat(bb_cols, axis=1))
    
#  Now we need to stack these indicators
#  Each one is (T_price,M).Convert to returns index to align with X/y timing.
   
    ind_list = [ind.reindex(rets.index).astype(np.float32) for ind in ind_list]

   
    # Drop any row where indicaotr is NaN (NaN can be caused by rolling windows)

    mask = np.ones(len(rets), dtype=bool)
    for ind in ind_list:
        mask &= ~ind.isna().any(axis=1).to_numpy()
        
    # Apply mask ONCE (positional)
    
    rets = rets.iloc[mask]
    ind_list = [ind.iloc[mask] for ind in ind_list]
    
    # Target y: next-day returns of target col (aligned with filtered rets)
    y_all = rets[target_col].values.astype(np.float32)

    # Optional scale X and indicators using Train-only data
    T_total = len(rets)
    train_T = int(train_split * T_total)
    
    X_all = rets.values.astype(np.float32)  # (T, d_input)
    
    if scale_X:
        X_all = _standardize_2d_train_only(X_all,train_T)
        
    if scale_indicators:
        scaled = []
        for ind in ind_list:
            arr = ind.values.astype(np.float32)  # (T, M)
            arr = _standardize_2d_train_only(arr, train_T)
            scaled.append(pd.DataFrame(arr, index=ind.index))
        ind_list = scaled

    # Build sequences + aligned indicator bank at decision time t
    X_seq = []
    y_seq = []
    ind_banks = [ [] for _ in range(len(ind_list)) ]

    # We predict next-day return, so sample at time t uses seq ending at t, target = return at t+1
    max_w = max(window_sizes)
    start_t = max(seq_len,max_w)
    
    for t in range(start_t, T_total - 1):
        X_seq.append(X_all[t - seq_len:t, :])           # (seq_len, d_input)
        y_seq.append(y_all[t + 1])                      # next step

        for i, ind in enumerate(ind_list):
            ind_banks[i].append(ind.iloc[t].values)     # (M,)

    X = torch.tensor(np.stack(X_seq), dtype=torch.float32)
    y = torch.tensor(np.array(y_seq), dtype=torch.float32)
    
    indicator_bank = [torch.tensor(np.stack(v), dtype=torch.float32) for v in ind_banks]

    # Return the filtered price series for the portfolio simulation. Useful for discrete 
    # capital update plots in LSTMResutls module
    
    # Dates aligned to each supervised sample (matches X/y length)
    sample_dates = rets.index[(seq_len + 1):]  # indices t+1 for t=seq_len..T_total-2

    # Price aligned to the same sample dates (used for evaluation plots)
    price_filtered = df[target_col].reindex(sample_dates).astype(float).values

    return X, indicator_bank, y, d_input, len(ind_list), price_filtered, sample_dates

# Addition : train on the full time-order series(big contiguous chuncks) in order 
# for the w_prev to be consistent accross the whole trajectory path.This might 
# cause overloading.Prefer it if GPU is available. In my setup PyTorch wont 
# recognize AMD Radeon Graphics GPU.

def flatten_loader_in_order(loader,assume_ordered=True):
    """
    Concatenate all batches from a shuffle=False DataLoader into one long series.
    Returns:
      D_input_X: (N, seq_len, d_input)
      indicator_bank_list: list of (N, M)
      y: (N,)
    """
    
    if assume_ordered is not True:
        raise ValueError("flatten_loader_in_order requires shuffle = False DataLoader")
    
    Xs, ys = [], []
    inds = None

    for x, ind_bank, y in loader:
        Xs.append(x)
        ys.append(y)
        if inds is None:
            inds = [[] for _ in range(len(ind_bank))]
        for i, t in enumerate(ind_bank):
            inds[i].append(t)

    X = torch.cat(Xs, dim=0) # This creates a huge graph + tensors.In my PC 
    # it breaks. I hope in yours not !:) 
    y = torch.cat(ys, dim=0)
    ind_cat = [torch.cat(v, dim=0) for v in inds]
    return X, ind_cat, y



# ============================================================================
# 6) Train / Eval
# ============================================================================
def train_one_epoch_hybrid(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cost: float = 5e-4,
    kappa: float = 5.0,
    mu_sharpe: float = 0.5,
    mu_ret: float = 0.5,
    mu_reg: float = 1e-4,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
):
    """
    Kernel-safe training:
    - batchwise forward/backward (no flattening)
    - I tried with a continuous flattened torch but my kernel died.
    - So here I try batch sized approaches on both train/eval
    - preserves turnover continuity across batches via prev_pos
    - computes hybrid loss over the whole epoch returns WITHOUT storing the whole graph

    Trick:
    - we accumulate a *detached* list of returns for metrics
    - for loss we do a running mean/std approximation, OR (simpler) compute loss per batch.
    
    Here we do a stable compromise:
    - optimize per-batch hybrid loss (approx of epoch loss)
    - report full-epoch metrics by concatenating detached returns
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prev_pos = None
    all_returns_detached = []

    # For reporting (detached)
    total_loss_val = 0.0
    n_batches = 0

    for b, (Xb, ind_banks, yb) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        Xb = Xb.to(device)
        ind_banks = [t.to(device) for t in ind_banks]
        yb = yb.to(device).reshape(-1)  # (B,)

        D = model(Xb, ind_banks).reshape(-1)  # (B,)
        pos = torch.tanh(kappa * D)          # smooth position in [-1,1]

        # turnover continuity across batches
        if prev_pos is None:
            prev_batch = torch.cat([torch.zeros_like(pos[:1]), pos[:-1]], dim=0)
        else:
            prev_batch = torch.cat([prev_pos, pos[:-1]], dim=0)

        turnover = torch.abs(pos - prev_batch)
        strat_returns = pos * yb - cost * turnover  # (B,)

        # --- batchwise hybrid loss (does not blow up memory) ---
        loss = hybrid_loss_with_reg(
            strat_returns,
            model=model,
            mu_sharpe=mu_sharpe,
            mu_ret=mu_ret,
            mu_reg=mu_reg,
        )

        (loss / grad_accum_steps).backward()

        if (b + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # update prev_pos (keep graph-free)
        prev_pos = pos[-1:].detach()

        # detached returns for epoch metrics
        all_returns_detached.append(strat_returns.detach().cpu())

        total_loss_val += float(loss.detach().cpu().item())
        n_batches += 1

    # in case batches not divisible by grad_accum_steps
    if n_batches % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    epoch_returns = torch.cat(all_returns_detached, dim=0)  # CPU tensor
    mean = float(epoch_returns.mean().item())
    std = float(epoch_returns.std(unbiased=False).item())
    sharpe = 0.0 if std < 1e-12 else mean / std
    total_return = float(epoch_returns.sum().item())

    avg_loss = total_loss_val / max(n_batches, 1)

    metrics = {
        "loss": avg_loss,
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "total_return": total_return,
    }
    return metrics

@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cost: float = 5e-4,
    kappa: float = 5.0,
):
    model.eval()
    prev_pos = None
    all_returns = []

    for Xb, ind_banks, yb in loader:
        Xb = Xb.to(device)
        ind_banks = [t.to(device) for t in ind_banks]
        yb = yb.to(device).reshape(-1)

        D = model(Xb, ind_banks).reshape(-1)
        pos = torch.tanh(kappa * D)

        if prev_pos is None:
            prev_batch = torch.cat([torch.zeros_like(pos[:1]), pos[:-1]], dim=0)
        else:
            prev_batch = torch.cat([prev_pos, pos[:-1]], dim=0)

        turnover = torch.abs(pos - prev_batch)
        strat_returns = pos * yb - cost * turnover

        prev_pos = pos[-1:].detach()
        all_returns.append(strat_returns.detach().cpu())

    rets = torch.cat(all_returns, dim=0)
    mean = float(rets.mean().item())
    std = float(rets.std(unbiased=False).item())
    sharpe = 0.0 if std < 1e-12 else mean / std
    total_return = float(rets.sum().item())

    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "total_return": total_return,
    }

@torch.no_grad()
def get_series_on_loader(
    model,
    loader,
    device,
    delta: float | None = None,
    cost: float = 5e-4,
    kappa: float = 5.0,
    return_actions: bool = True,
    return_asset: bool = False,
):
    model.eval()

    decisions_cpu = []
    strat_cpu = []
    asset_cpu = [] if return_asset else None

    prev_pos = None

    for Xb, ind_banks, yb in loader:
        Xb = Xb.to(device)
        ind_banks = [b.to(device) for b in ind_banks]
        yb = yb.to(device).reshape(-1)

        D = model(Xb, ind_banks).reshape(-1)

        if delta is not None:
            pos = torch.where(
                D > delta, torch.ones_like(D),
                torch.where(D < -delta, -torch.ones_like(D), torch.zeros_like(D))
            )
        else:
            pos = torch.tanh(kappa * D)

        if prev_pos is None:
            prev_batch = torch.cat([torch.zeros_like(pos[:1]), pos[:-1]], dim=0)
        else:
            prev_batch = torch.cat([prev_pos, pos[:-1]], dim=0)

        turnover = (pos - prev_batch).abs()
        strat = pos * yb - cost * turnover

        prev_pos = pos[-1:].detach()

        strat_cpu.append(strat.detach().cpu())
        if return_actions:
            decisions_cpu.append(pos.detach().cpu())
        if return_asset:
            asset_cpu.append(yb.detach().cpu())

    strat_all = torch.cat(strat_cpu).numpy()

    decisions_all = torch.cat(decisions_cpu).numpy() if return_actions else None
    asset_all = torch.cat(asset_cpu).numpy() if return_asset else None

    return decisions_all, strat_all, asset_all


@torch.no_grad()
def collect_diagnostics(model, loader, device, window_sizes):
    
    """
    The following function collects deliverables: 
        Learned window choices and thresholds over time on a loader.
    It returns:
      D_series: (N,)
      lambda_series: list per indicator, each (N,) chosen window size
      th_plus_series: list per indicator, each (N,)
      th_minus_series: list per indicator, each (N,)
    """

    model.eval()
    ws = torch.tensor(window_sizes, dtype=torch.long)

    all_D = []
    all_lambda = None
    all_thp = None
    all_thm = None

    for X, ind_bank, _ in loader:
        X = X.to(device)
        ind_bank = [b.to(device) for b in ind_bank]

        D, details = model(X, ind_bank, return_details=True)

        all_D.append(details["D"].detach().cpu())

        if all_lambda is None:
            num_ind = len(details["weights"])
            all_lambda = [[] for _ in range(num_ind)]
            all_thp = [[] for _ in range(num_ind)]
            all_thm = [[] for _ in range(num_ind)]

        for i in range(len(details["weights"])):
            w = details["weights"][i].detach().cpu()  # (B, M)
            m_idx = w.argmax(dim=1)                   # (B,)
            lam = ws[m_idx].numpy()                   # (B,)

            all_lambda[i].append(torch.from_numpy(lam))
            all_thp[i].append(details["theta_plus"][i].detach().cpu())
            all_thm[i].append(details["theta_minus"][i].detach().cpu())

    D_series = torch.cat(all_D).numpy().reshape(-1)
    lambda_series = [torch.cat(v).numpy().reshape(-1) for v in all_lambda]
    thp_series = [torch.cat(v).numpy().reshape(-1) for v in all_thp]
    thm_series = [torch.cat(v).numpy().reshape(-1) for v in all_thm]

    return D_series, lambda_series, thp_series, thm_series


# =============================================================================
# Discrete Trade Execution + Capital Update per trading actions on the LSTM.
# =============================================================================
def simulate_portfolio_discrete(prices, decisions, delta=0.2, C0=1000.0, fee=1.0, lambda_short=1.5):
    """
    Single-asset discrete trading:
      D(t) >  delta -> long
      D(t) < -delta -> short
      else          -> flat

    Uses LongPosition / ShortPosition primitives for feasibility and cashflows.
    Returns equity curve C(t) length T.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    decisions = np.asarray(decisions, dtype=float).reshape(-1)

    m = min(len(prices), len(decisions))
    prices = prices[:m]
    decisions = decisions[:m]
    T = m

    C = np.zeros(T, dtype=float)
    C[0] = C0

    pos = 0       # -1 short, 0 flat, +1 long
    q = 0.0
    entry_price = None
    

    for t in range(T - 1):
        P = float(prices[t])
        P_next = float(prices[t + 1])

        # Safety guard against bad data
        if (not np.isfinite(P)) or (not np.isfinite(P_next)) or (P <= 0) or (P_next <= 0):
            C[t + 1] = C[t]
            continue

        D = float(decisions[t])

        desired = 0
        if D > delta:
            desired = 1
        elif D < -delta:
            desired = -1

        # Close if switching or going flat
        if desired != pos and pos != 0:
            if pos == 1:
                lp = LongPosition(C[t], entry_price, fee)
                C[t] = C[t] + lp.finalInflow(q, P)
            elif pos == -1:
                sp = ShortPosition(C[t], entry_price, fee, lambda_short)
                C[t] = C[t] + sp.finalInflow_short(q, P, lambda_short)
            pos = 0
            q = 0.0
            entry_price = None

        # Open new position if flat
        if pos == 0 and desired != 0:
            if desired == 1:
                lp = LongPosition(C[t], P, fee)
                q = lp.q_maxShares()
                if q > 0:
                    C[t] = C[t] - lp.entryCost(q)
                    pos = 1
                    entry_price = P
            elif desired == -1:
                sp = ShortPosition(C[t], P, fee, lambda_short)
                q = sp.q_short_maxShares()
                if q > 0:
                    C[t] = C[t] + sp.initialInflow_short(q)
                    pos = -1
                    entry_price = P

        # Mark-to-market next step
        C[t + 1] = C[t]
        if pos == 1:
            C[t + 1] += q * (P_next - P)
        elif pos == -1:
            C[t + 1] += q * (P - P_next)

    return C


# =============================================================================
# Optional: quick local test (won't run unless you call it)
# =============================================================================
if __name__ == "__main__":
    print("LSTM_Base loaded. Device:", device)
    print("Tip: Use LSTM_Results.py to run training/plots cleanly.")