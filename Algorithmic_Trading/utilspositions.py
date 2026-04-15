# ======== Library: Classes of Long Positions =============

#======Long Position : Entry Constrains ==========
# importing some neccessary libraries to work with

import math

#Creating the Long Position Class Object 
class LongPosition():
    
#============ ENTRY CONSTRAINT DERIVATION PART================
    def __init__(self,capital,price,transaction_fee):
        
        
        """ Description of needed Interpetations for the class to be functional:
        q * P(t0) + f <= C(t0)
        q_max = floor((C(t0) - f) / P(t0))  (clipped at 0)
        q ∈ {0,1,...,q_max}
        """
       
        self.C_t0 = float(capital)
        self.P_t0 = float(price)
        self.f    = float(transaction_fee)
        
        if self.C_t0 < 0:
            raise ValueError("Initial Capital must be >=0")
        if self.P_t0 <= 0:
            raise ValueError("Price must be again positive")
        if self.f < 0:
            raise ValueError("Transaction fees need to exist")
        
    def q_maxShares(self):
        """
        Condition needed for maxShares = 
        q_maxlong = floor [ (C_t0 - f) / P_t0]
        
        A valid position of shares must be q = int(0,1,...q_maxlong)
        """
        if self.C_t0 < self.f :
            return 0
        else :
            return int(math.floor((self.C_t0-self.f)/self.P_t0))
        
# We need now to have a valid q = number of shares
    def valid_q(self,q):
        return isinstance(q,int) and 0 <= q <= self.q_maxShares()
    
    def entryCost(self,q):
        """
        Here we are setting up the Entry(opening) cost of our long position:
            
        cost to open = q * P_t0 +f    
        """
        # a small catching error if statement that checks that q is "active"
        
        if not self.valid_q(q):
            raise ValueError(f"Invalid q(shares).They must be an int in the domain of[0,{self.q_maxShares()}]")
        
        return q * self.P_t0 + self.f
    

#========== PROFIT DERIVATION PART=================

    def initialOutflow(self,q):
        """
        Here we need to define the Initial Capital Outfflow :
            = q * P_t0 +f which happens to be equal to the entryCost
        """
        return self.entryCost(q)
    
    def finalInflow(self,q,exit_price):
        
        """
        In this section we defined the function in order to represent 
        the cashflow we receive once the exit of our position occurs
        
        Final Inflow = q * P_t -f
        """
        
        if not self.valid_q(q):
            raise ValueError("Invalid q.")
        
        # We need to also have that the exit_price will always be positive
        if exit_price <= 0 :
            raise ValueError("Exit Price must be positive")
            
        return q * float(exit_price) - self.f
    
    def profit(self,q,exit_price):
        
        """
        Last but not least every investor who is trading stocks cares always 
        about the potential profit he will make
        
        Profit_long(t0,t) = Final Inflow - Initial Outflow
        """
        
        return self.finalInflow(q,exit_price) - self.initialOutflow(q)
    
# =========== SAME PATTERN FOR THE SHORT POSITION OF THE HOLDER=======


class ShortPosition():
    
    def __init__(self,capital,price,transaction_fee,lambda_):
        
     """ 
     Now in the Short Position the trader borrows q shares and then sells them 
     at the market price P_t0 and thus he is receiving back q*P_t0. 
     Then he must subtract from that amount the transaction fee f.
     So we have now that the Initial Inflow is := q*P_t0 - f.
     
     Important notice : To close the shorting position ,
     he needs to repurchase q shares. Lets say that the trader
     has enough Capital to repurchase these shares at a worst-case future price 
     : P_worst = lambda*P_t0 with lambda>1
     
     Thus the feasibility condition is : q*P_worst + f <= C_t0
     So solving for q we obtain :
         
         q_short_max = floor[(C_t0 - f)/P_worst]
     """
     
     self.C_t0 = float(capital)
     self.P_t0 = float(price)
     self.f = float(transaction_fee)
     self.lmbd = float(lambda_)
 
     if self.C_t0 < 0:
         raise ValueError("Initial Capital must be positive.")
         
     if self.P_t0 <= 0:
         raise ValueError("Price must be positive.")
    
     if self.f < 0:
         raise ValueError("Transcaction fee cannot be negative.")
     
     if self.lmbd <= 1:
         raise ValueError("Lambda must be > 1 as set by default.")
         
         
    def P_worst(self):
        
        return self.lmbd * self.P_t0 
    
    def q_short_maxShares(self):
        
        #Same logic as the Long Position is applied :
           
        if self.C_t0 <= self.f:
            return 0 
        # Initial Capital must always exceed the transaction fee
        
        return int(math.floor((self.C_t0 - self.f)/self.P_worst()))
    
    def valid_q(self,q):
        
        return isinstance(q,int) and 0 <= q <= self.q_short_maxShares()
    
# ======== SHORT POSITION -- PROFIT DERIVATION===========

    def initialInflow_short(self,q):
        
        if not self.valid_q(q):
            raise ValueError("Invalid q.")
        
        return q * self.P_t0 - self.f
    
    def finalOutflow_short(self,q,exit_price):
        """
        Final Outflow = q * P(t) + f
        (buy-to-cover q shares at price P(t), pay fee f)
        """
        if not self.valid_q(q):
            raise ValueError(f"Invalid q. Must be int in [0, {self.q_short_maxShares()}].")
        if exit_price <= 0:
            raise ValueError("exit_price must be positive")

        return q * float(exit_price) + self.f

    def profit(self, q, exit_price):
        """
        Profit_short(t0, t) = Initial Inflow - Final Outflow
                           = q*(P(t0) - P(t)) - 2f
        """
        return self.initialInflow_short(q) - self.finalOutflow_short(q, exit_price)        

    

