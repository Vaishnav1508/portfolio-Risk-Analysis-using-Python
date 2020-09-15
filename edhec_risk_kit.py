import pandas as pd
import numpy as np

def drawdown(return_series:pd.Series):
    '''Takes a time series of asset return
    computers & returns a dataframe that contails:
    The wealth peaks,the previous peaks,percent drawdowns '''
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks= wealth_index.cummax()
    drawdowns =(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({"wealth": wealth_index,
                         "Peaks": previous_peaks,
                         "Drawdown": drawdowns
                         })
def get_ffme_returns():
    '''load fama french dataset for the returns of the top and bottom deciles by market cap'''
    m = pd.read_csv (r'C:\Users\Vaishnav\Documents\investment Management using python\data\Portfolios_Formed_on_ME_monthly_EW.csv' , header = 0,                              index_col =0, parse_dates=True, na_values=-99.99)
    rets = m[['Lo 30','Hi 30']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m").to_period('M')
    return rets 
    
def get_hfi_returns():
    '''load fama french dataset for the returns of the top and bottom deciles by market cap'''
    hfi = pd.read_csv (r'C:\Users\Vaishnav\Documents\investment Management using python\data\edhec-hedgefundindices.csv' , header = 0,                              index_col =0, parse_dates=True, na_values=-99.99)

    hfi=hfi/100
    hfi.index=pd.to_datetime(hfi.index,format="%Y%m").to_period('M')
    return hfi

def get_ind_returns():
    ind = pd.read_csv(r'\Users\Vaishnav\Documents\investment Management using python\data\ind30_m_vw_rets.csv', header =0, index_col=0,                                      parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    ind_return = get_ind_returns()
    ind_size = get_ind_size()
    ind_nfirms= get_ind_nfirms()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight= ind_mktcap.divide(total_mktcap,axis="rows")
    total_market_return= (ind_capweight*ind_return).sum("columns")
    return total_market_return
    
    
    
    
    
def get_ind_size():
    ind = pd.read_csv(r'\Users\Vaishnav\Documents\investment Management using python\data\ind30_m_size.csv', header =0, index_col=0,                                      parse_dates=True)
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind
def get_ind_nfirms():
    ind = pd.read_csv(r'\Users\Vaishnav\Documents\investment Management using python\data\ind30_m_nfirms.csv', header =0, index_col=0,                                      parse_dates=True)
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind



def skewness(r):
    '''alternate to scipy.stats.skew() 
    computes the skewness 
    returns the series'''
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    sk = exp/sigma_r**3
    return sk

def kurtosis(r):
    '''alternate to scipy.stats.skew() 
    computes the skewness 
    returns the series'''
    demeaned_rk = r - r.mean()
    sigma_rk = r.std(ddof=0)
    expk = (demeaned_rk**4).mean()
    kur = expk/sigma_rk**4
    return kur


import scipy.stats
def is_normal(r, level=0.01):
    '''applies the jarque bera test to see if the distribution is normal or not
    test is applied at the 1% lvl by default
    return true if the hypothesis of normality is accepted'''
    statistics,p_value = scipy.stats.jarque_bera(r)
    return p_value>level


def semideviation(r):
    ''' returns the semi deviation of R '''
    is_negative=r<0
    return r[is_negative].std(ddof=0)
 
def var_historic(r,level=5):
    '''Returns the historic value at risk at specified level
    i.e returns the number such that "level" percent of the returns fall 
    below that number , and the (100-level) percent or above'''
    if isinstance (r ,pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance (r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be series or DataFrame")

from scipy.stats import norm
def var_gaussian(r,level=5,modified=False):
    '''returns the parametric gaussian Var of a series or DataFrame'''
    #compute the z score assiuming it was gaussian 
    z=norm.ppf(level/100)
    if modified:
        #modify the z score based on skewness and kertosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 - 1 )*s/6 +
                 (z**3 -3*z)*(k-3)/24 -
                 (2*z**3 - 5*z)*(s**2)/36
            ) 
    csv = -( r.mean() + z*r.std(ddof=0))
    return csv

def cvar_historic(r,level=5):
    '''Returns the historic value at risk at specified level
    i.e returns the number such that "level" percent of the returns fall 
    below that number , and the (100-level) percent or above'''
    if isinstance (r ,pd.Series):
        is_beyond= r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance (r, pd.DataFrame):
        return r.aggregate(cvar_historic)
    else:
        raise TypeError("Expected r to be series or DataFrame")
        
def annualize_vol (r,periods_per_year):
    '''annualize the vol of a set of returns'''
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r , riskfree_rate ,periods_per_year):
    '''annualized sharpe ratio'''
    rf_per_period = ( 1 + riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret,periods_per_year)
    ann_vol = annualize_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol

def annualize_rets(r, periods_per_year):
    compound_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compound_growth**(periods_per_year/n_periods)-1

def portfolio_return(weights, returns):
    """weights to returns """
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    '''weight to voltaility'''
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points,er,cov):
    '''plot the 2 asset efficient frontier'''
    weights =[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w , er) for w in weights]
    vols = [portfolio_vol(w , cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets ,"Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style='.-')

from scipy.optimize import minimize
def minimize_vol(target_return,er,cov):
    '''target_ret to weights'''
    n = er.shape[0]
    init_guess= np.repeat(1/n,n)
    bounds =((0.0,1.0),)*n
    return_is_target = {
        'type' :'eq',
        'args': (er,),
        'fun' : lambda weights,er :target_return - portfolio_return(weights,er)
    }
    

    weights_sum_to_1={
        'type' :'eq',
        'fun' : lambda weights: np.sum(weights)-1
    }
    
    results = minimize(portfolio_vol, init_guess, args =(cov,), method="SLSQP",
                      constraints =(return_is_target,weights_sum_to_1),
                       bounds = bounds 
                      )
    return results.x


    

def optimal_weight (n_points,er,cov):
    '''list of weights to run the optimizer to minimize the colatility'''
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights= [minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate,er,cov):
    '''Risk free rate+er+cov to weights'''
    n = er.shape[0]
    init_guess= np.repeat(1/n,n)
    bounds =((0.0,1.0),)*n
    
    weights_sum_to_1={
        'type' :'eq',
        'fun' : lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        '''returns the negative of the sharpe ratio, given weights'''
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess, args =(riskfree_rate,er,cov,), method="SLSQP",
                       options ={'disp':False},
                      constraints =(weights_sum_to_1),
                       bounds = bounds 
                      )
    return results.x

def gmv(cov):
    '''returns the weights of the global minimum volatility portfolio given the covariance matrix'''
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)

def plot_ef(n_points , er,cov,show_cml=False, style='.-',riskfree_rate=0,show_ew=False,show_gmv=False):
    '''plot the n asset efficient frontier'''
    weights = optimal_weight(n_points,er,cov)
    rets = [portfolio_return(w , er) for w in weights]
    vols = [portfolio_vol(w , cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets ,"Volatility":vols})
    ax= ef.plot.line(x="Volatility",y="Returns",style=style)
    if show_ew:
        n= er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        ax.plot([vol_ew],[r_ew], color="goldenrod",marker="o", markersize=10)
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv,er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv], color="midnightblue",marker="o", markersize=10)
        
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate,er,cov)
        r_msr = portfolio_return(w_msr,er)
        col_msr = portfolio_vol(w_msr,cov)
        cml_x= [0,col_msr]
        cml_y= [riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color="green",marker ="o",linestyle="dashed",markersize=12,linewidth=2)
    return ax 

def run_cppi(risky_r, safe_r=None , m =3,start=1000,floor=0.8,riskfree_rate=0.03,drawdown=None):
    '''Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    returns a dictionary containing: Asset Value History,Risk Budget History, risk Weight History'''
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak= start
    if isinstance (risky_r ,pd.Series) :
        risky_r =pd.DataFrame(risky_r , columns=["R"])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 #fast way to set all values to a number
    
    account_history =pd.DataFrame().reindex_like(risky_r)
    cushion_history =pd.DataFrame().reindex_like(risky_r)
    risky_w_history =pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak,account_value)
            floor_value= peak*(1-drawdown)
            
        cushion = (account_value - floor_value)/account_value 

        risky_w = m*cushion
        risky_w = np.minimum(risky_w,1)
        risky_w = np.maximum(risky_w,0)
        safe_w = 1 - risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value * safe_w
        ## update the account value for this time step
        account_value = (account_value*risky_w)*(1 + risky_r.iloc[step]) + (account_value * safe_w)*(1+safe_r.iloc[step])
        # save the values so I can look at the history and plot it etc
        ##print(step)
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result ={
        "Wealth":account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m":m,
        "start":start,
        "floor":floor,
        "risky_r": risky_r,
        "safe_r":safe_r
    }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03):

    '''Return a DataFrame that contains aggregated summary stats for the returns in the columns of r'''

    ann_r = r.aggregate(annualize_rets, periods_per_year=12) 
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate= riskfree_rate, periods_per_year=12) 
    dd = r.aggregate(lambda r: drawdown (r).Drawdown.min()) 
    skew = r.aggregate(skewness) 
    kurt = r.aggregate(kurtosis) 
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic) 
    return pd. DataFrame ({

        "Annualized Return": ann_r,

        "Annualized Vol": ann_vol,

        "Skewness": skew,

        "Kurtosis": kurt,

        "Cornish-Fisher VaR (5%)": cf_var5,

        "Historic CVaR (5%)": hist_cvar5,

        "Sharpe Ratio": ann_sr,

        "Max Drawdown": dd

        })

def funding_ratio(assets, liabilities, r ):
    '''
    computes the funding ratio of some assets given liabilities aND INTEREST RATES 
    '''
    return assets/pv(liabilities,r)

def pv(flows,r):
    '''compute the present value of a seq of liabilities
    l is a indexed by time,and the values are the amounts of each liability it returns the present value of  the sequence'''
    dates = flows.index 
    discounts = discount(dates,r)
    return discounts.multiply(flows,axis='rows').sum()

def funding_ratio(assets, liabilities, r ):
    '''
    computes the funding ratio of some assets given liabilities aND INTEREST RATES 
    '''
    return pv(assets,r)/pv(liabilities,r)

def discount (t,r):
    '''compute the price of a pure discount bond that pays a dollar' at time t , given interest rate r '''
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index= t
    return discounts 

def inst_to_ann(r):
    '''
    converts short rates to annualized rates
    '''
    return np.expm1(r) ## same sa exp(r)-1

def ann_to_inst(r):
    """converts annualized to a short rate"""
    return np.log1p(r) ## np.log(1+r) but more efficient

import math
def cir(n_years=10, n_scenarios=10, a=0.05, b=0.03, sigma=0.05, periods_per_year=12, r_0=None):
    '''
    Evolution of (instantaneous) interest rates and corresponding zero-coupon bond using the CIR model:
        dr_t = a*(b-r_t) + sigma*sqrt(r_t)*xi,
    where xi are normal random variable N(0,1). 
    The analytical solution for the zero-coupon bond price is also computed.
    The method returns a dataframe of interest rate and zero-coupon bond prices
    '''
    if r_0 is None:  r_0 = b
    dt = 1 / periods_per_year   
    def price(ttm,r,h):
        _A = ( ( 2*h*np.exp(0.5*(a+h)*ttm) ) / ( 2*h + (a+h)*(np.exp(h*ttm)-1) ) )**(2*a*b/(sigma**2))
        _B = ( 2*(np.exp(h*ttm)-1) ) / ( 2*h + (a+h)*(np.exp(h*ttm)-1) ) 
        return _A * np.exp(-_B * r)
    
    
    num_steps = int(n_years * periods_per_year) + 1
    
    # get the nominal (instantaneous) rate 
    r_0 = ann_to_inst(r_0)
    
    # the schock is sqrt(dt)*xi_t, with xi_t being standard normal r.v.
    shock = np.random.normal(loc=0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    
    # Rates initialization
    rates = np.empty_like(shock)
    rates[0] = r_0 
    
    # Price initialization and parameters
    prices = np.empty_like(shock)
    h = np.sqrt(a**2 + 2*sigma**2)
    prices[0] = price(n_years,r_0,h)

    for step in range(1,num_steps):
        # previous interest rate
        r_t = rates[step-1]
    
        # Current (updated) interest rate: CIR equation
        rates[step] = r_t + a*(b - r_t) + sigma*np.sqrt(r_t)*shock[step]
        
        # Current (updated) ZCB price
        prices[step] = price(n_years - dt*step, r_t, h)       
 
    # the rates generated (according to the periods_per_year) are transformed back to annual rates
    rates = pd.DataFrame( data = inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame( data =prices,index=range(num_steps) )

    return rates, prices

def bond_cash_flows(maturity,principal=100, coupon_rate=0.03, coupons_per_year=2):
    '''
    Generates a pd.Series of cash flows of a regular bond. Note that:
    '''
    # total number of coupons 
    n_coupons = round(maturity * coupons_per_year)
    
    # coupon amount 
    coupon_amt = (coupon_rate / coupons_per_year) * principal 
    coupon_times = np.arange(1,n_coupons+1)
    # Cash flows
    cash_flows = pd.Series(data=coupon_amt, index = coupon_times)
    cash_flows.iloc[-1] = cash_flows.iloc[-1] + principal 
        
    return cash_flows
    
"""def bond_price(principal=100, maturity=10, coupon_rate=0.02, coupons_per_year=2, discount_rate=0.03, cf=None):
    '''
    Return the price of regular coupon-bearing bonds
    Note that:
    - the maturity is intended as an annual variable (e.g., for a 6-months bond, maturity is 0.5);
    - the principal (face value) simply corresponds to the capital invested in the bond;
    - the coupon_rate has to be an annual rate;
    - the coupons_per_year is the number of coupons paid per year;
    - the ytm is the yield to maturity: then ytm divided by coupons_per_year gives the discount rate of cash flows
    The ytm variable can be both a single value and a pd.DataFrame. 
    In the former case, a single bond price is computed. In addition, if the flux of cash flows is computed beforehand, 
    the method can takes it as input and avoid recomputing it.
    In the latter case, the dataframe is intended as a t-by-scenarios matrix, where t are the dates and scenarios denotes
    the number of rates scenario in input. Here, for each scenario, single bond prices are computed according to different ytms.
    '''
    # single bond price 
    def single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, discount_rate=discount_rate, cf=cf):
        if cf is None:            
            # compute the bond cash flow on the fly
            cf = bond_cash_flows(maturity=maturity, principal=principal, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)             
        bond_price = present_value(cf, discount_rate/coupons_per_year)[0]
        return bond_price
    
    if isinstance(discount_rate,pd.Series):
        raise TypeError("Expected pd.DataFrame or a single value for discount_rate")

    if isinstance(discount_rate,pd.DataFrame):
        # ytm is a dataframe of rates for different scenarios 
        n_scenarios = discount_rate.shape[1]
        bond_price  = pd.DataFrame()
        # we have a for over each scenarios of rates (ytms)
        for i in range(n_scenarios):
            # for each scenario, a list comprehension computes bond prices according to ytms up to time maturity minus 1
            prices = [single_price_bond(principal=principal, maturity=maturity - t/coupons_per_year, coupon_rate=coupon_rate,
                                        coupons_per_year=coupons_per_year, discount_rate=y, cf=cf) for t, y in zip(discount_rate.index[:-1], discount_rate.iloc[:-1,i]) ] 
            bond_price = pd.concat([bond_price, pd.DataFrame(prices)], axis=1)
        # rename columns with scenarios
        bond_price.columns = discount_rate.columns
        # concatenate one last row with bond prices at maturity for each scenario
        bond_price = pd.concat([ bond_price, 
                                 pd.DataFrame( [[principal+principal*coupon_rate/coupons_per_year] * n_scenarios], index=[discount_rate.index[-1]]) ], 
                                axis=0)
        return bond_price 
    else:
        # base case: discount_rate is a value and a single bond price is computed 
        return single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, 
                                 coupons_per_year=coupons_per_year, discount_rate=discount_rate, cf=cf)        

def bond_returns(principal, bond_prices, coupon_rate, coupons_per_year, periods_per_year, maturity=None):
    '''
    Computes the total return of a coupon-paying bond. 
    The bond_prices can be a pd.DataFrame of bond prices for different ytms and scenarios 
    as well as a single bond price for a fixed ytm. 
    In the first case, remind to annualize the computed returns.
    In the latter case, the maturity of the bond has to passed since cash-flows needs to be recovered. 
    Moreover, the computed return does not have to be annualized.
    '''
    if isinstance(bond_prices, pd.DataFrame):
        coupons = pd.DataFrame(data=0, index=bond_prices.index, columns=bond_prices.columns)
        last_date = bond_prices.index.max()
        pay_date = np.linspace(periods_per_year/coupons_per_year, last_date, int(coupons_per_year*last_date/periods_per_year), dtype=int  )
        coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
        tot_return = (bond_prices + coupons)/bond_prices.shift(1) - 1 
        return tot_return.dropna()
    else:
        cf = bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year) 
        tot_return = ( cf.sum() / bond_prices )**(1/maturity) - 1
        return tot_return[0]
        
"""

def bond_price(maturity,principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    if isinstance(discount_rate,pd.DataFrame):
        princing_dates = discount_rate.index
        prices = pd.DataFrame(index=princing_dates,columns=discount_rate.columns)
        for t in princing_dates:
            prices.loc[t]= bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year, discount_rate.loc[t])
        return prices
    
    else: # base case..... single time period
        if maturity <=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows= bond_cash_flows(maturity, principal,coupon_rate,coupons_per_year)
        return pv(cash_flows,discount_rate/coupons_per_year)
""" 
def mac_duration(flows, discount_rate):
    '''
    computes Macualy duration of a seq of cash flows
    '''
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    avg = np.average(flows.index, weights=weights)
    return avg
"""
def mac_duration(cash_flows, discount_rate):
    '''
    Computed the Macaulay duration of an asset involving regular cash flows a given discount rate
    Note that if the cash_flows dates are normalized, then the discount_rate is simply the YTM. 
    Otherwise, it has to be the YTM divided by the coupons per years.
    '''
    if not isinstance(cash_flows,pd.DataFrame):
        raise ValueError("Expected a pd.DataFrame of cash_flows")

    dates = cash_flows.index

    # present value of single cash flows (discounted cash flows)
    discount_cf = discount( dates, discount_rate ) * cash_flows
    
    # weights: the present value of the entire payment, i.e., discount_cf.sum() is equal to the principal 
    weights = discount_cf / discount_cf.sum()
    
    # sum of weights * dates
    return ( weights * pd.DataFrame(dates,index=weights.index) ).sum()[0] 

def match_duration(cf_t,cf_s, cf_l, discount_rate):
    """
    returns the weights W in cf_s that along with (1-W) in cf_l will have an effective duration that matches cf_t
    """
    d_t = mac_duration(cf_t,discount_rate)
    d_s = mac_duration(cf_s,discount_rate)
    d_l = mac_duration(cf_l,discount_rate)
    return (d_l-d_t)/(d_l-d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int  )
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    tot_return = (monthly_prices + coupons)/monthly_prices.shift()-1 
    return tot_return.dropna()

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    
    rets_g8020 = erk.bt_mix(rets_eq, rets_bonds,allocator=erk.glidepath_allocator, start_glide=0.80, end_glide=.20)
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths
def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

""" calling functions 
    rates, zc_prices = erk.cir(10,n_scenaRIOS=000,mu=,sigma=)

"""