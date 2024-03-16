import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
# https://github.com/laholmes/risk-adjusted-return/blob/master/app.py
# performance measure 참고


def value_func(x, alpha, beta, lambda_):
#     print(x)
    return np.power(x,alpha) if x >= 0 else -lambda_ * np.power(-x,beta)



def get_pairs(x,p):
    pairs = []
    for i in range(len(x)):
        pairs.append((x[i],p[i]))
    pairs.sort(key=lambda y: y[0])
    x_sorted = []
    p_sorted = []
    for i in range(len(x)):
        x_sorted.append(pairs[i][0])
        p_sorted.append(pairs[i][1])
    
    return x_sorted, p_sorted
    
# def w(p, pos_neg):
#     if pos_neg == 'pos':
#         res = (0.65*(p**0.6))/(0.65*(p**0.6)+((1-p)**0.6))
#     else:
#         res = (0.84*(p**0.65))/(0.84*(p**0.65)+((1-p)**0.6))
#     return res

# def pi(i, n, p, pos_neg):
# #     print('pi new')
    
#     p = np.array(p)  
#     if pos_neg == 'pos':
#         sum1 = p[i:n].sum()
#         sum2 = p[i+1:n].sum()
#         res = w(sum1, pos_neg) - w(sum2, pos_neg)
#     else:
#         sum1 = p[0:i+1].sum()
#         sum2 = p[0:i].sum()
#         res = w(sum1, pos_neg) - w(sum2, pos_neg)
#     return res

def get_pis(x,p):
    pis = []
    x = np.array(x)
    p = np.array(p)
    pos_sum1 = np.flip(np.flip(p).cumsum())
    pos_sum2 = np.roll(pos_sum1,-1)
    
    pos_w_fnc = lambda t: (0.65*np.power(t,0.6))/(0.65*np.power(t,0.6)+(np.power(1-t,0.6)))
    pos_w = np.vectorize(pos_w_fnc)
    pos1 = pos_w(pos_sum1)
    pos2 = pos_w(pos_sum2)
    pos2[-1] = 0.0
    
    
    pos_diff = pos1 - pos2
    
    neg_sum1 = p.cumsum()
    neg_sum2 = np.roll(neg_sum1,1)
    
    neg_w_fnc = lambda t: (0.84*np.power(t,0.65))/(0.84*np.power(t,0.65)+np.power(1-t,0.65))
    neg_w = np.vectorize(neg_w_fnc)
    
    
    neg1 = neg_w(neg_sum1)
    neg2 = neg_w(neg_sum2)
    
    neg2[0] = 0.0
    
    neg_diff = neg1 - neg2
    pos_neg = x >= 0
    
    res = pos_neg * pos_diff + (1-pos_neg)*neg_diff
    
    res[0] = 0.0
    res[-1] = 1-np.cumsum(res)[-2]
    
    return res

# def get_pis(x,p):
#     pis = []
#     for i in range(len(x)):
#         print(i)
#         if x[i] >= 0:
#             pos_neg = 'pos'
#         else:
#             pos_neg = 'neg'
        
#         pi_ = pi(i=i, n=len(x), p=p, pos_neg=pos_neg)
#         if np.isnan(pi_):
#             pi_ = p[i]
#         pis.append(pi_)
#     return pis




def CPV(x, pis, alpha, beta, lambda_):
    CPVs = []
    vals = []
    for i in range(len(x)):
        if x[i] >= 0:
            pos_neg = 'pos'
        else:
            pos_neg = 'neg'
        val = value_func(x[i], alpha=alpha, beta=beta, lambda_=lambda_)
        vals.append(val)
        CPVs.append(pis[i]*val)
    
#     print('pis:', pis)
#     print('pis sum:', np.sum(pis))
#     print('val:', vals)
#     print('CPVs:', CPVs)
    return np.sum(CPVs)

def MPV(x, pis, alpha, beta, lambda_):
    MPVs = []
    new_pis = []
    for i in range(len(x)):
        if x[i] >= 0:
            pos_neg = 'pos'
        else:
            pos_neg = 'neg'
        
        if pis[i] == 0:
            new_pis.append(0)
        else:
            new_pis.append(1.0)
        
        val = value_func(x[i], alpha=alpha, beta=beta, lambda_=lambda_)
        MPVs.append(val)
    
    MPVs = np.array(MPVs)
    new_pis = np.array(new_pis)
    new_pis = new_pis / new_pis.sum()
    res = MPVs * new_pis
    
#     print('lambda_:', lambda_)
#     print('val:', MPVs)
#     print('new_pis:', new_pis)
    return res.sum()

def semi(r, freq):
    if freq == 'D':
        win = 252
    elif freq == 'Y':
        win = 1
    
    r = pd.Series(r)
    
    return r[r <= r.mean()].std()*np.sqrt(win)

def VaR(r, level, freq):
    if freq == 'D':
        win = 252
    elif freq == 'Y':
        win = 1
    
    return np.percentile(r, level) * win

def ES(r, level, freq):
    if freq == 'D':
        win = 252
    elif freq == 'Y':
        win = 1

    return r.loc[r < VaR(r,5,freq=freq)/win].mean() * win 

def Omega(ret, threshold, freq):
    if freq == 'D':
        threshold = (threshold+1)**(1/252)-1
    
    excess_ret = ret - threshold
    pos_sum = excess_ret.loc[excess_ret>0].sum()
    neg_sum = excess_ret.loc[excess_ret<0].sum()
    
    return - pos_sum / neg_sum

# def Omega2(ret, threshold):
#     daily_threshold = (threshold+1)**np.sqrt(1/252)-1
    
#     x = pd.Series(sorted(list(ret)))
#     y = pd.Series(np.arange(len(x)) / float(len(x)))

#     pos_ind = x.loc[x > daily_threshold].index
#     neg_ind = x.loc[x < daily_threshold].index
    
#     return (1-y.iloc[pos_ind]).sum() / y.iloc[neg_ind].sum()
        
def MDD(price):
    Roll_Max = pd.Series(price).rolling(252, min_periods=1).max()
    Daily_Drawdown = price/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(252, min_periods=1).min()
    
    return Max_Daily_Drawdown.min()

def AvDD(price):
    Roll_Max = pd.Series(price).rolling(252, min_periods=1).max()
    Daily_Drawdown = price/Roll_Max - 1.0
    
    return Daily_Drawdown.mean()

def quad_u(r,b):
    r = np.array(r)
    res = 1+r - 0.5*b*(1+r)**2 
    return res.mean()

def log_u(r):
    # HARA (hyperbolic absolute risk aversion), r -> 0
    r = np.array(r)
    res = np.log(1.0000000001+r)
    return res.mean()

def power_u(r, a):

    r = np.array(r)
    res = (1+r)**a
    return res.mean()

def exp_u(r, b):
    # a > 0
    # constant ARA (Absolute Risk Aversion)
    # HARA (hyperbolic absolute risk aversion), r -> -inf
    r = np.array(r)
    res = -np.exp(-b*(1+r))
    return res.mean()

def d1(S,K,r,sigma,T,t):
    #S/K = a
    #S = a * K
#     print('S_0:', S)
#     print('K:', K)
#     print('rf:', r)
#     print('sigma:', sigma)
#     print('T:', T)
    c = np.log(S/K)+(r+(sigma**2)/2)*(T-t)
#     print('c:', c)
    b = sigma*np.sqrt(T-t)
#     print('b:', b)
    return c/b

def min_sigma(S,K,r,T):
    a = 2*(np.log(S/K)+r*T)
    return np.sqrt(a/T)

def get_E(price, ret, vol, T, a, rf, trad_filter):
    #set the strategy params
    K = price[0] * a
    r = rf[0]
    E = []
    sigma = vol[0] * np.sqrt(12)#np.sqrt(252)
    # GBM의 경우 아래의 식은 성능의 약간의 저하를 보여줌 (비용이니까 맞는 결과)
#     sigma = sigma * np.sqrt(1+np.sqrt(2/np.pi)*(0.001/(sigma*np.sqrt(1/252))))
    sigma = sigma * np.sqrt(1+np.sqrt(2/np.pi)*(0.001/(sigma*np.sqrt(1/12))))
    
    
#     count_small = 0
#     count_big = 0
    for i in range(len(price)):
        
        #t = i % (T*252) / 252
        t = i % (T*12) / 12
        r = rf[i]
        
        if i == 0:
            d1_result = d1(price[i],K,r,sigma,T,t)
            Nd1 = norm.cdf(d1_result)
        else:
            if np.abs(ret[i-1]) > trad_filter:
                d1_result = d1(price[i],K,r,sigma,T,t)
                Nd1 = norm.cdf(d1_result)
            else:
                Nd1 = prev_Nd1
        
        
#         d1_list = []
#         d1_x = []
#         for j in range(1000):
#             d1_x.append(j/100)
#             d1_list.append(d1(price[i],K,r,j/100,T,t))
        
#         plt.plot(d1_x, d1_list)
#         plt.show()
#         plt.close()
        
#         min_sig = min_sigma(price[i],K,r,T)
    
        
#         print('Sigma original:', vol[0] * np.sqrt(252))
#         print('Sigma adjusted:', )
#         print('Sigma min:', min_sig)
        
#         if np.isnan(min_sig):
#             count_big = count_big + 1
#         else:
#             if sigma < min_sig:
#                 count_small = count_small + 1
#             else:
#                 count_big = count_big + 1
        
        
        
        prev_Nd1 = Nd1
        
        
#         d1_result = d1(price[i],K,r,sigma,T,t)
#         Nd1 = norm.cdf(d1_result)
        
#         print(d1_result)
        
#         print('Nd1:', Nd1)
#         d2_result = Nd1 - sigma*np.sqrt(T)#(T-t)
#         Nd2 = norm.cdf(-d2_result)
#         w_risky = (price[i]*Nd1) / (price[i]*Nd1 + K*np.exp(-r*T)*Nd2)   
        
#         print('Nd1:', Nd1)
#         print('w_risky:', w_risky)
        E.append(Nd1)
#         E.append(w_risky)
#         K = a * (price[i] * Nd1 + K*np.exp(-r*T)*Nd2)
#         print('price:', price[i])
#         print('K', K)
          # GBM의 경우 매번 vol 업데이트는 있으나 마나 별 다를 바가 없음.
#         sigma = vol[i] * np.sqrt(252)# 저자가 롤링 윈도우 안쓰고 true라고 생각되는 어떤 one value를 쓴듯.
#         sigma = sigma * np.sqrt(1+np.sqrt(2/np.pi)*(0.001/(sigma*np.sqrt(1/252))))
#         print('sigma:', sigma)
#     print('count_total:', count_small+count_big)
#     print('count_small:', count_small)
#     print('count_big:', count_big)
    return E#, count_small, count_big

def get_E_stop_loss(price, vol, T, a, rf):
    # vol, T는 사용 안함
    S = price[0]
    K = S * a
    once_below = False
    
    E = []
    for i in range(len(price)):
#         discount = (1+rf)**(1/252)-1
        discount = (1+rf/12)-1
        if S < K / ((1+discount)**(len(price)-i)) or once_below == True:
            E.append(0.0)
            once_below = True
        else:
            E.append(1.0)

        S = price[i]
        
    return E

def delta_hedging(E, ret, price, rf, cost=True):
    cash = 1.0
    result = []
    
    prev_equity_invested = 0.0
    for i in range(0,len(ret)):
#         print(i)
#         print('prev_equity_invested:', prev_equity_invested)
        equity_w = E[i-1]
        
        if i == 0:# 수익률 반영
            result.append(cash)
            
        if i != 0:
            equity_invested = equity_invested * (1+ret[i])
            cash = equity_invested + bond_invested
#             print('수익률:', (1+ret[i]))
#             print('수익률을 반영한 내 스톡가치:', equity_invested)
#             print('수익률을 반영한 내 자산가치:', cash)
            result.append(cash)
        
        
        # 리밸런싱반영
#         print('스톡에 이만큼 투자:', equity_w)
        bond_w = 1 - equity_w#E[i]
#         print('현금에 이만큼 투자:', bond_w)
        equity_invested = cash*equity_w
#         print('스톡가치_after_rebal_before_cost:', equity_invested)
        bond_invested = cash*bond_w
#         print('현금가치:', bond_invested)
#         print('내 현재 자산 가치_before_cost:', cash)
        
        if cost:
            if  i == 0:
                sell_or_buy = cash * equity_w
            else:
                sell_or_buy = np.abs(equity_invested - prev_equity_invested)
            
#             print('사고판 총 액수:', sell_or_buy)
#             print('수수료:', sell_or_buy * (1.39/1000))
            
            equity_invested = equity_invested - sell_or_buy * (1.0/1000) #- sell_or_buy * (1/1000)
        
#         print('equity invested:', equity_invested)
#         print('bond invested:', bond_invested)
        
#         print('스톡가치_after_cost:', equity_invested)
        r = rf[i]
        cash = equity_invested + bond_invested * (1+r/252)
        prev_equity_invested = equity_invested
    
    return pd.Series(result)#, index=ret.index)

def delta_hedging_SP(price, rf, vol, a, T):
    cash = 1.0
    K = price[0] * a
    sigma = vol[0] * np.sqrt(12)
    # GBM의 경우 아래의 식은 성능의 약간의 저하를 보여줌 (비용이니까 맞는 결과)
    sigma = sigma * np.sqrt(1+np.sqrt(2/np.pi)*(0.001/(sigma*np.sqrt(1/12))))
    result = []
    
    for i in range(0,len(price)):
        
        t = i % (T*12) / 12       
        c = np.log(price[i]/K)+(rf+(sigma**2)/2)*(T-t)
        b = sigma*np.sqrt(T-t)
        d1_result = c/b
        Nd1 = norm.cdf(d1_result)
        d2_result = d1_result - sigma * np.sqrt(T-t)
        Nd2 = norm.cdf(-d2_result)
        
        res = price[i] * Nd1 + K * np.exp(-rf*(T-t)) * Nd2
        
        result.append(res)
        
    result = pd.Series(result)
    result = result / result.iloc[0]
    return result


def get_cppi(cppi_init, ret, m, f, T, rf, cost=True):

    result = []
    prev_equity_invested = cppi_init
    E_list = []
    for j in range(len(ret)):
        if j == 0: ## 첫날
            cppi = 1.0
            B = 1 - cppi_init
            floor_value = cppi * f * np.exp(-rf*T) # F0
#             print('floor_value:', floor_value)
            E_next = cppi_init * (1+ret[j])
#             print('E_next:', E_next)
            
        if j != 0:# 그 이후
            E = max(min(cushion * m, cppi), 0) # 내 재산 갖고 콜옵션 m개를 사자.
#             print('E:', E)
            B = cppi - E # 콜옵션 m개 사고 남는 돈은현금으로 갖고 있자.
#             print('B:', B)
            E_next = E * (1+ret[j])
#             print('E_next:', E_next)

        if cost:
            if  j == 0:
                sell_or_buy = cppi_init
            else:
                sell_or_buy = np.abs(E_next - prev_equity_invested)
            
#             print('사고판 총 액수:', sell_or_buy)
#             print('수수료:', sell_or_buy * (1.39/1000))
            
            E_next = E_next - sell_or_buy * (1.0/1000) #- sell_or_buy * (1/1000)
        
        cppi = E_next + B * (1+rf/12)
#             print('cppi:', cppi)
#         floor_value = floor_value * np.exp(rf*j) # Ft = F0*exp(rf*t)
        cushion = cppi - floor_value # 콜옵션 가격을 트래킹하는 것임.
#             print('cushion:', cushion)
        prev_equity_invested = E_next
        
        E_prop = E_next/(E_next+B)
        E_list.append(E_prop)
    
        result.append(cppi)
    res = pd.Series(result)#, index=ret.index)
    res = res / res.iloc[0]
    
    
            
    return res, E_list

def get_tipp(tipp_init, ret, m, f, T, rf, cost=True):

    result = []
    prev_equity_invested = tipp_init
    E_list = []
    for j in range(len(ret)):
        if j == 0: ## 첫날
            tipp = 1.0
            B = 1 - tipp_init
            floor_value = tipp * f * np.exp(-rf*T)
#             print('floor_value:', floor_value)
            E_next = tipp_init * (1+ret[j])
#             print('E_next:', E_next)
            
        if j != 0:# 그 이후
            
            E = max(min(cushion * m, tipp), 0) # 내 재산 갖고 콜옵션 m개를 사자.
#             print('E:', E)
            B = tipp - E # 콜옵션 m개 사고 남는 돈은현금으로 갖고 있자.
#             print('B:', B)
            E_next = E * (1+ret[j])
#             print('E_next:', E_next)

        if cost:
            if  j == 0:
                sell_or_buy = tipp_init
            else:
                sell_or_buy = np.abs(E_next - prev_equity_invested)
            
#             print('사고판 총 액수:', sell_or_buy)
#             print('수수료:', sell_or_buy * (1.39/1000))
            
            E_next = E_next - sell_or_buy * (1.0/1000) #- sell_or_buy * (1/1000)        
       
        tipp = E_next + B * (1+rf/12)
        
#         floor_value = floor_value * np.exp(rf*j) # Ft = F0*exp(rf*t)
        
        temp_floor_value = tipp * f * np.exp(-rf*T)
        if temp_floor_value > floor_value:
            floor_value = temp_floor_value # TIPP rule : 새 floor가 이전보다 클 때 floor value를 업데이트
        else:
            pass
#             print('cppi:', cppi)
        
        cushion = tipp - floor_value # 콜옵션 가격을 트래킹하는 것임.
#             print('cushion:', cushion)
        prev_equity_invested = E_next
    
        E_prop = E_next/(E_next+B)
        E_list.append(E_prop)
        
        result.append(tipp)
    res = pd.Series(result)#, index=ret.index)
    res = res / res.iloc[0]
            
    return res, E_list

def get_E_e_c(price, e):
    
    E = []
    for i in range(len(price)):
        E.append(e)
        
    return E

def BH_50_50(ret, rf):
    cash = 1.0
    result = []
    
    for i in range(0,len(ret)):
#         print(i)
#         print('prev_equity_invested:', prev_equity_invested)
        equity_w = 0.5
        bond_w = 0.5
        if i == 0:# 수익률 반영
            equity_invested = equity_w * cash
            bond_invested = bond_w * cash
            
        if i != 0:
            equity_invested = equity_invested * (1+ret[i])
            bond_invested = bond_invested * (1+rf/252)
            
        cash = equity_invested + bond_invested
        result.append(cash)
    
    return pd.Series(result)#, index=ret.index)

