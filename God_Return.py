import ccxt
import datetime as dt
from datetime import  datetime
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as s
import streamlit as st
pd.set_option("display.precision", 6)
# sns.set_style("whitegrid")

class Run_model(object) :
    def __init__(self ):
        self.pair_data = "BTC-PERP"
        self.timeframe = "1h"  
        self.loop_start = dt.datetime(2020, 6 , 30  , 0, 0)
        self.loop_end = dt.datetime(2020, 7 , 10  , 0, 0)
        self.input  = 'rsi'
        self.length = 30

    def dataset (self):
        self.exchange = ccxt.ftx({'apiKey': '' ,'secret': ''  , 'enableRateLimit': True }) 
        ohlcv = self.exchange.fetch_ohlcv(self.pair_data, self.timeframe  , limit=5000)
        ohlcv = self.exchange.convert_ohlcv_to_trading_view(ohlcv)
        df =  pd.DataFrame(ohlcv)
        df.t = df.t.apply(lambda  x :  datetime.fromtimestamp(x))
        return df

    @property
    def loop (self):
        df =  self.dataset()
        df = df[df.t >= self.loop_start] ; df = df[df.t <= self.loop_end]
        df =  df.set_index(df['t']) ; df = df.drop(['t'] , axis= 1 )
        df = df.rename(columns={"o": "open", "h": "high"  , "l": "low", "c": "close" , "v": "volume"})
        dataset = df  ; dataset = dataset.dropna()
        return dataset

    def represent (self):
        df = self.loop ; df.ta.ohlc4(append=True)
        return df

    def god_returns (self):
        god_returns = self.represent()
        god_returns['Mk_Returntime+1']  = np.log(god_returns['OHLC4'] / god_returns['OHLC4'].shift(1))
        god_returns['Mk_Returntime+1'] = god_returns['Mk_Returntime+1'].shift(-1)
        god_returns['God_Buyonly'] = np.where( god_returns['Mk_Returntime+1'] > 0 ,  god_returns['Mk_Returntime+1']    , 0  )
        god_returns['God_Sellonly'] = np.where( god_returns['Mk_Returntime+1'] < 0 ,  abs(god_returns['Mk_Returntime+1'])    , 0  )
        god_returns['God_Buysell'] = np.where( True ,  abs(god_returns['Mk_Returntime+1'])  ,  abs(god_returns['Mk_Returntime+1'])  )
        god_returns['Cum_Godbuyonly'] = np.cumsum(god_returns['God_Buyonly'])
        god_returns['Cum_Godsellonly'] = np.cumsum(god_returns['God_Sellonly'])
        god_returns['Cum_Buysell'] = np.cumsum(god_returns['God_Buyonly'])
        god_returns['Cum_Buyhold']  = np.cumsum(god_returns['Mk_Returntime+1'])
        god_returns = god_returns.iloc[: , -9:]
        return god_returns

    def fx (self):
        fx = self.represent()
        fx['Mk_Returntime+1']  = np.log(fx['OHLC4'] / fx['OHLC4'].shift(1))
        fx['Mk_Returntime+1'] = fx['Mk_Returntime+1'].shift(-1)
        try: fx['F(x)'] = fx.ta(kind =self.input , length= self.length , scalar=1 , append=False)
        except:pass
        fx = fx.iloc[: , 5:] ; fx = fx.fillna(0)  ; fx_toaction = fx
        fx_toaction['F(x)_Action'] = np.where( fx_toaction['F(x)'].shift(1) <  fx_toaction['F(x)'].shift(0)  , 'buy' , 'sell' )
        fx_toaction['F(x)_BuyReturn'] = np.where(fx_toaction['F(x)_Action'] == 'buy'  , fx_toaction['Mk_Returntime+1'] ,  0)
        fx_toaction['F(x)_CumBuyonly'] = np.cumsum(fx_toaction['F(x)_BuyReturn'])
        fx_toaction['F(x)_SellReturn'] = np.where(fx_toaction['F(x)_Action'] == 'sell'  , -fx_toaction['Mk_Returntime+1'] ,  0)
        fx_toaction['F(x)_CumSellonly'] = np.cumsum(fx_toaction['F(x)_SellReturn'])
        fx_toaction['F(x)_BuySellReturn'] = np.where( fx_toaction['F(x)_Action'] == 'buy' , fx_toaction['Mk_Returntime+1'] , -fx_toaction['Mk_Returntime+1'])
        fx_toaction['F(x)_CumBuySell'] = np.cumsum(fx_toaction['F(x)_BuySellReturn'])
        return  fx_toaction

    def fx_scatter (self):
        dataset = self.fx()
        dataset['buy'] = dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'buy' , x.OHLC4 , None) , axis=1)
        dataset['sell'] =  dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'sell'  , x.OHLC4 , None) , axis=1)
        plt.figure(figsize=(12,8))
        plt.plot(dataset.OHLC4 , color='k' , alpha=0.20 )
        plt.plot(dataset.buy , 'o',  color='g' , alpha=0.50 )
        plt.plot(dataset.sell , 'o', color='r' , alpha=0.50)      
        plt.show()
        
    def fx_chart (self):
        fx_chart = self.fx()
        plt.figure(figsize=(12,8))
        plt.plot(fx_chart['F(x)_CumBuyonly'], color='k',  alpha=0.60 )
        plt.plot(fx_chart['F(x)_CumSellonly'], color='g',  alpha=0.60 )
        plt.plot(fx_chart['F(x)_CumBuySell'], color='r',  alpha=0.60 )
        st.pyplot()

    def god_chart (self):
        god_chart = self.god_returns()
        plt.figure(figsize=(12,8))
        plt.plot(god_chart['Cum_Godbuyonly'], color='k',  alpha=0.60 )
        plt.plot(god_chart['Cum_Godsellonly'], color='g',  alpha=0.60 )
        plt.plot(god_chart['Cum_Buysell'], color='r',  alpha=0.60 )
        st.pyplot()

#____________________________________________________________________________  

if __name__ == "__main__":
    model =  Run_model()
    st.sidebar.header('....header..... \n')
    selectbox = lambda y : st.sidebar.selectbox('input F(x)',
            ( y ,'accbands','ad','adx','ao','aroon','atr','bbands',
            'bop','cci','cg','cmf','cmo','coppock','cross','decreasing','dema',
            'donchian','dpo','efi','ema','eom','fwma','hl2','hlc3','hma','ichimoku',
            'increasing','kc','kst','kurtosis','linear_decay','linreg','log_return',
            'long_run','mad','median','mfi','midpoint','midprice','mom','natr',
            'nvi','obv','ohlc4','percent_return','pvi','pvol','pvt','pwma','qstick',
            'quantile','rma','roc','rsi','rvi','short_run','sinwma','skew','slope','sma',
            'stdev','stoch','swma','t3','tema','trima','true_range','uo','variance',
            'vortex','vp','vwap','vwma','willr','wma','zlma','zscore'))
    
    st.sidebar.text("_"*45)
    model.pair_data =   st.sidebar.text_input('data' , "BTC-PERP")
    model.timeframe =   st.sidebar.selectbox('timeframe',('1h' , '5m' , '15m' , '1h', '4h' ,'1d'))
    model.loop_start =  np.datetime64(st.sidebar.date_input('loop_start', value= dt.datetime(2020, 7, 10, 0, 0)))
    model.loop_end =    np.datetime64(st.sidebar.date_input('loop_end', value= dt.datetime(2020, 7, 17, 0, 0)))

    
    
    st.sidebar.text("_"*45)
    model.input = selectbox('rsi')
    model.length = st.sidebar.slider('length_parameter' , 1 , 500 , 30)
    st.sidebar.text("_"*45)
    
    pyplot = model.god_chart()
    st.write(model.god_returns())
    st.write("_"*45)
    pyplot = model.fx_scatter()
    pyplot = model.fx_chart()
    st.write(model.fx())
    

# # st.sidebar.text("_"*45)
# pyplot = model.chart
# pyplot = model.nav
# if st.checkbox('df_plot'):
#     st.write(pyplot.iloc[: , :])
# st.text("")
# st.write('\n\nhttps://github.com/firstnattapon/test-stream/edit/master/app.py')
