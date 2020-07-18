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
        self.pair_data  = "BTC-PERP"
        self.timeframe  = "1h"  
        self.loop_start = dt.datetime(2020, 6 , 30  , 0, 0)
        self.loop_end   = dt.datetime(2020, 7 , 10  , 0, 0)
        self.input      = 'rsi'
        self.length     = 30
        self.BuySell    = st.checkbox('BuySell', value = True)
        self.Buyonly    = st.checkbox("Buyonly")
        self.Sellonly   = st.checkbox('Sellonly')
        self.Buyhold    = st.checkbox('Buyhold')

    def dataset (self):
        self.exchange = ccxt.ftx({'apiKey': '' ,'secret': ''  , 'enableRateLimit': True }) 
        ohlcv = self.exchange.fetch_ohlcv(self.pair_data, self.timeframe  , limit=500)
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
        god_returns['God_Buyonly+1'] = np.where( god_returns['Mk_Returntime+1'] > 0 ,  god_returns['Mk_Returntime+1']    , 0  )
        god_returns['God_Sellonly+1'] = np.where( god_returns['Mk_Returntime+1'] < 0 ,  abs(god_returns['Mk_Returntime+1'])    , 0  )
        god_returns['God_Buysell+1'] = np.where( True ,  abs(god_returns['Mk_Returntime+1'])  ,  abs(god_returns['Mk_Returntime+1'])  )
        god_returns['Cum_Godbuyonly'] = np.cumsum(god_returns['God_Buyonly+1'])
        god_returns['Cum_Godsellonly'] = np.cumsum(god_returns['God_Sellonly+1'])
        god_returns['Cum_Buysell'] = np.cumsum(god_returns['God_Buysell+1'])
        god_returns['Cum_Buyhold']  = np.cumsum(god_returns['Mk_Returntime+1'])
        god_returns = god_returns.iloc[: , -9:]
        god_returns = god_returns.dropna()
        return god_returns

    def fx (self):
        fx = self.represent()
        fx['Mk_Returntime+1']  = np.log(fx['OHLC4'] / fx['OHLC4'].shift(1))
        fx['Mk_Returntime+1'] = fx['Mk_Returntime+1'].shift(-1)
        try: fx['F(x)'] = fx.ta(kind =self.input , length= self.length , scalar=1 , append=False)
        except:pass
        fx = fx.iloc[: , 5:] ; fx_toaction = fx
        fx_toaction['F(x)_Action'] = np.where( fx_toaction['F(x)'].shift(1) <  fx_toaction['F(x)'].shift(0)  , 'buy' , 'sell' )
        fx_toaction['F(x)_BuyReturn'] = np.where(fx_toaction['F(x)_Action'] == 'buy'  , fx_toaction['Mk_Returntime+1'] ,  0)
        fx_toaction['F(x)_CumBuyonly'] = np.cumsum(fx_toaction['F(x)_BuyReturn'])
        fx_toaction['F(x)_SellReturn'] = np.where(fx_toaction['F(x)_Action'] == 'sell'  , -fx_toaction['Mk_Returntime+1'] ,  0)
        fx_toaction['F(x)_CumSellonly'] = np.cumsum(fx_toaction['F(x)_SellReturn'])
        fx_toaction['F(x)_BuySellReturn'] = np.where( fx_toaction['F(x)_Action'] == 'buy' , fx_toaction['Mk_Returntime+1'] , -fx_toaction['Mk_Returntime+1'])
        fx_toaction['F(x)_CumBuySell'] = np.cumsum(fx_toaction['F(x)_BuySellReturn'])
        fx_toaction['F(x)_CumBuyhold']  = np.cumsum(fx_toaction['Mk_Returntime+1'])
        fx_toaction = fx_toaction.dropna()
        return  fx_toaction

    def fx_scatter (self):
        dataset = self.fx()
        dataset['buy'] = dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'buy' , x.OHLC4 , None) , axis=1)
        dataset['sell'] =  dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'sell'  , x.OHLC4 , None) , axis=1)
        plt.figure(figsize=(12,8))
        plt.plot(dataset.OHLC4 , color='k' , alpha=0.20)
        plt.plot(dataset.buy , 'o',  color='g' , alpha=0.50 ,  label= 'fxtoaction = buy')
        plt.plot(dataset.sell , 'o', color='r' , alpha=0.50 ,  label= 'fxtoaction = sell')
        plt.legend(fontsize=12)
        plt.xlabel('price',fontsize=14)
        plt.ylabel('%',fontsize=14)
        st.pyplot()
        
    def fx_chart (self):
        fx_chart = self.fx()
        plt.figure(figsize=(12,8))
        plt.plot(fx_chart['F(x)_CumBuyonly'], color='g',  alpha=0.60 , label= 'F(x)_CumBuy' )
        plt.plot(fx_chart['F(x)_CumSellonly'], color='r',  alpha=0.60 ,label= 'F(x)_CumSell' )
        plt.plot(fx_chart['F(x)_CumBuySell'], color='b',  alpha=0.60 , label= 'F(x)_CumBuy&Sell' )
        plt.plot(fx_chart['F(x)_CumBuyhold'], color='k',  alpha=0.60 , label= 'F(x)_CumBuyhold')
        plt.axhline(y=0.0, color='k', linestyle='-.')
        plt.legend(fontsize=12)
        plt.xlabel('cycle',fontsize=14)
        plt.ylabel('%',fontsize=14)
        st.pyplot()

    def god_chart (self):
        god_chart = self.god_returns()
        plt.figure(figsize=(12,8))
        plt.plot(god_chart['Cum_Godbuyonly'], color='g',  alpha=0.60  , label= 'Cum_Godmaxbuy' )
        plt.plot(god_chart['Cum_Godsellonly'], color='r',  alpha=0.60  , label= 'Cum_Godmaxsell' )
        plt.plot(god_chart['Cum_Buysell'], color='b',  alpha=0.60  , label= 'Cum_GodmaxBuy&sell' )
        plt.plot(god_chart['Cum_Buyhold'], color='k',  alpha=0.60  , label= 'Cum_Buyhold' )
        plt.axhline(y=0.0, color='k', linestyle='-.')
        plt.legend(fontsize=12)
        plt.xlabel('cycle',fontsize=14)
        plt.ylabel('%',fontsize=14)
        st.pyplot()
        
#____________________________________________________________________________  

if __name__ == "__main__":
    st.subheader('Information\n')
    if st.checkbox('Information'):
        st.markdown("""\n
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Welcome file</title>
              <link rel="stylesheet" href="https://stackedit.io/style.css" />
            </head>
            <body class="stackedit">
            <p><img src="https://img.soccersuck.com/images/2020/07/18/profilbild-removebg-preview11e027b125057deb6666.png" alt="N|Solid"></p><hr>
            <p><em><strong>God_Returns</strong></em>  Data Frame</p>
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>ความหมาย</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Time</td>
            <td>loop_Start ถึง loop_End</td>
            </tr>
            <tr>
            <td>Data</td>
            <td>Markets Symbol จาก Ftx</td>
            </tr>
            <tr>
            <td>TimeFrame</td>
            <td>1H , 4H , 1D , 1W</td>
            </tr>
            <tr>
            <td>Loop_Start</td>
            <td>เริ่ม cycle</td>
            </tr>
            <tr>
            <td>Loop_End</td>
            <td>จบ cycle</td>
            </tr>
            <tr>
            <td>Input F(x)</td>
            <td>ฟังก์ชั่นที่สนใจ</td>
            </tr>
            <tr>
            <td>Length_Param</td>
            <td>Param  ฟังก์ชั่น</td>
            </tr>
            </tbody>
            </table><hr>
            <p><em><strong>God_Returns</strong></em>  Data Frame</p>
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>ความหมาย</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Time</td>
            <td>loop_start ถึง loop_end</td>
            </tr>
            <tr>
            <td>OHLC4</td>
            <td>(Open + High + low + close / 4)  ตัวแทนของราคา</td>
            </tr>
            <tr>
            <td>Mk_Returntime + 1</td>
            <td>(Market Returntime time +1 ) ผลตอบแทนของวันพรุ่งนี้</td>
            </tr>
            <tr>
            <td>God_Buysell + 1</td>
            <td>ผลตอบแทนสูงสุดของ Buy &amp; Sell   ขอของวันพรุ่งนี้</td>
            </tr>
            <tr>
            <td>God_Buyonly + 1</td>
            <td>ผลตอบแทนสูงสุดของ Buy  ของวันพรุ่งนี้</td>
            </tr>
            <tr>
            <td>God_Sellonly + 1</td>
            <td>ผลตอบแทนสูงสุดของ Sell   ของวันพรุ่งนี้</td>
            </tr>
            <tr>
            <td>Cum_Godbuyonly</td>
            <td>Cumulative ผลตอบแทนสะสมของ Buy</td>
            </tr>
            <tr>
            <td>Cum_Godsellonly</td>
            <td>Cumulative ผลตอบแทนสะสมของ Sell</td>
            </tr>
            <tr>
            <td>Cum_Buysell</td>
            <td>Cumulative ผลตอบแทนสะสมของ Buysell</td>
            </tr>
            <tr>
            <td>Cum_Buyhold</td>
            <td>Cumulative  ผลตอบแทนซื้อถือยาว</td>
            </tr>
            </tbody>
            </table><hr>
            </div>	
            </body>
            </html>
            """, unsafe_allow_html=True)
    st.write("_"*45)
    
    st.subheader('Method Isolate\n')
    model =  Run_model()
    st.sidebar.header('Input Parameter\n')
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
    
    model.pair_data =   st.sidebar.selectbox('data' ,('BTC-PERP', 'XRP-PERP'))
    model.timeframe =   st.sidebar.selectbox('timeframe',('1h', '4h' ,'1d' ,'1w'))
    model.loop_start =  np.datetime64(st.sidebar.date_input('loop_start', value= dt.datetime(2020, 7, 10, 0, 0)))
    model.loop_end =    np.datetime64(st.sidebar.date_input('loop_end', value= dt.datetime(2020, 7, 18, 0, 0)))
    
    model.input = selectbox('rsi')
    model.length = st.sidebar.slider('length_parameter' , 1 , 30 , 15)
    st.write("_"*45)
    st.subheader('Details\n')
    if st.checkbox('Details', value = True):
        st.subheader('God Returns\n')
        pyplot = model.god_chart()
        god = model.god_returns()
        st.write(god)
        st.write('Cumulative GodmaxBuy  :',round ((god['Cum_Godbuyonly][-1]),3)*100 , '%',
                 ('Cumulative GodmaxSell :',round ((god['Cum_Godsellonly'][-1]),3)*100, '%' ))
#         st.write('Cumulative GodmaxSell :',round ((god['Cum_Godsellonly'][-1]),3)*100, '%' ))
        st.write('Cumulative GodBuy&Sell:',round ((god['Cum_Buysell'][-1]), 3)*100, '%',
                 ('Cumulative Buyhold    :',round ((god['Cum_Buyhold'][-1]), 3)*100, '% ))
                 
#         st.write('Cumulative Buyhold    :',round ((god['Cum_Buyhold'][-1]), 3)*100, '%' )
     
                st.write("_"*45)
        st.subheader('F(x) Returns\n')
        pyplot = model.fx_scatter()
        pyplot = model.fx_chart()
        fx = model.fx()
        st.write(fx)
        st.write('Cumulative Buy     :' ,round ((fx['F(x)_CumBuyonly'][-1]), 3)*100 , '%',
                 ('Cumulative Sell    :' ,round((fx['F(x)_CumSellonly'][-1]), 3)*100, '%' ))
                                                       
#         st.write('Cumulative Sell    :' ,round((fx['F(x)_CumSellonly'][-1]), 3)*100, '%' )
        st.write('Cumulative Buy&Sell:' , round((fx['F(x)_CumBuySell'][-1]), 3)*100, '%',
                ('Cumulative Buyhold :' , round((fx['F(x)_CumBuyhold'][-1]), 3)*100, '%' ))
#         st.write('Cumulative Buyhold :' , round((fx['F(x)_CumBuyhold'][-1]), 3)*100, '%' )
    
    st.write("_"*45)
    st.subheader('Python coding\n')
    if st.checkbox('python coding', value = 0 ):

        code = """\n
    class Run_god(object) :
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
        def  loop (self):
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
            god_returns['Cum_GodSellonly'] = np.cumsum(god_returns['God_Sellonly'])
            god_returns['Cum_Buysell'] = np.cumsum(god_returns['God_Buyonly'])
            god_returns['Cum_Buyhold']  = np.cumsum(god_returns['Mk_Returntime+1'])
            god_returns = god_returns.iloc[: , -9:]
            return god_returns
        def  fx (self):
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
        def  fx_scatter (self):
            dataset = self.fx()
            dataset['buy'] = dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'buy' , x.OHLC4 , None) , axis=1)
            dataset['sell'] =  dataset.apply(lambda x : np.where(x['F(x)_Action'] == 'sell'  , x.OHLC4 , None) , axis=1)
            plt.figure(figsize=(12,8))
            plt.plot(dataset.OHLC4 , color='k' , alpha=0.20 )
            plt.plot(dataset.buy , 'o',  color='g' , alpha=0.50 )
            plt.plot(dataset.sell , 'o', color='r' , alpha=0.50)      
            plt.show()
        def  fx_chart (self):
            fx_chart = self.fx()
            plt.figure(figsize=(12,8))
            plt.plot(fx_chart['F(x)_CumBuyonly'], color='k',  alpha=0.60 )
            plt.plot(fx_chart['F(x)_CumSellonly'], color='g',  alpha=0.60 )
            plt.plot(fx_chart['F(x)_CumBuySell'], color='r',  alpha=0.60 )
            plt.show()
        def god_chart (self):
            god_chart = self.god_returns()
            plt.figure(figsize=(12,8))
            plt.plot(god_chart['Cum_Godbuyonly'], color='k',  alpha=0.60 )
            plt.plot(god_chart['Cum_Godsellonly'], color='g',  alpha=0.60 )
            plt.plot(god_chart['Cum_Buysell'], color='r',  alpha=0.60 )
            plt.show()
    """
        st.code(code, language='python')

# # st.sidebar.text("_"*45)
# pyplot = model.chart
# pyplot = model.nav
# if st.checkbox('df_plot'):
#     st.write(pyplot.iloc[: , :])
# st.text("")
# st.write('\n\nhttps://github.com/firstnattapon/test-stream/edit/master/app.py')
