import pandas as pd
import numpy as np
#from costatrade import Strategy
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor #十字光标
import matplotlib.ticker as ticker
import time
import os
# Record start time. We will record too end time and culculate total run time at the end of program.
start_time = time.time()
# Create new folders to store outcome files, if they are not exist.
os.makedirs('./long outcome/perf', exist_ok=True)
os.makedirs('./long outcome/trans', exist_ok=True)


"""
This is with_diff's parent class
including __init__ and some definitions 
that would be used in def general_signal to calculate some variable.
可以全部放在同一个类里
分成父子类提高可读性
"""
class Strategy(object):
    # class initializing
    # assign values to object properties
    def __init__(
            self, quote, capital, open_minute, open_threshold, 
            open_continous_threshold, open_withdrawal_threshold, 
            close_minute, close_threshold, close_withdrawal_threshold, commision_percent
            ):
        # init parameters
        self.quote = quote  # quote就是标的
        self.capital = capital  # 起始资金，通常为100
        self.commision_percent = commision_percent  # commision parameter
        # position opening parameters
        self.open_minute = open_minute
        self.open_threshold = open_threshold
        self.open_continous_threshold = open_continous_threshold
        self.open_withdrawal_threshold = open_withdrawal_threshold
        # position closing parameters
        self.close_minute = close_minute
        self.close_threshold = close_threshold
        self.close_withdrawal_threshold = close_withdrawal_threshold
        # These properties equal to the return of self.generate_signals & self.generate_performance
        self.signal, self.signal_history, self.speed_close_count, self.withdrawal_close_count = self.generate_signals()
        self.performance, self.transactions = self.generate_performance(self.commision_percent)

    def get_increase(self, df):
        # input: df(a slice of minute stock data df)
        # output: the increase price points from the lowest point to highest point 
        # during the interval of df.
        if df.empty:
            print('received empty dataframe at get_increase function.')
            pass
        # 加入这个判断是因为有可能出现阴线，但还是将最高价作为high，导致incr_diff计算不符预期
        if df.iloc[0]['open'] >= df.iloc[0]['close']:
            low = df.iloc[0]['low']
            high = df.iloc[0]['high']
        else:
            low = df.iloc[0]['low']
            high = df.iloc[0]['close']
        increase = 0
        for index, row in df.iterrows():
            if row['low'] <= low:
                high = row['close']
                low = row['low']
            elif row['high'] > high:
                high = row['high']
            increase = high - low
        return increase

    def get_analysis_increase(self, df):
        if df.empty:
            print('received empty dataframe at get_increase function.')
            pass
        low = 0
        high = 0
        low = df.iloc[0]['low']
        high = df.iloc[0]['close']
        for index, row in df[1:].iterrows():
            if row['high'] > high:
                high = row['high']
        analysis_increase = high - low
        return analysis_increase

    def get_withdrawal(self, df, assumebarwithdrawal=True):
        with_high = 0
        with_low = 0
        withdrawal = 0
        for index, row in df.iterrows():
            if with_high == 0:
                with_high = row['close']
                with_low = row['close']
                withdrawal = with_high - with_low
            else:
                if row['high'] > with_high:
                    with_high = row['high']
                    with_low = row['close']
                elif row['low'] < with_low:
                    with_low = row['low']
                withdrawal = with_high - with_low
        return withdrawal

    # 这个算法会将回撤平仓的那一根bar的跌幅全部计算进去 因此得数可能会比回撤平仓的设定值高
    def get_max_withdrawal(self, df, assumebarwithdrawal=True):
        with_high = 0
        with_low = 0
        withdrawal = 0
        max_withdrawal = 0
        
        for index, row in df.iterrows():
            if with_high == 0:
                with_high = row['close']
                with_low = row['close']
                withdrawal = with_high - with_low
            else:
                if row['high'] > with_high:
                    with_high = row['high']
                    with_low = row['close']
                elif row['low'] < with_low:
                    with_low = row['low']
                withdrawal = with_high - with_low
            if withdrawal > max_withdrawal:
                max_withdrawal = withdrawal
        return max_withdrawal

'''
We use this class to instantiate historical data.
def generate_signals handle with data.
Then we use def generate_performance to calculate remain capical of our account.
'''

class with_diff(Strategy):  

    # 通过既定的策略生成持仓状态的信号 的方法
    def generate_signals(self):
        
        # Initialize the dataframe
        signal = pd.DataFrame(index=self.quote.index)  # index = self.quote.index即跟underlying一样数量的index
        signal['date'] = np.nan
        signal['withdrawal_data'] = np.nan
        signal['withdrawal_signal'] = 0.0
        signal['increase_data'] = np.nan
        signal['increase_signal'] = 0.0
        signal['ana_inc'] = np.nan
        signal['total_increase_data'] = np.nan
        signal['total_increase_signal'] = 0.0
        signal['max_increase'] = np.nan  # 不开仓的情况下，False部分一次计算的全部涨幅
        signal['max_withdrawal'] = np.nan  # low_index到high_index之间出现过的最大回撤
        signal['holding_withdrawal_data'] = np.nan
        signal['holding_withdrawal_signal'] = 0.0
        signal['holding_increase_data'] = np.nan
        signal['holding_increase_signal'] = 0.0
        signal['variable0'] = np.nan
        signal['low_index'] = np.nan  # 开始计算涨幅的那根k线的位置
        signal['high_index'] = np.nan  # 最高价出现的那根k线的位置
        signal['integer_index'] = np.nan  # 当前k线的位置
        signal['last_index'] = np.nan
        signal['execution_price'] = np.nan  # 执行交易的价格
        signal['period'] = np.nan  # 用于统计 持仓时间
        signal['new_opening_count'] = np.nan  # 用于计算开始计算后经过了多久
        signal['opening_signal'] = 0.0  # 应该叫holding signal才对 资金账户判断持仓与否的信号
        signal['close_type'] = np.nan  # 回撤平仓时value=1，速度平仓时value=2
        # signal['test1'] = 0.0 #测试参数
        
        #Initialize loop parameters
        have_holdings = False  # 初始状态为未持仓状态
        new_opening = False
        new_opening2 = False
        new_opening_count = 0  # 初始状态为未持仓状态
        variable0 = 0  # 初始状态为未持仓状态
        integer_index = 0  # 从第一根k线开始
        withdrawal_close_count = 0  # 回撤平仓次数，从0开始计算
        speed_close_count = 0  # 速度平仓次数，从0开始计算
        last_index = 0
        recent_low_index = 0
        missing_warning = 0
        
        for index, row in signal.iterrows():  # 对每一根k线进行运算
            # 0 if have no holdings, see if can open a position
            # 开仓模块
            if have_holdings == False:

                # 1 给last_index 赋值 这一段好像可以放在最后
                if new_opening:
                    last_index = integer_index - 1  # -1是为了从 判断未不合条件从而reset的那根k线 即 上一次判断的最后一根k线 开始计算
                    # 避免last_index = 0-1 = -1
                    new_opening = False  # reset前不再使last_index = integer_index -1
                if new_opening2:
                    last_index = recent_low_index
                    # 避免last_index = 0-1 = -1
                    new_opening2 = False  # reset前不再使last_index = integer_index -1

                # 2 如果new opening count大于open minute，将计算区域设为为 [last index:last index+open minute]。我哦们计算计算涨幅是否
                if new_opening_count >= self.open_minute:
                    last_index = integer_index - self.open_minute + 1  # +1
                new_opening_count += 1
                analysis_slice = self.quote[last_index:integer_index + 1]
                if variable0 == 0:  # 0是初始值
                    # 3 definitions of increase & withdrawal
                    increase = self.get_increase(analysis_slice)
                    withdrawal = self.get_withdrawal(analysis_slice)
                    signal.at[index, 'withdrawal_data'] = withdrawal
                    signal.at[index, 'increase_data'] = increase

                    # 4 定义cond1(increaser cond)和cond2(withdrawal cond)。满足条件时在该条件的signal的column上标记为1，不满足时标记为0。
                    cond1 = increase >= self.open_threshold # 条件1：increase 大于等于 open_threshold ，即满足开仓的速度
                    if cond1:
                        signal.at[index, 'increase_signal'] = 1
                    else:
                        signal.at[index, 'increase_signal'] = 0
                    cond2 = withdrawal < self.open_withdrawal_threshold # 条件2: if withdrawal is less than open_withdrawal_threshold
                    if cond2:
                        signal.at[index, 'withdrawal_signal'] = 1
                    else:
                        signal.at[index, 'withdrawal_signal'] = 0

                    # 5 如果cond2满足，那么判断一次cond1是否满足。两者都满足时，计算low_index位置，然后令variable0 = 1。之后不再判断cond1。
                    if signal.at[index, 'withdrawal_signal']:
                        if signal.at[index, 'increase_signal']:
                            # 满足cond1和cond2时，计算行情开始的位置 low_index，然后将v0设为1
                            for i in range(last_index, integer_index + 1):  # 循环会在integer_index处停止
                                low_index_slice = self.quote[i: integer_index + 1]
                                increase2 = self.get_increase(low_index_slice)
                                if increase2 == increase:
                                    low_index = i
                                    signal.at[index, 'low_index'] = low_index
                                    break
                                # in case it doesn't works
                                else:
                                    print(integer_index + 'did not find low_index' )
                            last_index = low_index
                            start_index = last_index  # start_index会一直等于last_index不会reset,用于计算high_index
                            variable0 = 1
                    
                    # 6 如果cond2不满足，那么reset
                    else:
                        if increase > self.open_continous_threshold:
                            missing_warning += 1
                            print(missing_warning)
                            print(index)
                        new_opening = True  # 移动last_index和holding_start_index的位置
                        new_opening_count = 1

                # 6 满足cond1和cond2后,对new_opening_count进行一次赋值,使其在当前k线等于integer_index - low_index +1
                if variable0 == 1:
                    new_opening_count = integer_index - low_index + 1
                    signal.at[index, 'low_index'] = low_index
                    variable0 = 2  # 通过将variable0的值设为2来进入 判断涨幅是否足以开仓 的阶段

                # 7 判断是否开仓的标准是 计算总涨幅是否大于open_continous_threshold
                if variable0 == 2:
                    cond3_analysis_slice = self.quote[low_index: integer_index + 1]
                    withdrawal = self.get_withdrawal(cond3_analysis_slice)
                    signal.at[index, 'withdrawal_data'] = withdrawal
                    cond3 = withdrawal < self.open_withdrawal_threshold  # 条件2: if withdrawal is less than open_withdrawal_threshold
                    if cond3:
                        signal.at[index, 'withdrawal_signal'] = 1     
                    else:
                        signal.at[index, 'withdrawal_signal'] = 0                        
                    if signal.at[index, 'withdrawal_signal']:  # 仍要继续判断回撤条件是否满足
                        if new_opening_count >= self.open_minute: 
                            increase_slice = self.quote[last_index: integer_index + 1]
                            analysis_increase = self.get_analysis_increase(increase_slice) # 这个区间的涨幅
                            signal.at[index, 'ana_inc'] = analysis_increase
                            # 如果不满足速度限制，令variable0 = 4，进入统计模块后reset
                            if analysis_increase != 0:  # 可能等于0吗？
                                if analysis_increase < self.open_threshold:
                                    variable0 = 4
                        
                        total_increase = self.get_analysis_increase(cond3_analysis_slice)  # 最低价到最高价
                        signal.at[index, 'total_increase_data'] = total_increase
                        first_cond1_price = cond3_analysis_slice.iloc[0]['low']
                        # 如果涨幅高于限制，就发出开仓信号
                        if total_increase >= self.open_continous_threshold:
                            signal.at[index, 'total_increase_signal'] = 1
                            # 如果不满足回撤限制，进入统计模块后reset
                    else:
                        variable0 = 3

                # 9 处于阶段二（满足了速度要求）时由于回撤超过设定值而reset，v0=3是在reset前得到统计模块。
                if variable0 == 3:
                    increase3_slice = self.quote[start_index: integer_index + 1]
                    increase3 = self.get_increase(increase3_slice)
                    for i in range(start_index + 1, integer_index + 2):  # last_index+1是为了在第一次计算时值为last_index所在位置的值
                        high_index_slice = self.quote[start_index: i]
                        increase4 = self.get_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = self.quote[low_index: high_index + 1]
                    max_withdrawal = self.get_max_withdrawal(max_slice)
                    max_increase = self.get_increase(max_slice)
                    signal.at[index, 'max_increase'] = max_increase
                    signal.at[index, 'max_withdrawal'] = max_withdrawal
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'low_index'] = low_index
                    # End
                    # 统计完成后reset模块
                    new_opening = True
                    variable0 = 0
                    new_opening_count = 1  # 为什么这里是1 下面平仓是0？
                    first_cond1_price = 0
                    analysis_increase = 0

                if variable0 == 4:
                    increase3_slice = self.quote[start_index: integer_index + 1]
                    increase3 = self.get_increase(increase3_slice)
                    for i in range(start_index + 1, integer_index+2):  # last_index+1是为了在第一次计算时值为last_index所在位置的值
                        high_index_slice = self.quote[start_index: i]
                        increase4 = self.get_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = self.quote[low_index: high_index+1]
                    max_withdrawal = self.get_max_withdrawal(max_slice)
                    max_increase = self.get_increase(max_slice)
                    signal.at[index, 'max_increase'] = max_increase
                    signal.at[index, 'max_withdrawal'] = max_withdrawal
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'low_index'] = low_index
                    # End
                    # 统计完成后reset模块
                    for i in range(last_index, integer_index+1):  # 循环会在integer_index处停止
                        low_index_slice = self.quote[i: integer_index+1]
                        increase2 = self.get_increase(low_index_slice)
                        if increase2 == increase:
                            recent_low_index = i
                    last_index = recent_low_index
                    new_opening2 = True
                    variable0 = 0
                    new_opening_count = integer_index - recent_low_index + 1
                    first_cond1_price = 0
                    analysis_increase = 0

                # 10 得到开仓信号后进行开仓，然后转入持仓模块
                if signal.at[index, 'total_increase_signal']:
                    open_execution_price = first_cond1_price + self.open_continous_threshold
                    # 跳空时在开盘价位置开仓
                    if open_execution_price < self.quote.loc[index, 'open']:
                        if open_execution_price > self.quote.loc[index-1, 'close']:
                            open_execution_price = self.quote.loc[index, 'open']
                    # End
                    signal.at[index, 'execution_price'] = open_execution_price
                    new_opening_count = integer_index - low_index
                    signal.at[index, 'opening_signal'] = 1 # open signal
                    variable0 = 0
                    have_holdings = True  # 开仓
                    new_opening = True

        # 持仓模块
            elif have_holdings == True:
                # 1 赋值
                if new_opening:
                    last_index = low_index
                    increase_start_index = low_index
                    holding_start_index = integer_index
                    new_opening = False
                if new_opening_count >= self.close_minute:
                    last_index = integer_index - self.close_minute
                new_opening_count += 1
                analysis_slice = self.quote[last_index + 1:integer_index + 1]
                # 不包括last_index当根bar的 分析区间(持仓中的一段时间)
                holding_slice = self.quote[increase_start_index:integer_index + 1]
                # 包括开仓位当根bar的 持仓区间（从持仓开始到当前的时间）
                if new_opening_count >= self.close_minute:
                    # holding_increase = self.get_increase(analysis_slice)#不能这么算
                    holding_increase = self.get_analysis_increase(analysis_slice)
                    signal.at[index, 'holding_increase_data'] = holding_increase
                # 2 条件1
                # cond1 = False #这条是用来干嘛的？
                if new_opening_count >= self.close_minute:
                    if holding_increase < self.close_threshold:
                        signal.at[index, 'holding_increase_signal'] = 1
                # 3 condition2: withdrawal exceeds close_withdrawal_threshold
                holding_withdrawal = self.get_withdrawal(holding_slice)
                signal.at[index, 'holding_withdrawal_data'] = holding_withdrawal
                if holding_withdrawal > self.close_withdrawal_threshold:
                    signal.at[index, 'holding_withdrawal_signal'] = 1
                # 4 计算持仓时间period
                period = integer_index - holding_start_index + 1

                # 平仓
                # 回撤平仓
                if signal.at[index, 'holding_withdrawal_signal'] == 1:
                    # 执行平仓
                    have_holdings = False
                    signal.at[index, 'opening_signal'] = 0
                    cond2_execution_price = max(holding_slice['high']) - self.close_withdrawal_threshold
                    signal.at[index, 'execution_price'] = cond2_execution_price
                    signal.at[index, 'period'] = period
                    withdrawal_close_count += 1
                    new_opening = True
                    new_opening_count = 0
                    # End
                    # 执行统计
                    increase3_slice = self.quote[start_index: integer_index + 1]
                    increase3 = self.get_increase(increase3_slice)
                    for i in range(start_index + 1, integer_index + 2):
                        high_index_slice = self.quote[start_index: i]
                        increase4 = self.get_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = self.quote[low_index: high_index + 1]
                    max_withdrawal = self.get_max_withdrawal(max_slice)
                    max_inc = self.get_analysis_increase(max_slice)
                    signal.at[index, 'max_increase'] = max_inc
                    signal.at[index, 'max_withdrawal'] = max_withdrawal
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'low_index'] = low_index
                    signal.at[index, 'close_type'] = 1
                    # End
                # End

                # 速度平仓
                elif signal.at[index, 'holding_increase_signal'] == 1:
                    # 执行平仓
                    have_holdings = False
                    signal.at[index, 'opening_signal'] = 0
                    signal.at[index, 'execution_price'] = self.quote.loc[index]['close']
                    signal.at[index, 'period'] = period
                    speed_close_count += 1
                    new_opening = True
                    new_opening_count = 0
                    # End
                    # 执行统计
                    increase3_slice = self.quote[start_index: integer_index + 1]
                    increase3 = self.get_increase(increase3_slice)
                    for i in range(start_index + 1, integer_index + 2):
                        # index+1是为了在第一次计算时值为last_index所在位置的值
                        high_index_slice = self.quote[start_index: i]
                        increase4 = self.get_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = self.quote[low_index: high_index + 1]
                    max_withdrawal = self.get_withdrawal(max_slice)
                    max_inc = self.get_analysis_increase(max_slice)
                    signal.at[index, 'max_increase'] = max_inc
                    signal.at[index, 'max_withdrawal'] = max_withdrawal
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'low_index'] = low_index
                    signal.at[index, 'close_type'] = 2
                    # End
                # End

                else:
                    signal.at[index, 'opening_signal'] = 3 # holding signal
                    
            # 6 不管有没有持仓都set_value和values的variable
            signal.at[index, 'integer_index'] = integer_index
            signal.at[index, 'last_index'] = last_index
            signal.at[index, 'new_opening_count'] = new_opening_count
            
            # 7 全部结束后，标注当前日期，进入下一根index的计算。
            # todays_dateperiod = self.quote[integer_index:integer_index + 1]
            todaysdate = self.quote.iat[integer_index,0]
            signal.at[index, 'date'] = todaysdate
            
            integer_index += 1
            
            #print(integer_index) # print出进度 会降低计算速度
            
        # 建立一个名叫df_signal的DataFrame，return给self.signal
        df_signal = pd.DataFrame({
            'date': signal.date, 
            'signal': signal.opening_signal, 
            'execution_price': signal.execution_price,
             'close_type': signal.close_type
             })
        # 25行 self.signal, self.signal_history, self.speed_close_count, self.withdrawal_close_count  = self.generate_signals()
        return df_signal, signal, speed_close_count, withdrawal_close_count

    # 计算账户剩余资金的函数
    def generate_performance(self, commision_percent):  
        """
        when signal = 0.0, it means no holdings
        when signal = 1.0, it means a longing signal
        when signal = 2.0, it means a shorting signal
        when signal = 3.0, it means a holding signal
        calculate performance based on signals
        exposure is used for calculating performance when we hold a shorting position
        """
        starting_capital = self.capital
        self.signal['capital'] = 0.0  # self.signal就是df_signal 374行
        transactions_df = pd.DataFrame(columns=[
            'Date', 'Type', 'Price', 'Close_type', 'Capital', 'Percent'])  # 最后会输出为trans
        type = None
        cost = None
        for index, row in self.signal.iterrows():  # self.signal就是
        
            # if have no holdings
            if row['signal'] == 0.0:
                # two situation: 1.  when you don't have positions at all.  2.
                # when you are closing a position (from 3 to 0)
                if type == None:
                    
                    row['capital'] = starting_capital
                elif type == 'long':
                    # close a long position
                    percent = row['execution_price'] / cost
                    starting_capital = starting_capital * percent  # * (1 - commision_percent) 买入不用手续费
                    row['capital'] = starting_capital
                    # reset type
                    type = None
                    transactions_df.loc[index] = [row['date'], 'sell', row['execution_price'], row['close_type'],
                                                  row['capital'], percent]

            # if you have longing signal
            elif row['signal'] == 1.0:
                # assume that there is no bid-ask spread cost
                # record cost and types of transaction
                starting_capital = starting_capital * (1 - commision_percent)
                row['capital'] = starting_capital
                cost = row['execution_price']
                # cost = self.quote.close[index]
                type = 'long'
                transactions_df.loc[index] = [row['date'], type, cost, "", "", "" ]
            
            # if have a holding
            elif row['signal'] == 3.0:

                # if it is a long holding, update capital by multiply capital
                # with (1 + percent change) in price
                if type == 'long':
                    percent = self.quote.close[index] / cost
                    temp = starting_capital * percent
                    row['capital'] = temp
                '''
                # opposite direction (1 - percent change)
                elif type == 'short':
                    percent = 1 - ((self.quote.close[index] - cost) / cost)
                    temp = starting_capital * percent
                    row['capital'] = temp
                '''
        return self.signal, transactions_df




'''
Select a data file 
Turn the data file into dataframe
Select a part of dataframe for our backtest
'''
def read_data(filename):  
    df = pd.read_csv(filename, names=['Date', 'open', 'high', 'low', 'close'])
    return df




df = read_data('20s OHLC result.txt')# 读取数据文件并转换为dataframe
# 回测的时间区间
startdate = 00000
enddate = 20000
df5  = df[startdate : enddate]
underlying = df5




'''
输入参数&实例化
'''
step1 = 5
step2 = 0.5
for_num_1 = 1
for_num_2 = 1
plot_df = pd.DataFrame()#用来放每一次循环的最后结果 然后plot出来
# A double Loop 
for num in range(for_num_1):
    for i in range(for_num_2):
    # Parameters
        print(str(num) +' '+ str(i))
        print('\n')
        # strategy parameters
        open_minute = close_minute = 40 + (i*step1)
        open_threshold = close_threshold = 1
        open_continous_threshold = 3
        open_withdrawal_threshold = close_withdrawal_threshold = 1 + (num * step2)
        commision_percent = 0.0001
        capital = 100.0  # 初始资本
        # 实例化
        strategy = with_diff(underlying, capital, \
                             open_minute, open_threshold, open_continous_threshold, open_withdrawal_threshold, \
                             close_minute, close_threshold, close_withdrawal_threshold,
                             commision_percent) 
        # 至此，策略部分结束。以下部分是控制结果如何输出（图与文件）



        performance = strategy.performance.reset_index(drop=True)  #reset index number
        del performance['signal']
        del performance['execution_price']
        del performance['close_type']
        del performance['date']
        underlying1 = underlying.copy()
        underlying1 = underlying1.reset_index(drop=True)
        print('total close count = ' + str(strategy.withdrawal_close_count + strategy.speed_close_count))
        print('withdrawal close count = ' + str(strategy.withdrawal_close_count))
        print('speed close count = ' + str(strategy.speed_close_count))
        print(str(startdate) + '-' + str(enddate) + ' ' + str(strategy.open_minute) + ' ' + str(strategy.open_threshold)
              + ' ' + str(strategy.open_continous_threshold) + ' ' + str(strategy.open_withdrawal_threshold) 
              + ' ' + str(strategy.close_minute) + ' ' + str(strategy.close_threshold) + ' ' 
              + str(strategy.withdrawal_close_count) + '+' + str(strategy.speed_close_count))

    # Plot
        factor = underlying1['open'][0] / 100 # 以第一个open为基准
        x = underlying1['close'].copy() / factor# 行情的变动比例
        fig = plt.figure(figsize=(19, 9.8))
        #size
        left = 0.043
        width = 0.943
        bottom = 0.07
        height = 0.9
        rect_line = [left, bottom, width, height] # below parameter
        #ax
        ax = fig.add_axes(rect_line)
        date_list = underlying1.Date.to_list()
        b = [ str(i) for i in date_list ]
        xaxis1 = b
        yaxis1 = performance
        xaxis2 = x.index
        yaxis2 = x
        ax.grid(b = True, linestyle = 'dashed', color='#DBD8D8')
        ax.xaxis.set_major_locator(ticker.LinearLocator(30))
        ax.yaxis.set_major_locator(ticker.LinearLocator(10))
        Plot1 = ax.plot(xaxis1, yaxis1, linewidth=1.2)
        Plot2 = ax.plot(xaxis2, yaxis2, linewidth=1.2)
        #locs, labels=plt.xticks()
        #plt.xticks(locs, b, rotation=30, horizontalalignment='right')
        cursor = Cursor(ax, useblit=True, color='red', linewidth=0.7)# 十字光标
        #remove spot's frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #show and save as pdf
        degrees=45# x轴日期偏斜的角度
        plt.xticks(rotation=degrees)
        plt.title( '%s' % ( ' a' + str(strategy.open_minute) 
                    + ' b' + str(strategy.open_threshold) + ' c' + str(strategy.open_continous_threshold) 
                    + ' d' + str(strategy.open_withdrawal_threshold) + ' ' + str(strategy.withdrawal_close_count) 
                    + '+' + str(strategy.speed_close_count) + ' ' + str(startdate) + '-' + str(enddate) ) )
        plt.show()
        plt.savefig('long outcome/' + ' a' + str(strategy.open_minute) 
                    + ' b' + str(strategy.open_threshold) + ' c' + str(strategy.open_continous_threshold) 
                    + ' d' + str(strategy.open_withdrawal_threshold) + ' ' + str(strategy.withdrawal_close_count) 
                    + '+' + str(strategy.speed_close_count) + ' ' + str(startdate) + '-' + str(enddate)+ ' Long.pdf', dpi=1000)

    # Perf
        # 建立一个新的dataframe，用于最后输出为perf.xlsx(统计文档)
        perf_csv = pd.concat([strategy.signal_history, df5], axis=1, join='inner')  # signal_history在25行
        perf_csv.drop(
            ['opening_signal', 'increase_signal', 'withdrawal_signal', 'holding_withdrawal_signal', 'total_increase_signal',
             'holding_increase_signal'], axis=1, inplace=True)  # signal不用时可以drop掉
        perf_csv.drop(['variable0'], axis=1, inplace=True)  # 在使用不开仓的统计时,drop掉开仓才能用上的column
        # 在使用不开仓的统计时,drop掉开仓才能用上的column
        # perf_csv.drop(['holding_withdrawal','holding_increase','execution_price','period'], axis =1, inplace = True)
        # perf_csv.drop(['arrive_time'], axis =1, inplace = True)
    
        # 将perf_csv输出perf
        writer1 = pd.ExcelWriter(
            'long outcome/perf/' + 'a' + str(strategy.open_minute) + ' b' + str(strategy.open_threshold) \
            + ' c' + str(strategy.open_continous_threshold) + ' d' + str(strategy.open_withdrawal_threshold) \
            + ' ' + str(strategy.withdrawal_close_count) + '+' + str(strategy.speed_close_count) + ' ' \
            + 'Long ' + str(startdate) + '-' + str(enddate) + ' ' + 'perf.xlsx', engine='xlsxwriter')
        perf_csv.to_excel(writer1, sheet_name='stats')
        
        # improving the appearence of perf_stats.xlsx
        workbook = writer1.book
        worksheet = writer1.sheets['stats']
        worksheet.set_default_row(20)
        worksheet.autofilter('A1:Z1')
        format = workbook.add_format()
        format.set_font_name('Microsoft YaHei UI Light')
        format.set_align('justify')
        format.set_align('center')
        format.set_align('vjustify')
        format.set_align('vcenter')
        format.set_font_size(12)
        format1 = workbook.add_format({'num_format': '0'})
        format1.set_align('justify')
        format1.set_align('center')
        format1.set_align('vjustify')
        format1.set_align('vcenter')
        worksheet.set_column('A:A', 11, format)
        worksheet.set_column('B:B', 13, format1)
        worksheet.set_column('C:C', 10, format)
        worksheet.set_column('D:D', 10.5, format)
        worksheet.set_column('E:E', 14, format)
        worksheet.set_column('F:F', 13.5, format)
        worksheet.set_column('G:G', 13, format)
        worksheet.set_column('H:H', 11, format)
        worksheet.set_column('I:I', 14, format)
        worksheet.set_column('J:J', 13, format)
        worksheet.set_column('K:K', 12, format)
        worksheet.set_column('L:L', 14, format)
        worksheet.set_column('M:Q', 8, format)
        worksheet.set_column('S:W', 10, format)
        # worksheet.set_column('O:O', 16, format)
        worksheet.freeze_panes(1, 2)
        writer1.save()
        # End
    
    # trans.xlsx
        writer2 = pd.ExcelWriter(
            'long outcome/trans/' + 'a' + str(strategy.open_minute) + ' b' + str(strategy.open_threshold) \
            + ' c' + str(strategy.open_continous_threshold) + ' d' + str(strategy.open_withdrawal_threshold) \
            + ' ' + str(strategy.withdrawal_close_count) + '+' + str(strategy.speed_close_count) + ' ' \
            + 'Long ' + str(startdate) + '-' + str(enddate)+ 'trans.xlsx', engine='xlsxwriter')
        strategy.transactions.to_excel(writer2, sheet_name='stats')  # 26行

    # improving the appearence of perf_stats.xlsx
        workbook2 = writer2.book
        worksheet2 = writer2.sheets['stats']
        worksheet2.set_default_row(21)
        
        format3 = workbook2.add_format()
        format3.set_num_format('0')
        format3.set_font_name('Microsoft YaHei UI Light')
        format3.set_align('justify')
        format3.set_align('center')
        format3.set_align('vjustify')
        format3.set_align('vcenter')
        worksheet2.set_column('B:B', 17, format3)
        
        format2 = workbook2.add_format()
        format2.set_font_name('Microsoft YaHei UI Light')
        format2.set_align('justify')
        format2.set_align('center')
        format2.set_align('vjustify')
        format2.set_align('vcenter')
        format2.set_font_size(12)
        worksheet2.set_column('A:A', 11, format2)
        worksheet2.set_column('C:D', 11, format2)
        worksheet2.set_column('E:E', 14, format2)
        worksheet2.set_column('F:G', 13, format2)
        # worksheet2.freeze_panes(1, 1)
    # End
        writer2.save()


        perf_temp = performance[-1:]
        plot_df = pd.concat([plot_df,perf_temp], ignore_index=True)

# 如果是计算多个结果，那么将每个结果都plot到同一个折线图上
if for_num_2 > 1:
    fig2 = plt.figure(figsize=(48, 20))
    plt.plot(plot_df)
    fig2.show()
    plt.savefig('long outcome/' +  str(for_num_1)+ ' ' +  str(for_num_1)+' outcome.pdf', dpi=1000)
print("time = --- %s seconds ---" % (time.time() - start_time))  # 显示总运算时间
