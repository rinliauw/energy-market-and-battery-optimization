import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

# IMPORT BATTERY CLASS
from Battery import Battery

### GLOBAL VARIABLE ###
EMPTY = ' ' # EMPTY STRING
PERIOD = 48 # PERIOD IN A DAY (CHANGE THIS IF PERIOD IN A DAY IS NOT 48)

######################### HELPER FUNCTIONS #########################
def GetSpotPrice(data, selected_periods = False, region = 'VIC'):
    '''
    A function to get spot prices based on selected regions.

    Parameters
    ----------
        - data : Pandas Dataframe 
            The targeted dataset
        - selected_periods : list
            Selected period for charging or discharging.
        - region : str
            The targeted region, default has been set to 'VIC' for mandatory task.
    
    Returns
    -------
        - spot_prices : list
            If selected_periods is not given, just return the spot_price 
            of the dataframe.

        - retrieved_prices : list
            If selected_periods is given, return spot_price of the dataframe 
            with its respective ranking.
    
    Created by: Gilbert, P.
    '''
    if region == 'VIC':
        spot_price = data['Regions VIC Trading Price ($/MWh)']
    elif region == 'NSW':
        spot_price = data['Regions NSW Trading Price ($/MWh)']
    elif region == 'SA':
        spot_price = data['Regions SA Trading Price ($/MWh)']
    elif region == 'TAS':
        spot_price = data['Regions TAS Trading Price ($/MWh)']
    
    spot_price = np.array(spot_price)
    
    # Find the spot prices from selected region. Periods are
    # index + 1, therefore to use the index we need to subtract
    # it by 1.
    if (selected_periods):
        retrieved_prices = []
        for period in selected_periods:
            # append(minimum or maximum ranking, spot_price[index])
            retrieved_prices.append((period[1], spot_price[period[1] - 1]))
        return retrieved_prices
    else:
        return spot_price

########################################################################################

def GetMinMax(data, region = 'VIC', buy_threshold = 5, sell_threshold = 4):
    '''
    A function to find minimum and maximum point rank given threshold.

    Parameters
    ----------
        - data : Pandas Dataframe 
            The targeted dataset
        - region : str
            The targeted region, default has been set to 'VIC' for mandatory task.
        - buy_threshold : list
            maximum number of buying point, default has been set to optimise Checkpoint 3.
        - sell_threshold : list
            maximum number of selling point, default has been set to optimise Checkpoint 3.
    
    Returns
    -------
        - selected_min_price : list
            Selected minimum point

        - selected_max_price : list
            Selected maximum point

    Efficiency : O(3N + NLogN)

    Created by: Gilbert, P.
    '''
    if region == 'VIC':
        spot_price = data['Regions VIC Trading Price ($/MWh)']
    elif region == 'NSW':
        spot_price = data['Regions NSW Trading Price ($/MWh)']
    elif region == 'SA':
        spot_price = data['Regions SA Trading Price ($/MWh)']
    elif region == 'TAS':
        spot_price = data['Regions TAS Trading Price ($/MWh)']
    
    price = np.array(spot_price)
    minimum_price = np.argsort(price, kind = 'merge*sort') # (O(NlogN)), mergesort the minimum prices.
    maximum_price = minimum_price[::-1][:len(price)] # (O(N)), maximum is the reverse order of minimum.
    
    selected_min_price = [EMPTY for i in minimum_price] # (O(N)), set an empty array for the whole period.
    selected_max_price = [EMPTY for i in minimum_price] # (O(N)), set an empty array for the whole period.
    
    # Select the lowest price spot over the given
    # buy_threshold as the minimum buying point.
    i = 0
    for b_t in range(buy_threshold):
        selected_min_price[minimum_price[i]] = b_t + 1
        i += 1
        
    # Select the highest price spot over the given
    # sell_threshold as the maximum selling point.
    i = 0
    for s_t in range(sell_threshold):
        selected_max_price[maximum_price[i]] = s_t + 1
        i += 1
        
    return selected_min_price, selected_max_price

########################################################################################

def FindBatteryPairs(buy_period, sell_period):
    '''
    A function to Find Battery Charge and Discharge pairs in backward order.
    Backward order from 48th period to the 1st.

    Parameters
    ----------
        - buy_period : list
            Selected minimum price point as it will be where we buy energy for charging.
        - sell_period : list
            Selected maximum price point as it will be where we sell energy for discharging.
    
    Return
    ------
        - battery : list
            list of selling point where we are discharging and list of buying point where
            we are charging.

    Efficiency: O(N)

    Created by: Gilbert, P.     
    '''
    MAX_SELL_PERIOD = 4 # MAXIMUM SELLING PERIOD PER PAIR
    MAX_BUY_PERIOD = 5 # MAXIMUM BUYING PERIOD PER PAIR
    
    period = len(buy_period)
    
    battery = []
    sell = OrderedDict() # Initialise battery selling point. (Ordered Dictionary)
    buy = OrderedDict() # Initialise battery buying point. (Orderered Dictionary)
    
    # Iterate over the whole period backwards
    for p in range(period - 1, -1, -1):
        # If maximum selling point is not empty, add (order, period)
        # as key-value pair into the OrderedDict.
        if sell_period[p] != EMPTY and buy_period[p] == EMPTY:
            # If battery buying point period is less MAXIMUM SELLING 
            # PERIOD PER PAIR, add new period.
            if len(sell) < MAX_SELL_PERIOD:
                sell[sell_period[p]] = sell_period.index(sell_period[p]) + 1
            # else, if battery selling point is full and there is 
            # higher maximum selling point then remove the lowest
            # selling point and add the new one into Dictionary.    
            else:
                max_key = max(sell, key=int)
                if sell_period[p] < max_key:
                    sell.pop(max_key)
                    sell[sell_period[p]] = sell_period.index(sell_period[p]) + 1

        # If battery selling point is not empty and minimum buying 
        # point is not empty.
        if len(sell) != 0 and sell_period[p] == EMPTY and buy_period[p] != EMPTY :
            # If battery buying point period is less MAXIMUM BUYING 
            # PERIOD PER PAIR, add new period.
            if len(buy) < MAX_BUY_PERIOD and len(buy) < math.ceil(len(sell) * 1.25):
                buy[buy_period[p]] = buy_period.index(buy_period[p]) + 1
            # else, if battery buying point is full and there is 
            # lower minimum buying point then remove the highest
            # buying point and add the new one into Dictionary.
            else:
                max_key = max(buy, key=int)
                if buy_period[p] < max_key:
                    buy.pop(max_key)
                    buy[buy_period[p]] = buy_period.index(buy_period[p]) + 1
        # If the next period is not empty and battery buying point
        # is not empty then battery charge-discharge pair has been
        # created.
        # Reinitialise a new battery setup.
        if sell_period[p - 1] != EMPTY and len(buy) != 0:
            battery.append([list(sell.items()), list(buy.items())])
            sell = OrderedDict()
            buy = OrderedDict()
    # Add the last battery charge-discharge pair occuring 
    # before 1st period.
    battery.append([list(sell.items()), list(buy.items())]) 

    # Check whether there is too many selling points, then
    # remove selling point until the number of selling points
    # is equal to the number of buying points while removing
    # the lowest selling point.
    for b in battery:
        sell_tmp = np.array(b[0])
        buy_tmp = b[1]
        while len(sell_tmp) > len(buy_tmp): 
            row = 0
            index = np.where(sell_tmp[:,0] == sell_tmp[:,0].max())[0][0]
            sell_tmp = np.delete(sell_tmp, index, axis = row)
        b[0] = sell_tmp.tolist() # Change numpy array to list

    return battery

########################################################################################

def SetChargeDischarge(data, battery_pairs, region = "VIC"):  
    '''
    A function to set optimal charge and discharge amount of battery pairs 
    while setting it into `Battery Class`.

    Parameters
    ----------
        - data : pandas dataframe
            The targeted dataset.
        - battery_pairs : list
            List of selling point where we are discharging and list of buying point where
            we are charging.
    
    Returns
    -------
        - battery_pairs: list 
            List containing Battery Class which are optimised.

    Efficiency: O(N)

    Created by: Gilbert, P.
    '''  
    all_batteries = []
    
    battery_pairs = battery_pairs[::-1]
    for b in battery_pairs:
        sell_period = b[0][::-1] # Reverse the order
        buy_period = b[1][::-1] # Reverse the order
        if len(sell_period)!= 0 or len(buy_period) != 0:
            sell_price = GetSpotPrice(data, sell_period, region = region)
            buy_price = GetSpotPrice(data, buy_period, region = region)

            battery = Battery(buy_period, buy_price, sell_period, sell_price)
            battery.Setting() # SET THE BATTERY SETTINGS
            battery.FirstOptimisation() # FIRST OPTIMISATION
            battery.SecondOptimisation() # SECOND OPTIMISATION

            all_batteries.append(battery)

    return all_batteries

########################################################################################

def ComputeDailyRevenue(all_batteries):
    '''
    A function to calculate daily revenue.

    Parameters
    ----------
        - all_batteries : list
            List of Battery Class pairs
    
    Returns
    -------
        - revenues : float
            Revenues of the respective set of batteries.
    
    Created by : Gilbert, P.
    '''
    revenues = 0
    for battery in all_batteries:
        revenues += battery.Revenue()
    return revenues

########################################################################################

###################################
# A function to optimise battery charging and discharging period. This is where 
# mainly the optimisations are performed with the helper functions.
#
# Parameters:
#      - daily_data : the targeted daily dataset, minimum dataset length of 48.
#
# Return:
#      - raw_power : List of Daily Raw Power for charging and discharging
#      - market_dispatch : List of Daily Market Dispatch for charging and discharging
#      - opening_capacity : List of Daily Opening Capacity for charging and discharging
#      - closing_capacity : List of Daily Closing Capacity for charging and discharging
#
# Efficiency: O(N^2 * NLogN) = O(N^3LogN)
# Created by: Gilbert
###################################
def PeriodOptimisation(given_data, region = "VIC"):
    best_batteries = OrderedDict() # Initialise an Ordered Dictionary
    
    # Iterate over all possible combinations of battery pairs. (O(0.5N^2))
    period = len(given_data)

    for s in range(1, period + 1):
        for b in range(1 , period + 1 - s): 
            # Get the minimum and maximum price based on the given threshold
            min_price, max_price = GetMinMax(given_data, buy_threshold = b, sell_threshold = s, region = region)
            # Get the battery pairs based on minimum and maximum price
            battery_pairs = FindBatteryPairs(min_price, max_price)
            # Get battery optimisation for the selected threshold
            all_batteries = SetChargeDischarge(given_data, battery_pairs, region = region)
            # Compute daily revenues of selected battery combinations
            dailyrev = ComputeDailyRevenue(all_batteries)

            # If revenue negative, skip.
            if dailyrev < 0:
                break

            # Insert revenue as key, batteries combination and threshold as value
            if dailyrev not in best_batteries:
                best_batteries[dailyrev] = (all_batteries, (b, s))
                
    # Find the highest revenue amongst possible combinations in that day
    best_revenue = max(best_batteries)

    # The Best battery combinations
    battery = best_batteries[best_revenue][0] 
    
    # Initialise raw_power, market_dispatch, opening_capacity and closing capacity
    raw_power = [0 for i in range(period)]
    market_dispatch = [0 for i in range(period)]
    opening_capacity = [0 for i in range(period)]
    closing_capacity = [0 for i in range(period)]
    
    # Iterate over battery combinations to set raw_power, market_dispatch,
    # opening_capacity, closing_capacity into an array to be prepared for 
    # merging with the dataset.
    for b in battery:   
        # Charging Period
        for cp in range(len(b.charge_period)):
            raw_power[b.charge_period[cp][1] - 1] = b.charge_raw_power[cp]
            market_dispatch[b.charge_period[cp][1] - 1] = b.charge_market_dispatch[cp]
            opening_capacity[b.charge_period[cp][1] - 1] = b.charge_capacity[cp][0]
            closing_capacity[b.charge_period[cp][1] - 1] = b.charge_capacity[cp][1]
        # Discharge Period
        for dp in range(len(b.discharge_period)):
            raw_power[b.discharge_period[dp][1] - 1] = b.discharge_raw_power[dp]
            market_dispatch[b.discharge_period[dp][1] - 1] = b.discharge_market_dispatch[dp]
            opening_capacity[b.discharge_period[dp][1] - 1] = b.discharge_capacity[dp][0]
            closing_capacity[b.discharge_period[dp][1] - 1] = b.discharge_capacity[dp][1]
    
    # Formatting the opening and closing capacity.
    for i in range(1, len(opening_capacity)):
        if closing_capacity[i - 1] != 0.0 and opening_capacity[i] == 0.0:
            opening_capacity[i] = closing_capacity[i - 1]
            closing_capacity[i] = opening_capacity[i]
                
    return raw_power, market_dispatch, opening_capacity, closing_capacity, best_revenue

# A Main function to run the algorithm.
# FixedOptimisation : This algorithm generates over a period given that there is no dependency.
def FixedOptimisation(data, period = 48, region = 'VIC'):
    raw_power = []
    market_dispatch = []
    opening_capacity = []
    closing_capacity = []

    start = 0
    end = period 
    
    # Given the data iterate over the given period.
    while end <= len(data):
        tmp_data = data.iloc[start:end, :]
        daily_rp, daily_md, daily_oc, daily_cc, _ = PeriodOptimisation(tmp_data, region = region) # Optimise the period given
        raw_power.extend(daily_rp)
        market_dispatch.extend(daily_md)
        opening_capacity.extend(daily_oc)
        closing_capacity.extend(daily_cc)

        start += period
        end += period

    data['Raw Power (MW)'] = pd.Series(raw_power)
    data['Market Dispatch (MWh)'] = pd.Series(market_dispatch)
    data['Opening Capacity (MWh)'] = pd.Series(opening_capacity)
    data['Closing Capacity (MWh)'] = pd.Series(closing_capacity)
    
    return data


def setStatus(dataframe):
    status = []
    for i in range(len(dataframe)):
        market_d = dataframe.loc[i, 'Market Dispatch (MWh)']
        opening_c = dataframe.loc[i, 'Opening Capacity (MWh)']
        closing_c = dataframe.loc[i, 'Closing Capacity (MWh)']
        
        if market_d > 0:
            status.append('Discharge')
        elif market_d < 0:
            status.append('Charge')
        elif market_d == 0 and (opening_c != 0 or closing_c != 0):
            status.append('Between')
        else:
            status.append('Nothing')
    return status

def selectBetweenPeriod(dataframe):
    betweenIndex = dataframe[dataframe['Status'] == 'Between'].index
    period = []
    tmp = [betweenIndex[0]]
    for i in range(1, len(betweenIndex)):
        prev = betweenIndex[i - 1]
        curr = betweenIndex[i]
        if curr - prev > 1:
            period.append(tmp)
            tmp = [curr]
        else:
            tmp.append(curr)
            
    period.append(tmp)
    
    tmp = []
    for p in range(len(period)):
        if len(period[p]) >= 2:
            tmp.append([period[p][0], period[p][-1]])
    period = tmp
    
    timePeriod = []
    for p in period:
        timePeriod.append((dataframe.iloc[p[0], 0], dataframe.iloc[p[1], 0]))
    
    return timePeriod

def GetCapacity(data, selected_periods = False):
    opening_cap = np.array(data['Opening Capacity (MWh)'])
    closing_cap = np.array(data['Closing Capacity (MWh)'])
    
    if (selected_periods):
        retrieved_cap = []
        for period in selected_periods:
            retrieved_cap.append([opening_cap[period[1] - 1], closing_cap[period[1] - 1]])
        return retrieved_cap
    else:
        return [opening_cap, closing_cap]
    
def findBatteryPairsReverse(buy_period, sell_period, dataframe):
    MAX_SELL_PERIOD = 4 # MAXIMUM SELLING PERIOD PER PAIR
    MAX_BUY_PERIOD = 5 # MAXIMUM BUYING PERIOD PER PAIR
    
    period = len(buy_period)
    
    cap = GetCapacity(dataframe)
    max_cap = max(cap[0])
    #print(max_cap,max_cap//150 + 1)
    battery = []
    sell = OrderedDict() # Initialise battery selling point. (Ordered Dictionary)
    buy = OrderedDict() # Initialise battery buying point. (Orderered Dictionary)
    
    for p in range(period - 1):
        if sell_period[p] != EMPTY and buy_period[p] == EMPTY:
            if len(sell) < MAX_SELL_PERIOD and len(sell) < max_cap // 150 + 1:
                #print("YES")
                sell[sell_period[p]] = sell_period.index(sell_period[p]) + 1
            else:
                max_key = max(sell, key=int)
                if sell_period[p] < max_key:
                    sell.pop(max_key)
                    sell[sell_period[p]] = sell_period.index(sell_period[p]) + 1

        if len(sell) != 0 and sell_period[p] == EMPTY and buy_period[p] != EMPTY:
            if len(buy) < MAX_BUY_PERIOD and len(buy) < math.ceil(len(sell) * 1.25) and len(buy) < max_cap // 135 + 1:
                buy[buy_period[p]] = buy_period.index(buy_period[p]) + 1
            else:
                max_key = max(buy, key=int)
                if buy_period[p] < max_key:
                    buy.pop(max_key)
                    buy[buy_period[p]] = buy_period.index(buy_period[p]) + 1
        if sell_period[p + 1] != EMPTY and len(buy) != 0:
                battery.append([list(sell.items()), list(buy.items())])
                sell = OrderedDict()
                buy = OrderedDict()
    battery.append([list(sell.items()), list(buy.items())]) 

    for b in battery:
        sell_tmp = np.array(b[0])
        buy_tmp = b[1]
        while len(sell_tmp) > len(buy_tmp): 
            row = 0
            index = np.where(sell_tmp[:,0] == sell_tmp[:,0].max())[0][0]
            sell_tmp = np.delete(sell_tmp, index, axis = row)
        b[0] = sell_tmp.tolist() # Change numpy array to list
        if len(b[0]) == 0 or len(b[1]) == 0:
            battery.remove(b)

    return battery

def SetChargeDischargeReverse(data, battery_pairs, region = "VIC"):
    opening_cap = data['Opening Capacity (MWh)']
    closing_cap = data['Closing Capacity (MWh)']
    
    all_batteries = []
    
    for b in battery_pairs:
        sell_period = b[0] 
        buy_period = b[1] 
        if len(sell_period)!= 0 or len(buy_period) != 0:
            sell_price = GetSpotPrice(data, sell_period, region = region)
            buy_price = GetSpotPrice(data, buy_period, region = region)
            battery = Battery(buy_period, buy_price, sell_period, sell_price)

            sell_cap = GetCapacity(data, sell_period)
            buy_cap = GetCapacity(data, buy_period)
            battery.ReverseSetting(sell_cap, buy_cap)
            battery.SecondOptimisation()

            all_batteries.append(battery)

    return all_batteries

def localOptimisation(dataframe):
    period = len(dataframe)
    best_batteries = OrderedDict()
    
    for s in range(1, len(dataframe) + 1):
        for b in range(1, len(dataframe) + 1 - s):
            min_price, max_price = GetMinMax(dataframe, buy_threshold = b, sell_threshold = s)
            battery_pairs = findBatteryPairsReverse(min_price, max_price, dataframe)
            all_batteries = SetChargeDischargeReverse(dataframe, battery_pairs)
            dailyrev = ComputeDailyRevenue(all_batteries)
            if dailyrev not in best_batteries:
                best_batteries[dailyrev] = (all_batteries, (b, s))

    # Find the highest revenue amongst possible combinations in that day
    best_revenue = max(best_batteries)
    best_threshold = best_batteries[best_revenue][1]
    battery = best_batteries[best_revenue][0] # The Best battery combinations

    # Initialise raw_power, market_dispatch, opening_capacity and closing capacity
    raw_power = [0 for i in range(period)]
    market_dispatch = [0 for i in range(period)]
    opening_capacity = dataframe.loc[:, 'Opening Capacity (MWh)'].to_numpy()
    closing_capacity = dataframe.loc[:, 'Closing Capacity (MWh)'].to_numpy()
    
    # Iterate over battery combinations to set raw_power, market_dispatch,
    # opening_capacity, closing_capacity into an array to be prepared for 
    # merging with the dataset.
    for b in battery:   
        # Charging Period
        for cp in range(len(b.charge_period)):
            raw_power[b.charge_period[cp][1] - 1] = b.charge_raw_power[cp]
            market_dispatch[b.charge_period[cp][1] - 1] = b.charge_market_dispatch[cp]
            opening_capacity[b.charge_period[cp][1] - 1] = b.charge_capacity[cp][0]
            closing_capacity[b.charge_period[cp][1] - 1] = b.charge_capacity[cp][1]
        # Discharge Period
        for dp in range(len(b.discharge_period)):
            raw_power[b.discharge_period[dp][1] - 1] = b.discharge_raw_power[dp]
            market_dispatch[b.discharge_period[dp][1] - 1] = b.discharge_market_dispatch[dp]
            opening_capacity[b.discharge_period[dp][1] - 1] = b.discharge_capacity[dp][0]
            closing_capacity[b.discharge_period[dp][1] - 1] = b.discharge_capacity[dp][1]
            
       
    # Formatting the opening and closing capacity.
    change = 0
    for i in range(1, len(opening_capacity)):
        if closing_capacity[i - 1] != opening_capacity[i]:
            opening_capacity[i] = closing_capacity[i - 1]
            closing_capacity[i] = opening_capacity[i]

    return raw_power, market_dispatch, opening_capacity, closing_capacity, best_revenue

def localMaximisation(dataframe, timePeriod):
    tmp_df = dataframe.copy()
    for period in timePeriod:
        start_t = period[0]
        end_t = period[1]

        data_interval = tmp_df.loc[(tmp_df['Time (UTC+10)'] >= start_t) & 
                                      (tmp_df['Time (UTC+10)'] <= end_t)]
        
        start_index = data_interval.index[0]

        price = GetSpotPrice(data_interval).tolist()
        min_price = (price.index(min(price)), min(price))
        max_price = (price.index(max(price)), max(price))
        
        raw_power, market_dispatch, opening_capacity, closing_capacity, _ = localOptimisation(data_interval)
        for i in range(len(raw_power)):
            tmp_df.loc[i + start_index, 'Raw Power (MW)'] = raw_power[i]
            tmp_df.loc[i + start_index, 'Market Dispatch (MWh)'] = market_dispatch[i]
            tmp_df.loc[i + start_index, 'Opening Capacity (MWh)'] = opening_capacity[i]
            tmp_df.loc[i + start_index, 'Closing Capacity (MWh)'] = closing_capacity[i]

    return tmp_df

def fullMaximisation(dataframe, period):
    tmp_df = dataframe.copy()
    raw_power = []
    market_dispatch = []
    opening_capacity = []
    closing_capacity = []

    start = 0
    end = period 
    
    # Given the data iterate over the given period.
    while end <= len(tmp_df):
        tmp_data = tmp_df.iloc[start:end, :]
        daily_rp, daily_md, daily_oc, daily_cc, _ = PeriodOptimisation(tmp_data) # Optimise the period given
        raw_power.extend(daily_rp)
        market_dispatch.extend(daily_md)
        opening_capacity.extend(daily_oc)
        closing_capacity.extend(daily_cc)

        start += period
        end += period

    tmp_df['Raw Power (MW)'] = pd.Series(raw_power)
    tmp_df['Market Dispatch (MWh)'] = pd.Series(market_dispatch)
    tmp_df['Opening Capacity (MWh)'] = pd.Series(opening_capacity)
    tmp_df['Closing Capacity (MWh)'] = pd.Series(closing_capacity)

    status = setStatus(tmp_df)
    tmp_df['Status'] = pd.Series(status)
    
    betweenPeriod = selectBetweenPeriod(tmp_df)
    newData = localMaximisation(tmp_df, betweenPeriod)
    
    return newData

def renameColumns(dataframe, region = 'VIC'):
    if region == 'VIC':
        price_name = 'Regions VIC Trading Price ($/MWh)'
    elif region == 'NSW':
        price_name = 'Regions NSW Trading Price ($/MWh)'
    elif region == 'SA':
        price_name = 'Regions SA Trading Price ($/MWh)'
    elif region == 'TAS':
        price_name = 'Regions TAS Trading Price ($/MWh)'
        
    dataframe.columns = ['Time (UTC+10)', price_name, 'Raw Power (MW)', 'Market Dispatch (MWh)', 'Opening Capacity (MWh)', 'Closing Capacity (MWh)']
    return dataframe

def computeRevenue(dataframe, region = "VIC"):
    region = 'VIC'
    if region == 'VIC':
        price_name = 'Regions VIC Trading Price ($/MWh)'
    elif region == 'NSW':
        price_name = 'Regions NSW Trading Price ($/MWh)'
    elif region == 'SA':
        price_name = 'Regions SA Trading Price ($/MWh)'
    elif region == 'TAS':
        price_name = 'Regions TAS Trading Price ($/MWh)'
        
    dataframe['mlf'] = 0
    dataframe.loc[dataframe['Market Dispatch (MWh)'] > 0, 'mlf'] = 0.991
    dataframe.loc[dataframe['Market Dispatch (MWh)'] < 0, 'mlf'] = 1/0.991
    dataframe['Revenue ($)'] = dataframe[price_name] * dataframe['mlf'] * dataframe['Market Dispatch (MWh)']
    return dataframe['Revenue ($)'].sum()

# Get dependency on the given dataframe.
# If there is capacity in the end of the day and the beginning of the day that counts as dependency
def getDependency(dataframe):
    dayBorder = []
    
    dependency = []
    for i in range(PERIOD, len(dataframe), PERIOD):
        dayBorder.append((i - 1, i))
        # If the closing capacity of last period of the day are not zero and the opening capacity the following day are 
        # not empty, then there is dependency between the days. Therefore, append.
        if dataframe.loc[i - 1, 'Closing Capacity (MWh)'] != 0 and dataframe.loc[i, 'Opening Capacity (MWh)'] != 0:
            dependency.append((i - 1, i))
    
    consecutiveDependency = []
    tmp = []
    for d in range(len(dependency) - 1):
        curr_d = dependency[d][0]
        next_d = dependency[d + 1][0]
        tmp.append(dependency[d])
        if (next_d - curr_d != PERIOD):
            consecutiveDependency.append(tmp)
            tmp = []
    
    if len(dependency) != 0 and dependency[-1][0] - consecutiveDependency[-1][0][0] == PERIOD:
        consecutiveDependency[-1].append(dependency[-1])
    else:
        consecutiveDependency.append([dependency[-1]])
        
    # Selects only the the beginning and last period of the consecutive dependencies.
    # For example, (monday, tuesday, wednesday, thursday, friday). So select (monday, friday) 
    for c in range(len(consecutiveDependency)):
        current = consecutiveDependency[c]
        if len(current) != 1:
            consecutiveDependency[c] = [current[0], current[-1]]
            
    return consecutiveDependency

# A function to create the time period given the index
def createPeriod(current, until):
    tmp = []
    for i in range(current, until, PERIOD):
        tmp.append((i, i + PERIOD - 1))
    return tmp

# Get the timeperiod of all the data including the dependencies.
def getTimePeriod(dataframe):
    dependency = getDependency(dataframe)
    tmp = []
    for depend in dependency:
        openPeriod = depend[0][0] - PERIOD + 1     # 48th period - 47 = 1st period
        closePeriod = depend[-1][-1] + PERIOD - 1  # 1st period + 47 = 48th period
        tmp.append((openPeriod, closePeriod))
    
    timeIndex = []
    timeIndex.extend(createPeriod(0, tmp[0][0]))
    
    for t in range(1, len(tmp)):
        timeIndex.append((tmp[t-1]))
        timeIndex.extend(createPeriod(tmp[t-1][1] + 1, tmp[t][0]))
    timeIndex.append(tmp[t])  
    
    last_index = timeIndex[-1][-1]
    len_df = len(dataframe)
    current = last_index
    
    while current != len_df - 1:
        timeIndex.append((current + 1, current + 1 + 47))
        current = timeIndex[-1][-1]

    timePeriod = []
    for index in timeIndex:
        timePeriod.append((dataframe.iloc[index[0], 0], dataframe.iloc[index[1], 0], index[1] - index[0] + 1))
  
    return timePeriod


def computeRawPower(dataframe):
    tmp_df = dataframe.copy()
    tmp_df['Efficiency'] = 0
    tmp_df.loc[tmp_df['Actual'] > 0, 'Efficiency'] = 2 / 0.9
    tmp_df.loc[tmp_df['Actual'] < 0, 'Efficiency'] = 2
    tmp_df['Raw Power'] = tmp_df['Efficiency'] * tmp_df['Actual']
    dataframe['Raw Power'] = tmp_df['Raw Power']
    return dataframe

def DependencyOptimisation(data, timePeriod, region = "VIC"):
    raw_power = []
    market_dispatch = []
    opening_capacity = []
    closing_capacity = []
    
    count = 0
    # Iterate over each timePeriod.
    for time in timePeriod:
        start_t = time[0]
        end_t = time[1]
        period_t = time[2]
        
        # Select the interval
        data_interval = data.loc[(data['Time (UTC+10)'] >= start_t) & \
                                 (data['Time (UTC+10)'] <= end_t)].copy()
        data_interval = data_interval.reset_index(drop = True)
        original_revenue = computeRevenue(data_interval) # Revenue of the previos model/given dataframe.
        
        daily_rev = 0
        # If selected interval period is greater than 48, then that means there is dependency.
        # Compute the revenue of independent model and dependent model, compare revenue with original revenue as well.
        if period_t > PERIOD:
            start = 0
            end = PERIOD 
            tmp_rp = []
            tmp_md = []
            tmp_oc = []
            tmp_cc = []
            while end <= len(data_interval):
                tmp_data = data_interval.iloc[start:end, :].copy()
                tmp_data = tmp_data.reset_index(drop = True)
                # Compute independent maximisation
                daily_data = fullMaximisation(tmp_data, period = len(tmp_data))
                daily_rev += computeRevenue(daily_data)
                tmp_rp.extend(daily_data['Raw Power (MW)'])
                tmp_md.extend(daily_data['Market Dispatch (MWh)'])
                tmp_oc.extend(daily_data['Opening Capacity (MWh)'])
                tmp_cc.extend(daily_data['Closing Capacity (MWh)'])
                
                start += PERIOD
                end += PERIOD
        # Compute dependent maximisation.
        dependent_data = fullMaximisation(data_interval, period = len(data_interval))
        revenue_2 = computeRevenue(dependent_data)
        # If dependent model revenue is higher or independent model revenue is higher than the original
        if (revenue_2 > original_revenue or daily_rev > original_revenue):
            # If there period is more than 48, compare the independent and dependent revenue
            # if independent is higher, settle with independent.
            # else settle with dependent.
            if daily_rev != 0 and daily_rev > revenue_2:
                raw_power.extend(tmp_rp)
                market_dispatch.extend(tmp_md)
                opening_capacity.extend(tmp_oc)
                closing_capacity.extend(tmp_cc)
            else:
                raw_power.extend(dependent_data['Raw Power (MW)'])
                market_dispatch.extend(dependent_data['Market Dispatch (MWh)'])
                opening_capacity.extend(dependent_data['Opening Capacity (MWh)'])
                closing_capacity.extend(dependent_data['Closing Capacity (MWh)'])
        else:
            # Original data
            raw_power.extend(data_interval['Raw Power (MW)'])
            market_dispatch.extend(data_interval['Market Dispatch (MWh)'])
            opening_capacity.extend(data_interval['Opening Capacity (MWh)'])
            closing_capacity.extend(data_interval['Closing Capacity (MWh)'])
            
        print(count) # To show progress of the algorithm
        count += 1
    
    data['Raw Power (MW)'] = pd.Series(raw_power)
    data['Market Dispatch (MWh)'] = pd.Series(market_dispatch)
    data['Opening Capacity (MWh)'] = pd.Series(opening_capacity)
    data['Closing Capacity (MWh)'] = pd.Series(closing_capacity)
    
    return data