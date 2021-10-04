# PYTHON LIBRARIES ##################
import pandas as pd
import numpy as np
import math

from collections import OrderedDict

# CUSTOM CLASSES ####################
import Battery

# GLOBAL VARIABLE ###################

# CHANGE THIS TO THE INTERVAL DESIRED. AS THE DATA WE HAVE NOW IS EVERY 30 MINUTES,
# THEREFORE IN 24 HOUR THERE WILL BE 48 PERIODS
PERIOD = 48 

###################################
# A function to get spot prices based on selected regions.
#
# Parameters:
#      - data : the targeted dataset, minimum dataset length of 48.
#      - selected_periods : selected period for charging or discharging.
#      - region : the targeted region, default has been set to 'VIC' for mandatory task.
#
# Return:
#      - List of spot prices given period
#
# Created by: Gilbert
###################################
def GetSpotPrice(data, selected_periods = False, region = 'VIC'):
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

###################################
# A function to find minimum and maximum point rank given threshold.
#
# Parameters:
#      - data : the targeted dataset, minimum dataset length of 48.
#      - region : the targeted region, default has been set to 'VIC' for mandatory task.
#      - buy_threshold : maximum number of buying point, default has been set to optimise Checkpoint 3.
#      - sell_threshold : maximum number of selling point, default has been set to optimise Checkpoint 3.
#
# Return:
#      - List of selected minimum point, list of selected maximum point
#
# Efficiency: O(3N + NLogN) = O(NLogN)
#
# Created by: Gilbert
###################################
def GetMinMax(data, region = 'VIC', buy_threshold = 5, sell_threshold = 4):
    EMPTY = ' '
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

###################################
# A function to Find Battery Charge and Discharge pairs in backward order.
# Backward order from 48th period to the 1st.
#
# Parameters:
#      - buy_period : Selected minimum price point as it will be where we buy energy for charging.
#      - sell_period : Selected maximum price point as it will be where we sell energy for discharging.
#
# Return:
#      - List of battery class pairs
#
# Efficiency: O(N)
#
# Created by: Gilbert
###################################
def FindBatteryPairs(buy_period, sell_period):
    MAX_SELL_PERIOD = 4 # MAXIMUM SELLING PERIOD PER PAIR
    MAX_BUY_PERIOD = 5 # MAXIMUM BUYING PERIOD PER PAIR
    EMPTY = ' '
    
    period = len(buy_period)
    
    battery = []
    sell = OrderedDict() # Initialise battery selling point. (Ordered Dictionary)
    buy = OrderedDict() # Initialise battery buying point. (Orderered Dictionary)
    
    # Iterate over the whole period backwards
    for p in range(period - 1, -1, -1):
        # If maximum selling point is not empty, add (order, period)
        # as key-value pair into the OrderedDict.
        if sell_period[p] != EMPTY:
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
        if len(sell) != 0 and buy_period[p] != EMPTY:
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

###################################
# A function to set optimal charge and discharge amount of battery pairs.
#
# Parameters:
#      - data : the targeted dataset, minimum dataset length of 48.
#      - battery_pairs : list of all battery class pairs.
#
# Return:
#      - List of all battery class pairs
#
# Efficiency: O(N)
#
# Created by: Gilbert
###################################
def SetChargeDischarge(data, battery_pairs, region = "VIC"):    
    all_batteries = []
    
    battery_pairs = battery_pairs[::-1]
    for b in battery_pairs:
        sell_period = b[0][::-1] # Reverse the order
        buy_period = b[1][::-1] # Reverse the order
        if len(sell_period)!= 0 or len(buy_period) != 0:
            sell_price = GetSpotPrice(data, sell_period, region = region)
            buy_price = GetSpotPrice(data, buy_period, region = region)

            battery = Battery(buy_period, buy_price, sell_period, sell_price)
            battery.Setting()
            battery.FirstOptimisation()
            battery.SecondOptimisation()

            all_batteries.append(battery)

    return all_batteries

########################################################################################

###################################
# A function to calculate daily revenue.
#
# Parameters:
#      - all_batteries : List of battery class pairs.
#
# Return:
#      - Daily revenues
#
# Created by: Gilbert
###################################
def ComputeDailyRevenue(all_batteries):
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
    
    # TODO: OPTIMISE EFFICIENCY HERE! Make this at least < N^2
    # Iterate over all possible combinations of battery pairs. (O(N^2))
    period = len(given_data)
    count = 0
    for s in range(1, period + 1):
        for b in range(1 , period + 1): # change this to range(1, period - s + 1) to reduce by half
            # Get the minimum and maximum price based on the given threshold
            min_price, max_price = GetMinMax(given_data, buy_threshold = b, sell_threshold = s, region = region)
            # Get the battery pairs based on minimum and maximum price
            battery_pairs = FindBatteryPairs(min_price, max_price)
            # Get battery optimisation for the selected threshold
            all_batteries = SetChargeDischarge(given_data, battery_pairs, region = region)
            # Compute daily revenues of selected battery combinations
            dailyrev = ComputeDailyRevenue(all_batteries)
            #print((b, s), dailyrev)
            if dailyrev < 0:
                break
            count += 1
            # Insert revenue as key, batteries combination and threshold as value
            if dailyrev not in best_batteries:
                best_batteries[dailyrev] = (all_batteries, (b, s))
            #print(count)
    #print(count)           
    # Find the highest revenue amongst possible combinations in that day
    best_revenue = max(best_batteries)
    best_threshold = best_batteries[best_revenue][1]
    #print(best_threshold)
    battery = best_batteries[best_revenue][0] # The Best battery combinations
    
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

def FirstOptimisation(data, period = 48, region = 'VIC'):
    raw_power = []
    market_dispatch = []
    opening_capacity = []
    closing_capacity = []

    start = 0
    end = period 

    while end <= len(data):
        tmp_data = data.iloc[start:end, :]
        daily_rp, daily_md, daily_oc, daily_cc, _ = PeriodOptimisation(tmp_data)
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



def getDependency(dataframe):
    dayBorder = []
    
    dependency = []
    for i in range(PERIOD, len(dataframe), PERIOD):
        dayBorder.append((i - 1, i))
        if dataframe.loc[i - 1, 'Closing Capacity'] != 0 and dataframe.loc[i, 'Closing Capacity'] != 0:
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
       
    if dependency[-1][0] - consecutiveDependency[-1][0][0] == PERIOD:
        consecutiveDependency[-1].append(dependency[-1])
    else:
        consecutiveDependency.append([dependency[-1]])

    for c in range(len(consecutiveDependency)):
        current = consecutiveDependency[c]
        if len(current) != 1:
            consecutiveDependency[c] = [current[0], current[-1]]
            
    return consecutiveDependency

def createPeriod(current, until):
    tmp = []
    for i in range(current, until, PERIOD):
        tmp.append((i, i + PERIOD - 1))
    return tmp

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
    
    timePeriod = []
    for index in timeIndex:
        timePeriod.append((dataframe.iloc[index[0], 0], dataframe.iloc[index[1], 0], index[1] - index[0] + 1))
   
    return timePeriod