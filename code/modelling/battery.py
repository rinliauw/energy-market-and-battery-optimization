import numpy as np

class Battery:
    """
    A Class to store battery functionaly such as revenue, charge and discharge
    periods, charge and discharge spot prices, charge and discharge market dispatch.

    ```NOTE``` : IF THERE IS ANY CHANGE IN CALCULATION, CHANGE CALCULATION IN THE FUNCTION
     MARKED BY (BATTERY CALCULATION).

    ---
    Consists of:
      - charge_period : period when it should charge.
      - discharge_period : period when it should discharge.
      - charge_price : spot price given charging period.
      - discharge_price : spot price given discharging period.
      - charge_market_dispatch : set amount of market dispatch given charging period.
      - discharge_market_dispatch : set amount of market dispatch given discharging period.

    ---
    Functions:
      - ComputeRevenue: To calculate revenue given discharge and charge period pairs.
      - Setting : Set all the battery functionalities. (BATTERY CALCULATION)
      - FirstOptimisation : Ensure that energy are not wasted or not used. (BATTERY CALCULATION)
      - SecondOptimisation : Ensure that to always charge the highest price less and discharge
                             highest price more. (BATTERY CALCULATION)
    ---
    Created and Developed by: Gilbert Putra
    """
    # BATTERY SPECIFICATIONS
    mlf = 0.991                 # Marginal Loss Factor
    battery_capacity = 580      # Battery Capacity
    battery_power = 300         # Battery Power
    charge_efficiency = 0.9     # Charge Efficiency
    discharge_efficiency = 0.9  # Discharge Efficiency
        
    def __init__(self, charge_period, charge_spot_price,
                 discharge_period, discharge_spot_price):
        """
        Initialise Battery Class.

        Parameters
        ----------

        charge_period : list (int)
            Possible ranked charging period. Ranked from lowest price to highest.
        charge_spot_price : list (int)
            Spot price of the possible charge period.
        discharge_period : list (int)
            Possible ranked discharge period. Ranked from highest price to lowest.
        discharge_spot_price : list (int)
            Spot price of the possible discharge period.
        """ 
        # Charge and Discharge Period
        self.charge_period = charge_period
        self.discharge_period = discharge_period
        
        # Spot Price during Charge and Discharge
        self.charge_price = charge_spot_price
        self.discharge_price = discharge_spot_price
    
    ########################################################################################    
    # CALCULATE REVENUE OF THE RESPECTIVE BATTERY.            
    def Revenue(self):
        if self.charge_market_dispatch == [] or self.discharge_market_dispatch == []:
            return 0
        
        # Spot Prices
        charge_sp = np.array(self.charge_price)[:, 1]
        discharge_sp = np.array(self.discharge_price)[:, 1]
        
        # Market Dispatches
        charge_md = np.array(self.charge_market_dispatch).T
        discharge_md = np.array(self.discharge_market_dispatch).T
        
        # Revenues
        charge_revenue = (charge_sp @ charge_md) * (1 / self.mlf)
        discharge_revenue = (discharge_sp @ discharge_md) * (self.mlf)
        
        return discharge_revenue + charge_revenue
    
    ########################################################################################
    # SET THE RAW POWER, DISPATCH, OPENING AND CLOSING CAPACITY OF THE BATTERY.
    def FirstSetting(self):
        
        OPENING = 0
        CLOSING = 1
        
        battery_power = self.battery_power
        battery_cap = self.battery_capacity
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        self.charge_raw_power = ['' for i in range(len_charge)]
        self.discharge_raw_power = ['' for i in range(len_discharge)]
        
        self.charge_market_dispatch = ['' for i in range(len_charge)]
        self.discharge_market_dispatch = ['' for i in range(len_discharge)]
        
        self.charge_capacity = [[0, 0] for i in range(len_charge)]
        self.discharge_capacity = [[0, 0] for i in range(len_discharge)]
        
        # CHARGE PERIOD --------------------------------------------------------------------
        for t in range(len_charge):
            # RAW_POWER[t] = -MIN(BATTERY_POWER, (BATTERY_CAPACITY - OPENING_CAPACITY[t]) / CHARGE_EFFICIENCY * 2)
            self.charge_raw_power[t] = -min(battery_power, 
                                           (battery_cap - self.charge_capacity[t][OPENING]) / 
                                            self.charge_efficiency * 2)

            # MARKET_DISPATCH[t] = RAW_POWER / 2
            self.charge_market_dispatch[t] = self.charge_raw_power[t] / 2

            # CLOSING_CAPACITY[t] = MAX(0, MIN(OPENING_CAPACITY[t] - 
            #                        MARKET_DISPATCH[t] * CHARGE_EFFICIENCY, BATTERY_CAPACITY))
            self.charge_capacity[t][CLOSING] = max(0, min(self.charge_capacity[t][OPENING] - 
                                                        self.charge_market_dispatch[t] * self.charge_efficiency, 
                                                        battery_cap))

            # Ensuring that it doesn't exceeds array len limit
            if t + 1 < len_charge:
                self.charge_capacity[t + 1][OPENING] = self.charge_capacity[t][CLOSING]

        # DISCHARGE PERIOD -----------------------------------------------------------------

        # Set DISCHARGE CAPACITY AT t = 0 to be the the last t of CHARGING CAPACITY
        self.discharge_capacity[0][OPENING] = self.charge_capacity[-1][CLOSING]

        for t in range(len_discharge):
            # RAW_POWER[t] = MIN(BATTERY_POWER, OPENING_CAPACITY[t] * 2)
            self.discharge_raw_power[t] = min(battery_power, self.discharge_capacity[t][OPENING] * 2)

            # MARKET_DISPATCH[t] = RAW_POWER[t] / 2 * DISCHARGE EFFICIENCY
            self.discharge_market_dispatch[t] = self.discharge_raw_power[t] / 2 * self.discharge_efficiency

            # CLOSING CAPACITY[t] = MAX(0, MIN(OPENING_CAPACITY[t] -
            #                        MARKET_DISPATCH[t] * (1/DISCHARGE_EFFICIENCY), BATTERY_CAPACITY))
            self.discharge_capacity[t][CLOSING] = max(0, min(self.discharge_capacity[t][OPENING] -
                                                            self.discharge_market_dispatch[t] * (1 / self.discharge_efficiency),
                                                            battery_cap))
            # Ensuring that it doesn't exceeds array len limit
            if t + 1 < len_discharge:
                self.discharge_capacity[t + 1][OPENING] = self.discharge_capacity[t][CLOSING]

    def SecondSetting(self, opening_capacity, closing_capacity):
        
        return

    
    ########################################################################################
    # OPTIMISING BY CHOOSING THE CHARGE AND DISCHARGE AMOUNT REQUIRED    
    def FirstOptimisation(self):
        OPENING = 0
        CLOSING = 1
        
        battery_power = self.battery_power
        battery_cap = self.battery_capacity
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        MAX_CHARGE_PERIOD = 5
        MAX_DISCHARGE_PERIOD = 4
        
        if (len_charge >= MAX_CHARGE_PERIOD and len_discharge >= MAX_DISCHARGE_PERIOD) or (len_charge == len_discharge):
            #md = 117
            #self.discharge_raw_power[-1] = min(md * 2 / self.discharge_efficiency, self.discharge_raw_power[-1])
            #self.discharge_market_dispatch[-1] = self.discharge_raw_power[-1] / 2 * self.discharge_efficiency
            return
        elif (len_charge - len_discharge == 1):
            self.discharge_capacity[-1][CLOSING] = 0
            # OPENING_CAPACITY[-1] = MARKET_DISPATCH / DISCHARGE_EFFICIENCY
            self.discharge_capacity[-1][OPENING] = self.discharge_market_dispatch[-1] / self.discharge_efficiency
            for t in range(1, len_discharge):
                # CLOSING_CAPACITY[-t - 1] = OPENING_CAPACITY[-t]
                self.discharge_capacity[-t - 1][CLOSING] = self.discharge_capacity[-t][OPENING]
                # OPENING_CAPACITY[-t - 1] = CLOSING_CAPACITY[-t - 1] + MARKET_DISPATCH[-t - 1] / DISCHARGE_EFFICIENCY
                self.discharge_capacity[-t - 1][OPENING] = self.discharge_capacity[-t - 1][CLOSING] + self.discharge_market_dispatch[-t - 1] / self.discharge_efficiency
            
            # CLOSING_CAPACITY[-1] = OPENING_CAPACITY[0]
            self.charge_capacity[-1][CLOSING] = self.discharge_capacity[0][OPENING]
            # MARKET_DISPATCH[-1] = -(CLOSING_CAPACITY[-1] - OPENING_CAPACITY[-1]) / CHARGE_EFFICIENCY
            self.charge_market_dispatch[-1] = -(self.charge_capacity[-1][CLOSING] - self.charge_capacity[-1][OPENING]) / self.charge_efficiency
            # RAW_POWER[-1] = MARKET_DISPATCH[-1] * 2
            self.charge_raw_power[-1] = self.charge_market_dispatch[-1] * 2
            
    ########################################################################################
    # LOCALISED OPTIMISATION TO SET THE HIGHEST PRICE OF CHARGING TO HAVE THE LOWEST DISPATCH
    # AND TO SET THE LOWEST PRICE OF DISCHARGING TO HAVE THE LOWEST DISPATCH.
    def SecondOptimisation(self):
        OPENING = 0
        CLOSING = 1
        
        battery_power = self.battery_power
        battery_cap = self.battery_capacity
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        MAX_CHARGE_PERIOD = 5
        MAX_DISCHARGE_PERIOD = 4
        
        # CHARGE PERIOD --------------------------------------------------------------------
        highest_price_index = np.array(self.charge_period).argmax(axis=0)[0]
        lowest_charge_index = self.charge_market_dispatch.index(max(self.charge_market_dispatch))    
        
        # IF THE HIGHEST CHARGING PRICE DOESN'T HAVE THE LOWEST CHARGING RATE, SWAP!
        if (highest_price_index != lowest_charge_index):
            tmp = self.charge_market_dispatch[highest_price_index]
            self.charge_market_dispatch[highest_price_index] = self.charge_market_dispatch[lowest_charge_index]
            self.charge_market_dispatch[lowest_charge_index] = tmp
            # SET THE UPDATED BATTERY SETTINGS.
            for t in range(len_charge):
                # RAW_POWER[t] = MARKET_DISPATCH[t] * 2
                self.charge_raw_power[t] = self.charge_market_dispatch[t] * 2

                # CLOSING_CAPACITY[t] = MAX(0, MIN(OPENING_CAPACITY[t] - 
                #                        MARKET_DISPATCH[t] * CHARGE_EFFICIENCY, BATTERY_CAPACITY))
                self.charge_capacity[t][CLOSING] = max(0, min(self.charge_capacity[t][OPENING] - 
                                                            self.charge_market_dispatch[t] * self.charge_efficiency, 
                                                            battery_cap))
                # Ensuring that it doesn't exceeds array len limit
                if t + 1 < len_charge:
                    self.charge_capacity[t + 1][OPENING] = self.charge_capacity[t][CLOSING]

        # DISCHARGE PERIOD -----------------------------------------------------------------
        lowest_price_index = np.array(self.discharge_period).argmax(axis=0)[0] 
        min_discharge_index = self.discharge_market_dispatch.index(min(self.discharge_market_dispatch))   
        
        # IF THE LOWEST DISCHARGING PRICE DOESN'T HAVE THE LOWEST DISCHARGING RATE, SWAP!
        if (lowest_price_index != min_discharge_index):
            tmp = self.discharge_market_dispatch[lowest_price_index]
            self.discharge_market_dispatch[lowest_price_index] = self.discharge_market_dispatch[min_discharge_index]
            self.discharge_market_dispatch[min_discharge_index] = tmp
            # SET THE UPDATED BATTERY SETTINGS.
            for t in range(len_discharge):
                # RAW_POWER[t] = MARKET_DISPATCH[t] * 2 / DISCHARGE_EFFICIENCY
                self.discharge_raw_power[t] = self.discharge_market_dispatch[t] * 2 / self.discharge_efficiency

                # CLOSING_CAPACITY[t] = MAX(0, MIN(OPENING_CAPACITY[t] - 
                #                        MARKET_DISPATCH[t] * CHARGE_EFFICIENCY, BATTERY_CAPACITY))
                self.discharge_capacity[t][CLOSING] = max(0, min(self.discharge_capacity[t][OPENING] - 
                                                            self.discharge_market_dispatch[t] / self.discharge_efficiency, 
                                                            battery_cap))
                # Ensuring that it doesn't exceeds array len limit
                if t + 1 < len_discharge:
                    self.discharge_capacity[t + 1][OPENING] = self.discharge_capacity[t][CLOSING]