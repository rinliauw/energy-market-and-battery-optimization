import numpy as np

OPENING = 0
CLOSING = 1

class Battery:
    """
    A Class to store battery functionaly such as revenue, charge and discharge
    periods, charge and discharge spot prices, charge and discharge market dispatch.

    ```NOTE``` : IF THERE IS ANY CHANGE IN CALCULATION, CHANGE CALCULATION IN THE FUNCTION
     MARKED BY (BATTERY CALCULATION).

    Parameters
    ----------
        - charge_period : list
            List of period when it should charge.
        - discharge_period : list
            List of period when it should discharge.
        - charge_price : list
            List of spot price given charging period.
        - discharge_price : list
            List ofspot price given discharging period.
        - mlf : double
            Marginal Loss Factor (default = 0.991)
        - battery_capacity : double
            Maximum Battery Capacity (default = 580)
        - battery_power : double
            Maximum Battery Power (default = 300)
        - charge_efficiency : double
            Charging efficiency (default = 0.9)
        - discharge_efficiency : double
            Discharging efficiency (default = 0.9)
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

        - Setting : Set all the battery functionalities. `(BATTERY CALCULATION)`

        - FirstOptimisation : Ensure that energy are not wasted or not used. `(BATTERY CALCULATION)`
        - SecondOptimisation : Ensure that to always charge the highest price less and discharge
                                highest price more. `(BATTERY CALCULATION)`

        - ReverseSetting : A wrapper function for EqualSetting and NonEqualSetting. Use this function 
                            if battery capacity is not empty at the beginning.

        - EqualSetting : A function to set the battery functionalities if the len of charge and discharge
                        are `Equal`.
        - NonEqualSetting : A function to set the battery functionalities if the len of charge and discharge
                            are `Not Equal`.
    ---
    Created and Developed by: Gilbert Putra
    """
    # Battery Specifications
    # NOTE: IF THERE IS ANY CHANGE HERE IN THE SPECIFICATIONS CHANGE THE VALUE BELOW.
    # TODO: MAKE THIS SPECIFICATIONS AS PARAMETERS OF THIS BATTERY CLASS. (COMPLETED)
    mlf = 0.991                 # Marginal Loss Factor
    battery_capacity = 580      # Max Battery Capacity
    battery_power = 300         # Max Battery Power
    charge_efficiency = 0.9     # Charge Efficiency
    discharge_efficiency = 0.9  # Discharge Efficiency
        
    def __init__(self, charge_period, charge_spot_price,
                 discharge_period, discharge_spot_price):
        '''
                   
        Initialise the battery class.

        Parameters
        ----------
        charge_period : list
            Period when to charge
        charge_spot_price : list
            Spot Price at given charge period
        discharge_period : list
            Period when to discharge
        discharge_spot_price : list
            Spot Price at a given discharge period
        '''
        # Charge and Discharge Period
        self.charge_period = charge_period
        self.discharge_period = discharge_period
        
        # Spot Price during Charge and Discharge
        self.charge_price = charge_spot_price
        self.discharge_price = discharge_spot_price
    
    ########################################################################################    
                
    def Revenue(self):
        '''
        A function to compute the revenue of the battery pairs.
        '''
        if self.charge_market_dispatch == [] or self.discharge_market_dispatch == []:
            return 0
        
        # Spot Prices
        charge_sp = np.array(self.charge_price)[:, 1]
        discharge_sp = np.array(self.discharge_price)[:, 1]
        
        # Market Dispatches
        charge_md = np.array(self.charge_market_dispatch).T
        discharge_md = np.array(self.discharge_market_dispatch).T
        
        # Revenues
        # SPOT PRICE * MARKET DISPATCH * 1/MARGINAL LOSS FACTOR
        charge_revenue = (charge_sp @ charge_md) * (1 / self.mlf)
        # SPOT PRICE * MARKET DISPATCH * MARGINAL LOSS FACTOR
        discharge_revenue = (discharge_sp @ discharge_md) * (self.mlf)
        
        return discharge_revenue + charge_revenue
    
    ########################################################################################
    
    def Setting(self):
        '''
        A function to set the battery parameters. For example, set raw power, market dispatch,
        opening and closing capacity of the battery pairs.
        '''
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
        
        ################################## CHARGE PERIOD ##################################
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

        ################################## DISCHARGE PERIOD ##################################

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
    
    ########################################################################################
        
    def FirstOptimisation(self):
        '''
        A function to optimise the battery by charging the amount that is needed to fully 
        discharge.
        '''
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        MAX_CHARGE_PERIOD = 5
        MAX_DISCHARGE_PERIOD = 4
        
        if (len_charge >= MAX_CHARGE_PERIOD and len_discharge >= MAX_DISCHARGE_PERIOD) or (len_charge == len_discharge):
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

    def SecondOptimisation(self):
        '''
        A function to optimise battery settings to set the lowest charging price to have
        the highest charging rate and to set the highest discharging price to have the 
        highest price.
        '''
        battery_cap = self.battery_capacity
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        ################################## CHARGE PERIOD ##################################
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

        ################################## DISCHARGE PERIOD ##################################
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
    
    ########################################################################################                
    #
    # REVERSE BATTERY
    #
    ########################################################################################
    def ReverseSetting(self, discharge_capacity, charge_capacity):
        '''
        A function to set the battery while it has some capacities in 
        the beginning.

        Parameters
        ----------
        discharge_capacity : list
            The discharge capacity of the given period
        charge_capacity : list
            The charge capacity of the given period
        '''
        if len(self.charge_period) == len(self.discharge_period):
            self.EqualSetting(discharge_capacity, charge_capacity)
        else:
            self.NonEqualSetting(discharge_capacity, charge_capacity)

    ########################################################################################
            
    def EqualSetting(self, discharge_capacity, charge_capacity):
        '''
        A function to set the battery setting while starting with non empty
        capacity if it is equal in length of discharge and charge.
        
        Parameters
        ----------
        discharge_capacity : list
            The discharge capacity of the given period
        charge_capacity : list
            The charge capacity of the given period
        '''
        battery_cap = np.amax(charge_capacity)
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        MAX_RAW_POWER = self.battery_power
        MAX_CHARGE_MD = MAX_RAW_POWER / 2
        MAX_DISCHARGE_MD = MAX_RAW_POWER / 2 * self.discharge_efficiency
        
        self.charge_capacity = charge_capacity
        self.discharge_capacity = discharge_capacity
        
        self.charge_raw_power = ['' for i in range(len_charge)]
        self.discharge_raw_power = ['' for i in range(len_discharge)]
        
        self.charge_market_dispatch = ['' for i in range(len_charge)]
        self.discharge_market_dispatch = ['' for i in range(len_discharge)]
        
        ################################## CHARGE PERIOD ##################################
        # BACKTRACKING FROM THE CHARGING PERIOD TO ENSURE THERE IS NO LOSSES IN REVENUE DURING CHARGING
        # OPENING CAPACITY[-1] = MAX(0, MIN(CLOSING CAPACITY[-1] - 
        #                                   MAXIMUM CHARGING POSSIBLE * CHARGING EFFICIENCY, MAXIMUM BATTERY CAPACITY))
        self.charge_capacity[-1][OPENING] = max(0, min(self.charge_capacity[-1][CLOSING] - 
                                                        MAX_CHARGE_MD * self.charge_efficiency, battery_cap))
        # Traversing backwards                                        
        for t in range(1, len_charge):
            # CLOSING CAPACITY[-t -1] = OPENING CAPACITY[-t]
            self.charge_capacity[-t -1][CLOSING] = self.charge_capacity[-t][OPENING]
            # OPENING CAPACITY[-1] = MAX(0, MIN(CLOSING CAPACITY[-1] - 
        #                                   MAXIMUM CHARGING POSSIBLE * CHARGING EFFICIENCY, MAXIMUM BATTERY CAPACITY))
            self.charge_capacity[-t -1][OPENING] = max(0, min(self.charge_capacity[-t -1][CLOSING] - 
                                                                MAX_CHARGE_MD * self.charge_efficiency, battery_cap))
            # MARKET DISPATCH[-t] = -MIN(MAXIMUM CHARGING DISPATCH, (CLOSING CAPACITY[-t] - 
            #                                                        OPENING CAPACITY[-t]) / CHARGING EFFICIENCY)
            self.charge_market_dispatch[-t] = -min(MAX_CHARGE_MD, (self.charge_capacity[-t][CLOSING] - 
                                                                    self.charge_capacity[-t][OPENING]) / self.charge_efficiency)
            # RAW POWER[-t] = MAX(-MAXIMUM RAW POWER, MARKET DISPATCH[-t] * 2)
            self.charge_raw_power[-t] = max(-MAX_RAW_POWER, self.charge_market_dispatch[-t] * 2)
        # MARKET DISPATCH[0] = -MIN(MAXIMUM CHARGING DISPATCH, (CLOSING CAPACITY[0] - 
        #                                                        OPENING CAPACITY[0]) / CHARGING EFFICIENCY)
        self.charge_market_dispatch[0] = -min((self.charge_capacity[0][CLOSING] - 
                                                self.charge_capacity[0][OPENING]) / self.charge_efficiency, MAX_CHARGE_MD)
        # RAW POWER[0] = MAX(-MAXIMUM RAW POWER, MARKET DISPATCH[0] * 2)
        self.charge_raw_power[0] = max(-MAX_RAW_POWER, self.charge_market_dispatch[0] * 2)  
        
        ################################## DISCHARGE PERIOD ##################################
        # BACKTRACKING FROM THE MAXIMUM CHARGING POSSIBLE IN THIS BATTERY PAIRS TO ENSURE IT FOLLOWS THE
        # BATTERY SPECIFICATIONS.
        # DISCHARGING CLOSING CAPACITY[-1] = CHARING OPENING CAPACITY[0]
        self.discharge_capacity[-1][CLOSING] = self.charge_capacity[0][OPENING]
        # OPENING CAPACITY[-1] = MIN(MAXIMUM BATTERY CAPACITY, CLOSING CAPACITY[-1]) +
        #                                                       MAXIMUM DISCHARGING DISPATCH / DISCHARGE EFFICIENCY
        self.discharge_capacity[-1][OPENING] = min(battery_cap, self.discharge_capacity[-1][CLOSING] + 
                                                                    MAX_DISCHARGE_MD / self.discharge_efficiency)
        # MARKET DISPATCH[-1] = MIN(MAXIMUM DISCHARGING DISPATCH, DISCHARGING OPENING CAPACITY[-1] - 
        #                                                           DISCHARGING CLOSING CAPACITY[-1] * DISCHARGE EFFICIENCY)
        self.discharge_market_dispatch[-1] = min(MAX_DISCHARGE_MD, (self.discharge_capacity[-1][OPENING] - 
                                                                        self.discharge_capacity[-1][CLOSING]) * self.discharge_efficiency) 
        # DISCHARGE RAW POWER[-1] = MIN(MAXIMUM RAW POWER, DISCHARGE MARKET DISPATCH[-1] * 2 / DISCHARGE EFFICIENCY)
        self.discharge_raw_power[-1] = min(MAX_RAW_POWER, self.discharge_market_dispatch[-1] * 2 / self.discharge_efficiency)

        # Traversing backwards
        for t in range(1, len_discharge):
            # DISCHARGING CLOSING CAPACITY[-t -1] = CHARING OPENING CAPACITY[-t]
            self.discharge_capacity[-t -1][CLOSING] = self.discharge_capacity[-t][OPENING]
            # OPENING CAPACITY[-t -1] = MIN(MAXIMUM BATTERY CAPACITY, CLOSING CAPACITY[-t -1]) +
            #                                                       MAXIMUM DISCHARGING DISPATCH / DISCHARGE EFFICIENCY
            self.discharge_capacity[-t -1][OPENING] = min(battery_cap, self.discharge_capacity[-t -1][CLOSING] + 
                                                                        MAX_DISCHARGE_MD / self.discharge_efficiency)
            # MARKET DISPATCH[-t -1] = MIN(MAXIMUM DISCHARGING DISPATCH, DISCHARGING OPENING CAPACITY[-t -1] - 
            #                                                           DISCHARGING CLOSING CAPACITY[-t -1] * DISCHARGE EFFICIENCY)
            self.discharge_market_dispatch[-t-1] = min(MAX_DISCHARGE_MD, (self.discharge_capacity[-t -1][OPENING] - 
                                                                            self.discharge_capacity[-t -1][CLOSING]) * self.discharge_efficiency) 
            # DISCHARGE RAW POWER[-t -1] = MIN(MAXIMUM RAW POWER, DISCHARGE MARKET DISPATCH[-t -1] * 2 / DISCHARGE EFFICIENCY)
            self.discharge_raw_power[-t -1] = min(MAX_RAW_POWER, self.discharge_market_dispatch[-t -1] * 2 / self.discharge_efficiency)    
        
    ########################################################################################

    def NonEqualSetting(self, discharge_capacity, charge_capacity):
        '''
        A function to set the battery setting while starting with non empty
        capacity if it is ``not`` equal in length of discharge and charge.

        Parameters
        ----------
        discharge_capacity : list
            The discharge capacity of the given period
        charge_capacity : list
            The charge capacity of the given period

        '''
        battery_cap = np.amax(charge_capacity)
        
        len_charge = len(self.charge_period)
        len_discharge = len(self.discharge_period)
        
        
        MAX_RAW_POWER = self.battery_power
        MAX_CHARGE_MD = MAX_RAW_POWER / 2
        MAX_DISCHARGE_MD = MAX_RAW_POWER / 2 * self.discharge_efficiency
        
        self.charge_capacity = charge_capacity
        self.discharge_capacity = discharge_capacity
        
        self.charge_raw_power = ['' for i in range(len_charge)]
        self.discharge_raw_power = ['' for i in range(len_discharge)]
        
        self.charge_market_dispatch = ['' for i in range(len_charge)]
        self.discharge_market_dispatch = ['' for i in range(len_discharge)]
        
        ################################## DISCHARGE PERIOD ##################################
        # FORWARDING FROM THE MAXIMUM DISCHARGING POSSIBLE IN THIS BATTERY PAIRS TO ENSURE IT FOLLOWS THE
        # BATTERY SPECIFICATIONS.
        # DISCHARGE CLOSING CAPACITY[0] = MAX(0, MIN(DISCHARGE OPENING CAPACITY[0]) -
        #                                               MAXIMUM DISCHARGE MARKET DISPATCH / DISCHARGE EFFICIENCY, MAXIMUM BATTERY CAPACITY)
        self.discharge_capacity[0][CLOSING] = max(0, min(self.discharge_capacity[0][OPENING] - 
                                                         MAX_DISCHARGE_MD / self.discharge_efficiency, battery_cap))
        # Traversing forward
        for t in range(1, len_discharge):
            # DISCHARGE OPENING CAPACITY[t] = DISCHARGE CLOSING CAPACITY[t-1]
            self.discharge_capacity[t][OPENING] = self.discharge_capacity[t - 1][CLOSING]
            # DISCHARGE CLOSING CAPACITY[t] = MAX(0, MIN(DISCHARGING CAPACITY[t][OPENING]) - 
            #                                               MAX DISCHARGE DISPATCH / DISCHARGING EFFICIENCY, MAXIMUM BATTERY CAPACITY)
            self.discharge_capacity[t][CLOSING] = max(0, min(self.discharge_capacity[t][OPENING] - 
                                                                MAX_DISCHARGE_MD / self.discharge_efficiency, battery_cap))
            # DISCHARGE MARKET DISPATCH[t-1] = MIN((DISCHARGE OPENING CAPACITY[t-1] - 
            #                                       DISCHARGE CLOSING CAPACITY[t-1]) * DISCHARGE EFFICIENCY, MAXIMUM DISCHARGE DISPATCH)
            self.discharge_market_dispatch[t-1] = min((self.discharge_capacity[t-1][OPENING] - 
                                                        self.discharge_capacity[t-1][CLOSING]) * self.discharge_efficiency, MAX_DISCHARGE_MD)
            # DISCHARGE RAW POWER[t-1] = MIN(MAXIMUM RAW POWER, DISCHARGE MARKET DISPATCH[t-1] * 2 / DISCHARGE EFFICIENCY)
            self.discharge_raw_power[t-1] = min(MAX_RAW_POWER, self.discharge_market_dispatch[t-1] * 2 / self.discharge_efficiency)
        
        # DISCHARGE MARKET DISPATCH[-1] = MIN((DISCHARGE OPENING CAPACITY[-1] - 
        #                                       DISCHARGE CLOSING CAPACITY[-1]) * DISCHARGE EFFICIENCY, MAXIMUM DISCHARGE DISPATCH )
        self.discharge_market_dispatch[-1] = min((self.discharge_capacity[-1][OPENING] - 
                                                    self.discharge_capacity[-1][CLOSING]) * self.discharge_efficiency, MAX_DISCHARGE_MD)
        # DISCHARGE RAW POWER[-1] = MIN(MAXIMUM RAW POWER, DISCHARGE MARKET DISPATCH[-1] * 2 / DISCHARGE EFFICIENCY)
        self.discharge_raw_power[-1] = min(MAX_RAW_POWER, self.discharge_market_dispatch[-1] * 2 / self.discharge_efficiency)
        
        ################################## CHARGE PERIOD ##################################
        # FORWARDING FROM THE CHARGING PERIOD TO ENSURE THERE IS NO LOSSES IN REVENUE DURING CHARGING
        # BY CHARGING WHAT WE NEEDED FOR THE DISCHARGE.
        # CHARGE OPENING CAPACITY[0] = DISCHARGE CLOSING CAPACITY[-1]
        self.charge_capacity[0][OPENING] = self.discharge_capacity[-1][CLOSING]
        # CHARGE MARKET DISPATCH[0] = MIN(CHARGE CLOSING CAPACITY[0] -
        #                                  CHARGE OPENING CAPACITY[0] * CHARGING EFFICIENCY, MAXIMUM CHARGING DISPATCH)
        self.charge_market_dispatch[0] = min(self.charge_capacity[0][CLOSING] - 
                                                self.charge_capacity[0][OPENING] * self.charge_efficiency, MAX_CHARGE_MD)
        # CHARGE RAW POWER[0] = MIN(MAXIMUM RAW POWER, CHARGE MARKET DISPATCH[0] * 2)
        self.charge_raw_power[0] = min(MAX_RAW_POWER, self.charge_market_dispatch[0] * 2)
        # Traversing forward
        for t in range(1, len_charge):
            # CHARGE CLOSING CAPACITY[t-1] = MIN(MAXIMUM BATTERY CAPACITY, CHARGE OPENING CAPACITY[t-1] + MAXIMUM CHARGING DISPATCH * CHARGING EFFICIENCY)
            self.charge_capacity[t - 1][CLOSING] = min(battery_cap, self.charge_capacity[t-1][OPENING] + MAX_CHARGE_MD * self.charge_efficiency)
            # CHARGE OPENING CAPACITY[t] = CHARGE CLOSING CAPACITY[t-1]
            self.charge_capacity[t][OPENING] = self.charge_capacity[t - 1][CLOSING]
            # CHARGE MARKET DISPATCH[t-1] = -MIN((CHARGE CLOSING CAPACITY[t-1]) -
            #                                       CHARGE OPENING CAPACITY[t-1]) / CHARGING EFFICIENCY, MAXIMUM CHARGING DISPATCH
            self.charge_market_dispatch[t-1] = -min((self.charge_capacity[t-1][CLOSING] - 
                                                        self.charge_capacity[t-1][OPENING]) / self.charge_efficiency, MAX_CHARGE_MD)
            # CHARGE RAW POWER[t-1] = MAX(-MAXIMUM RAW POWER, CHARGE MARKET DISPATCH[t-1] * 2)
            self.charge_raw_power[t-1] = max(-MAX_RAW_POWER, self.charge_market_dispatch[t-1] * 2)
            
        # CHARGE MARKET DISPATCH[-1] = -MIN((CHARGE CLOSING CAPACITY[-1] -
        #                                       CHARGE OPENING CAPACITY[-1]) / CHARGE EFFICIENCY, MAXIMUM CHARGING DISPATCH)
        self.charge_market_dispatch[-1] = -min((self.charge_capacity[-1][CLOSING] - 
                                                    self.charge_capacity[-1][OPENING]) / self.charge_efficiency, MAX_CHARGE_MD)
        # CHARGE RAW POWER[-1] = MAX(-MAXIMUM RAW POWER, CHARGE MARKET DISPATCH[-1] * 2)
        self.charge_raw_power[-1] = max(-MAX_RAW_POWER, self.charge_market_dispatch[-1] * 2)