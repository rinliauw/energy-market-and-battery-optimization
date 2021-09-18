class Battery:
    def __init__(self, price, period, status = 'Charge'):
        self.status = status
        self.price = price
        self.period = period

    def windowCombinations(self):
        period = self.period
        if self.status == 'Charge':
            combi_1 = (period, period + 1, period + 2, period + 3, period + 4)
            combi_2 = (period - 1, period, period + 1, period + 2, period + 3)
            combi_3 = (period - 2, period - 1, period, period + 1, period + 2)
            combi_4 = (period - 3, period - 2, period - 1, period, period + 1)
            combi_5 = (period - 4, period - 3, period - 2, period - 1, period)

            tmp = [combi_1, combi_2, combi_3, combi_4, combi_5]
            combinations = []
            for c in tmp:
                if c[0] > 0 and c not in combinations:
                    if c[-1] <= 48:
                        combinations.append(c)
        elif self.status == 'Discharge':
            combi_1 = (period, period + 1, period + 2, period + 3)
            combi_2 = (period - 1, period, period + 1, period + 2)
            combi_3 = (period - 2, period - 1, period, period + 1)
            combi_4 = (period - 3, period - 2, period - 1, period)
            
            tmp = [combi_1, combi_2, combi_3, combi_4]
            combinations = []
            for c in tmp:
                if c[0] > 0 and c not in combinations:
                    if c[-1] <= 48:
                        combinations.append(c)
                
        return combinations
        