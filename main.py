import numpy as np
from scipy.stats import norm

class BsOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.q = q
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r': self.r,
                'q': self.q,
                'sigma': self.sigma}
    
    def d1(self):
        return (np.log(self.S / self.K) + (self.r - self.q + self.sigma**2 / 2) * self.T) \
                                / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def _call_value(self):
        return self.S * np.exp(-self.q * self.T) * self.N(self.d1()) - \
                    self.K * np.exp(-self.r * self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K * np.exp(-self.r * self.T) * self.N(-self.d2()) -\
                self.S * np.exp(-self.q * self.T) * self.N(-self.d1())
    
    def price(self, type_='C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value()
        if type_ == 'B':
            return {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
    
    def delta(self, type_='C'):
        d1 = self.d1()
        if type_ == 'C':
            return np.exp(-self.q * self.T) * self.N(d1)
        if type_ == 'P':
            return np.exp(-self.q * self.T) * (self.N(d1) - 1)
    
    def gamma(self):
        d1 = self.d1()
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, type_='C'):
        d1 = self.d1()
        d2 = self.d2()
        if type_ == 'C':
            return (-self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - \
                   self.r * self.K * np.exp(-self.r * self.T) * self.N(d2) + self.q * self.S * np.exp(-self.q * self.T) * self.N(d1)
        if type_ == 'P':
            return (-self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1)) / (2 * np.sqrt(self.T)) + \
                   self.r * self.K * np.exp(-self.r * self.T) * self.N(-d2) - self.q * self.S * np.exp(-self.q * self.T) * self.N(-d1)
    
    def vega(self):
        d1 = self.d1()
        return self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(d1)
    
    def rho(self, type_='C'):
        d2 = self.d2()
        if type_ == 'C':
            return self.K * self.T * np.exp(-self.r * self.T) * self.N(d2)
        if type_ == 'P':
            return -self.K * self.T * np.exp(-self.r * self.T) * self.N(-d2)
    
    def implied_volatility(self, option_price, type_='C'):
        # Use a numerical method to find the implied volatility
        tolerance = 1e-5
        max_iter = 1000
        lower_vol = 0.001
        upper_vol = 5.0
        
        for i in range(max_iter):
            mid_vol = (lower_vol + upper_vol) / 2
            self.sigma = mid_vol
            
            if type_ == 'C':
                option_price_est = self._call_value()
            elif type_ == 'P':
                option_price_est = self._put_value()
            
            error = option_price - option_price_est
            
            if abs(error) < tolerance:
                return mid_vol
            
            if error < 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
        
        raise Exception("Implied volatility not found within the specified tolerance.")
    
    def option_strategy_price(self, strategy_type):
        if strategy_type == 'Straddle':
            # Straddle: Buy a call and a put with the same strike and expiration
            call_price = self._call_value()
            put_price = self._put_value()
            return call_price + put_price
        elif strategy_type == 'Strangle':
            # Strangle: Buy a call and a put with different strikes but the same expiration
            # You can customize the strikes here
            call_strike = self.K + 10  # Modify this as needed
            put_strike = self.K - 10   # Modify this as needed
            call_option = BsOption(self.S, call_strike, self.T, self.r, self.sigma, self.q)
            put_option = BsOption(self.S, put_strike, self.T, self.r, self.sigma, self.q)
            call_price = call_option._call_value()
            put_price = put_option._put_value()
            return call_price + put_price
        elif strategy_type == 'CalendarSpread':
            # Calendar Spread: Buy a longer-term call and sell a shorter-term call with the same strike
            # You can customize the expiration of the short and long call options here
            long_option = BsOption(self.S, self.K, self.T + 0.5, self.r, self.sigma, self.q)
            short_option = BsOption(self.S, self.K, self.T, self.r, self.sigma, self.q)
            long_price = long_option._call_value()
            short_price = short_option._call_value()
            return long_price - short_price
        else:
            raise ValueError('Unrecognized strategy type')

if __name__ == '__main__':
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3
    S = 100
    
    # Create an instance of BsOption
    option = BsOption(S, K, T, r, sigma)
    
    # Calculate and print the price of a Call, Put, and Both
    call_price = option.price('C')
    put_price = option.price('P')
    both_prices = option.price('B')
    print(f'Call Price: {call_price}')
    print(f'Put Price: {put_price}')
    print(f'Both Prices: {both_prices}')
    
    # Calculate and print the Greeks for a Call option
    call_delta = option.delta('C')
    call_gamma = option.gamma()
    call_theta = option.theta('C')
    call_vega = option.vega()
    call_rho = option.rho('C')
    print(f'Call Delta: {call_delta}')
    print(f'Call Gamma: {call_gamma}')
    print(f'Call Theta: {call_theta}')
    print(f'Call Vega: {call_vega}')
    print(f'Call Rho: {call_rho}')
    
    # Calculate and print the implied volatility for a Call option
    option_price = 10.0  # Replace with the actual option price
    implied_volatility = option.implied_volatility(option_price, 'C')
    print(f'Implied Volatility: {implied_volatility}')
    
    # Calculate and print the price of option strategies
    straddle_price = option.option_strategy_price('Straddle')
    strangle_price = option.option_strategy_price('Strangle')
    calendar_spread_price = option.option_strategy_price('CalendarSpread')
    print(f'Straddle Price: {straddle_price}')
    print(f'Strangle Price: {strangle_price}')
    print(f'Calendar Spread Price: {calendar_spread_price}')
