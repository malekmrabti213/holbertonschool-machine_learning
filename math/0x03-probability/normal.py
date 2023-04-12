#!/usr/bin/env python3

class Normal():

    def __init__(self, data=None, mean=0., stddev=1.):

        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                sigma = 0
                for i in range(0, len(data)):
                    x = (data[i] - self.mean) ** 2
                    sigma += x
                self.stddev = (sigma / len(data)) ** (1 / 2)

    def z_score(self, x):
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        return self.stddev * z + self.mean
    
    π = 3.1415926536
    e = 2.7182818285

    def pdf(self, x):

        p1 = 1 / (self.stddev * ((2 * Normal.π) ** 0.5))
        p2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return p1 * Normal.e ** (-p2)
    
    def cdf(self, x):

        stand_x = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        erf = (((4 / Normal.π) ** 0.5) * (stand_x - (stand_x ** 3) / 3 +
                                             (stand_x ** 5) / 10 - (stand_x ** 7) / 42 +
                                             (stand_x ** 9) / 216))
        cdf = (1 + erf) / 2
        return cdf