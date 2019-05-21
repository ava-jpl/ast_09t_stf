#!/usr/bin/env python

'''
ASTER 09T TIR temperature fitting algoritm using SciPy
'''

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.optimize import curve_fit
import pylab
import random
import math


AVERAGE_BACKGROUND_TEMP = 271  # will change for ASTER inputs
EMISSIVITY = 0.97
T1_LOWER_BOUND = 240  # degrees K
T1_UPPER_BOUND = 1411  # degrees K
T2_LOWER_BOUND = 273  # degrees K
T2_UPPER_BOUND = 330  # degrees K
MINIMUM_ALLOWED_AREA = 0  # meters ^2
# minimum radiance percentage of observable radiance in order to be used
# as a two temperature fit
RADIANCE__THRESHOLD_PERCENTAGE = 100
# minimum K to be over the background temperature in order for a two
# temperature fit to be ran
MINIMUM_TEMPERATURE_DELTA_THRESHOLD = 100

ASTER_BOUNDS = [8.125, 11.650]  # minimum and maximum observable wavelength
AREA_OF_PIXEL = 8100  # area in meters^2 of ASTER pixel

# calc maximum allowed area based on the value above
MAX_A = AREA_OF_PIXEL - MINIMUM_ALLOWED_AREA


class temp_fit():

    def __init__(self, radiance_values, wavelength_values=[8.2910, 8.6340, 9.0750, 10.6570, 11.3180]):
        # initialize input values
        self.radiance_values = radiance_values
        self.wavelength_values = wavelength_values
        self.is_hot_pixel = False

    def run_two_temp_fitting(self):
        '''
        runs an iterative two temperature fitting algorithm, and also calculates
        whether the pixel has a 'hot' component, calculating the emissivity and associated
        products if applicable
        '''
        p, c = self.fit()
        # save results
        if p[1] > p[0]:
            p[0], p[1] = p[1], p[0]
        self.t1 = p[0]
        self.t2 = p[1]
        self.a1 = p[2]
        self.a2 = 8100 - p[2]
        self.covariance_values = c[0]
        self.calculate_total_radiances()  # calculate radiance values
        if self.is_hot():
            self.calculate_convection()  # calculate convection values
            self.calculate_2_5_ratio()  # calculate the spectral ratios
            # self.print_results()

    def fit(self):
        '''
        runs a two temperature fitting algorithm
        '''
        xdata = self.wavelength_values
        ydata = self.radiance_values
        # weigh our outer points more
        sig = np.ones_like(ydata)
        sig[1] = 2
        sig[-1] = 2
        # p0 are initial values
        return curve_fit(temp_fit.blackbody, xdata, ydata, bounds=([T1_LOWER_BOUND, T2_LOWER_BOUND, MINIMUM_ALLOWED_AREA], [T1_UPPER_BOUND, T2_UPPER_BOUND, MAX_A]), p0=[T1_LOWER_BOUND+1, T2_LOWER_BOUND+1, 4000], sigma=sig, max_nfev=1000000)

    def run_one_temp_fitting(self):
        '''runs a simple iterative one temperature fitting algorithm, and
        returns only the temperature of that pixel (as a float)
        '''
        xdata = self.wavelength_values
        ydata = self.radiance_values
        # weigh our points less likely to be nearly saturated more (since this should look at high temp)
        sig = np.ones_like(ydata)
        sig[-2] = 2
        sig[-1] = 2
        p, c = curve_fit(temp_fit.one_temp_blackbody, xdata, ydata, bounds=(
            [T1_LOWER_BOUND], [T1_UPPER_BOUND]), p0=[270], sigma=sig, max_nfev=1000000)
        self.t = p[0]
        return p[0]

    def calculate_total_radiances(self):
        '''calculate total thermal radiance for the pixel and individual radiances, as well as radiances for the wavelength range'''
        # total t1 radiant emittance (watts)
        self.t1_emittance = self.emittance(self.t1, self.a1)
        self.t2_emittance = self.emittance(self.t2, self.a2)
        self.total_emittance = self.t1_emittance + self.t2_emittance
        # determine emittance over wavelength bands (integrate over observed portion)
        # only generate the tir emittance (requires significant processing) if
        # the temp is above threshold
        if self.t1 > self.t2 + MINIMUM_TEMPERATURE_DELTA_THRESHOLD:
            self.t1_tir_emittance = self.emittance_range(
                ASTER_BOUNDS[0], ASTER_BOUNDS[1], self.t1, self.a1)
            self.t2_tir_emittance = self.emittance_range(
                ASTER_BOUNDS[0], ASTER_BOUNDS[1], self.t2, self.a2)

    @staticmethod
    def emittance(t, a):
        # calculates total radiant emittance
        # TOT = ( 2 pi^5 k^4 t^4 ) /( 15 h^3 c^2)
        # boltzmann constant (J / K)
        k = float(1.38064852) * math.pow(float(10), -23)
        # planck constant (J s)
        h = float(6.626070040) * math.pow(float(10), -34)
        c = 299792458  # speed of light (m / s)
        tot = (2 * math.pow(math.pi, 5) * math.pow(k, 4) *
               math.pow(t, 4)) / (15 * math.pow(h, 3) * math.pow(c, 2)) * a
        return tot

    def emittance_range(self, lower_bound_wavelength, upper_bound_wavelength, temp, area):
        '''numerically calculate the emittance over the ASTER TIR range'''
        b = 300  # break into 300 pieces
        # size (of the wavelength range
        d = (upper_bound_wavelength - lower_bound_wavelength) / b
        x_list = np.linspace(lower_bound_wavelength, upper_bound_wavelength, b)
        radiance_list = self._one_temp_blackbody_plot(x_list, temp, area)
        e_list = [x * d for x in radiance_list]
        emittance = sum(e_list)
        return emittance

    def calculate_convection(self):
        '''calculate the forced and natural convection for the pixel'''
        ws = 1.0  # wind speed
        f = 0.0036  # friction factor
        Tenv = AVERAGE_BACKGROUND_TEMP  # environmental temperature
        # Surface temperature (using high temperature component) ??? (does this
        # make sense)
        Tsurf = self.t1
        Area_high_temp = self.a1  # area of the high temperature component
        # effective atmospheric temperature ??? (does this make sense)
        Teff = (Tsurf + Tenv) / 2
        p = 360.77819 * math.pow(Teff, -1.00336)  # Atmospheric density
        A = 0.14  # dimensionless value for hot plate
        g = 9.81  # gravity
        o = 5.67 * math.pow(10, -8)  # Stephan Boltzmann constant
        e = EMISSIVITY  # emissivity
        # determine c1 and c2
        c1 = 0.016201807  # between 97 to 247K
        c2 = 0.994363318
        if Teff > 248 and Teff < 433:
            c1 = 0.0088512
            c2 = 0.99682
        elif Teff >= 433:
            c1 = 0.0049086
            c2 = 0.998216
        B = c1 * math.pow(c2, Teff)  # Volume coefficient expansion
        cp = (1.9327 * math.pow(10, -10.0) * math.pow(Teff, 4)) - (7.9999 * math.pow(10, -7.0) * math.pow(Teff, 3)) + \
            (1.1407 * math.pow(10, -3.0) * math.pow(Teff, 2)) - \
            (0.4489 * Teff) + 1057.5  # specific heat capacity
        k = (1.5207 * math.pow(10, -11.0) * math.pow(Teff, 3)) - (4.8574 * math.pow(10, -8.0) * math.pow(Teff, 2)) + \
            (1.0184 * math.pow(10, -4.0) * Teff) - \
            0.00039333  # thermal conductivity of the atmosphere
        n = (-1.1555 * math.pow(10, -14.0) * math.pow(Teff, 3) + 9.5728 * math.pow(10, -11.0) * math.pow(Teff, 2) +
             3.7604 * math.pow(10, -8.0) * Teff - 3.4484 * math.pow(10, -6)) * p  # viscosity of atmosphere

        Fforce = ws * f * cp * p * (Tsurf - Tenv) / \
            1000 * Area_high_temp  # Forced convection
        Fnat = (A * math.pow(((cp * math.pow(p, 2) * g * B * (Tsurf - Tenv)) /
                              (n * k)), (1.0 / 3.0)) * ((Tsurf - Tenv) * k)) / 1000 * Area_high_temp  # natural convection
        Fconv = Fnat
        if Fforce > Fconv:
            Fconv = Fforce  # Fconv is the larger of the two
        Frad = o * e * (math.pow(Tsurf, 4) - math.pow(Tenv, 4)
                        ) / 1000 * Area_high_temp
        Ftot = Fconv + Frad

        # save vars
        self.p = p
        self.Teff = Teff
        self.Tsurf = Tsurf
        self.B = B
        self.cp = cp
        self.k = k
        self.n = n
        self.Fforce = Fforce
        self.Fnat = Fnat
        self.Fconv = Fconv
        self.Frad = Frad
        self.Ftot = Ftot

    def print_convection(self):
        '''prints convection results'''
        if self.is_hot_pixel:
            print('Tsurf:', self.Tsurf)
            print('Teff:', self.Teff)
            print('p:', self.p)
            print('B:', self.B)
            print('cp:', self.cp)
            print('k:', self.k)
            print('n:', self.n)
            print('Fforce:', self.Fforce, 'kW')
            print('Fnat:', self.Fnat, 'kW')
            print('Frad:', self.Frad, 'kW')
            print('Ftot:', self.Ftot, 'kW')

    def calculate_2_5_ratio(self):
        # calculate two and 5 micron spectral radiances and ratios ONLY FOR THE
        # HOT COMPONENT
        x = [2, 5]
        rad = self._one_temp_blackbody_plot(x, self.t1, self.a1)
        self.two_micron_radiance = rad[0]
        self.five_micron_radiance = rad[1]
        self.two_five_ratio = rad[0] / rad[1]

    def print_ratios(self):
        if self.is_hot_pixel:
            print('two micron radiance:', self.two_micron_radiance)
            print('five micron radiance:', self.five_micron_radiance)
            print('two five ratio:', self.two_five_ratio)

    @staticmethod
    def blackbody(x, T1, T2, A1):
        '''generate the blackbody function for fitting'''
        c1 = float(3.7413 * 10**8)  # W um^4 m^-2
        c2 = float(1.4388 * 10**4)  # um K
        e = EMISSIVITY
        #A2 = (8100 - A1)
        return np.divide((e * A1 * c1), np.power(x, 5) * (np.exp(np.divide(c2, (np.multiply(x,  T1)))) - 1)) + np.divide((e * (8100 - A1) * c1), np.power(x, 5) * (np.exp(np.divide(c2, (np.multiply(x,  T2)))) - 1))

    @staticmethod
    def one_temp_blackbody(x, T):
        '''generate the blackbody function for fitting for only one temperature'''
        c1 = float(3.7413 * 10**8)  # W um^4 m^-2
        c2 = float(1.4388 * 10**4)  # um K
        e = EMISSIVITY
        A = AREA_OF_PIXEL
        return np.divide((e * A * c1), np.power(x, 5) * (np.exp(np.divide(c2, (np.multiply(x,  T)))) - 1))

    def print_results(self):
        print('temperature/area fits: t1:%s t2:%s a1:%s a2:%s' % (str(int(self.t1)).ljust(5, ' '),
                                                                  str(int(self.t2)).ljust(5, ' '), str(int(self.a1)).ljust(5, ' '), str(int(self.a2)).ljust(5, ' ')))
        print('emittances: t1:%s t2:%s') % (str(int(self.t1_emittance)).ljust(
            10, ' '), str(int(self.t2_emittance)).ljust(10, ' '))
        #print ('covariance values: %s' % self.covariance_values)
        print('tir emittances: t1:%s t2:%s') % (str(int(self.t1_tir_emittance)).ljust(
            10, ' '), str(int(self.t2_tir_emittance)).ljust(10, ' '))
        self.print_convection()
        self.print_ratios()

    @staticmethod
    def _blackbody_plot(x, T1, T2, A1):
        '''used to create blackbody plots'''
        # y = empty_like(x):
        y = []
        for xval in x:
            c1 = float(3.7413 * 10**8)  # W um^4 m^-2
            c2 = float(1.4388 * 10**4)  # um K
            e = EMISSIVITY
            #A2 = (8100 - A1)
            yval = np.divide((e * A1 * c1), np.power(xval, 5) * (np.exp(np.divide(c2, (np.multiply(xval,  T1)))) - 1)) + \
                np.divide((e * (8100 - A1) * c1), np.power(xval, 5) *
                          (np.exp(np.divide(c2, (np.multiply(xval,  T2)))) - 1))
            y.append(yval)
        return y

    @staticmethod
    def _one_temp_blackbody_plot(x, T1, A1):
        '''used to create blackbody plots'''
        # y = empty_like(x):
        y = []
        for xval in x:
            c1 = float(3.7413 * 10**8)  # W um^4 m^-2
            c2 = float(1.4388 * 10**4)  # um K
            e = EMISSIVITY
            #A2 = (8100 - A1)
            yval = np.divide((e * A1 * c1), np.power(xval, 5) *
                             (np.exp(np.divide(c2, (np.multiply(xval,  T1)))) - 1))
            y.append(yval)
        return y

    def save_plot(self, full_path_to_plot='plot.png'):
        xdata = self.wavelength_values
        ydata = self.radiance_values

        # plot the input ASTER data points
        pylab.plot(xdata, ydata, 'r+', mew=6, ms=6)

        # plot the fit
        plotx = np.linspace(8, 12, 300)  # x goes from 8 to 12 microns
        # two temp plot
        ploty = temp_fit._blackbody_plot(plotx, self.t1, self.t2, self.a1)
        pylab.plot(plotx, ploty, 'b.')
        # t1 plot
        ploty = temp_fit._one_temp_blackbody_plot(plotx, self.t1, self.a1)
        pylab.plot(plotx, ploty, 'g.')
        # t2 plot
        ploty = temp_fit._one_temp_blackbody_plot(
            plotx, self.t2, 8100 - self.a1)
        pylab.plot(plotx, ploty, 'y.')

        pylab.title('T1=' + str(self.t1)[:5] + 'K, T2=' + str(self.t2)[
                    :5] + 'K, A1=' + str(self.a1)[:5] + ', A2=' + str(self.a2)[:5])
        # pylab.show()
        pylab.savefig(full_path_to_plot)
        pylab.close()

    def is_hot(self):
        '''returns true if it's a hot pixel, (above the threshold) false otherwise.'''
        # parses the temp results and ouputs to csv if over valid value
        if self.is_hot_pixel == True:
            return True
        if not self.t1 > self.t2 + MINIMUM_TEMPERATURE_DELTA_THRESHOLD:
            return False
        # emittance must be over 10% of low temp
        if not (self.t1_tir_emittance / self.t2_tir_emittance) > (float(RADIANCE__THRESHOLD_PERCENTAGE) / 100):
            return False
        self.is_hot_pixel = True
        return True


if __name__ == '__main__':
    # wavelength = [8.2910, 8.6340, 9.0750, 10.6570, 11.3180] #wavelengths
    #radiance = [247277.3037, 246890.409, 248188.7099, 246277.6477, 237865.5385]
    radiance = [597159,435032,396650,317210,275927]
    
    t = temp_fit(radiance)
    t.run_two_temp_fitting()
    t.calculate_2_5_ratio()
    t.print_results()
    # t.save_plot()
    # t.calculate_convection()
