import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def make_gaussian_fit(ax_hist):
    x, y = (ax_hist[1][1:]+ax_hist[1][:-1])/2, ax_hist[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[1,1,1])
    return x, popt
