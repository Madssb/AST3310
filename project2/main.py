"""
Author: Mads S. Balto
Description:
Models radiative and convective energy transport of a Sun-like star.
"""
import numpy as np

class EnergyTransport:
    def __init__(self):
        INIT_LUMINOSITY = 3.846e26 #W
        INIT_RADIUS = 6.96e8 #m
        INIT_MASS = 1.989e30 #kg
        AVERAGE_MASS_DENSITY_SUN = 1.408e3 #kg/m^3
        INIT_MASS_DENSITY = 1.42e-7*AVERAGE_MASS_DENSITY_SUN #kg/m^3
        INIT_TEMPERATURE = 5770 #K
        MASS_FRACTION_HYDROGEN = 0.7
        MASS_FRACTION_HELIUM_4 = 0.29
        MASS_FRACTION_HELIUM_3 = 1e-10
        MASS_FRACTION_LITHIUM_7 = 1e-7
        MASS_FRACTION_BERYLLIUM_7 = 1e-7
        MASS_FRACTION_NITROGEN_14 = 1e-11


    def read(self):
        data = np.loadtxt("opacity.txt",skiprows=1)
        print(f"{np.shape(data)=}")



instance = EnergyTransport()
instance.read()