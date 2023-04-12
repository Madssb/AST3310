"""
Defines the radiation pressure [Pa], and computes it for the temperature [K] 
of the solar core.
"""
RADIATION_CONSTANT = 7.6e-16 # J/m^3/k^4
STEFAN_BOLTZMANN = 5.67e-8 # W/m^2/K^4
SPEED_OF_LIGHT = 3e8 #m/s
T_CORE = 1.57e7 #K
TEMPERATURE_GRADIENT_SUN = 2.2e-2 #K/m
def radiation_pressure(temperature):
    """
    Computes the radiation pressure [Pa] forr a temperature [K]
    """
    return RADIATION_CONSTANT/3*temperature**4


def pressure_gradient(temperature):
    """
    compute the pressure gradient [Pa/m] for the sun for some temperature [K].
    """
    return RADIATION_CONSTANT/3*4*temperature**3*TEMPERATURE_GRADIENT_SUN


def mean_molecular_weight(x,y,z):
    """
    compute the mean molecular weight

    parameters:
    x: mass fraction of hydrogen 1,
    y: mass fraction of helium 4,
    z: mass fraction of other metals.
    """


print(f"temperature gradient for sun: {pressure_gradient(T_CORE):.4g} Pa/m")
print(f"Radiation pressure for sun core temperature = {radiation_pressure(T_CORE):.4g}Pa")

