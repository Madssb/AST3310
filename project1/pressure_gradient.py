"""
Compute the pressure gradient [Pa/m] for the sun.
"""
A = 7.6e-16 # J/m^3/k^4
STEFAN_BOLTZMANN = 5.67e-8 # W/m^2/K^4
SPEED_OF_LIGHT = 3
T_CORE = 1.57e7
TEMPERATURE_GRADIENT_SUN = 2.2e-2
def pressure_gradient(temperature):
    """
    compute the pressure gradient [Pa/m] for the sun for some temperature [K].
    """
    return A/3*4*temperature**3*TEMPERATURE_GRADIENT_SUN


print(f"temperature gradient for sun: {pressure_gradient(T_CORE):.4g} Pa/m")