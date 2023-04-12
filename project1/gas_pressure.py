"""
Computs the gas pressure [Pa] for the conditions of the sun,
i.e. temperature [K], mass density [kg/m^3] and 
"""

ATOMIC_MASS_UNIT = 1.66e-27 #kg
BOLTZMANN_CONSTANT = 1.38e-23 #J/K
TEMPERATURE_SUN_CORE = 1.57e7 #K
MASS_DENSITY_SUN_CORE = 1.62e5 #kg/m^3
TEMPERATURE_GRADIENT_SUN = 2.2e-2 #K/M
MEAN_MOLECULAR_WEIGHT_SUN = 0.6

def gas_pressure(
    temperature = TEMPERATURE_SUN_CORE,
    mass_density = MASS_DENSITY_SUN_CORE):
    """
    compute the gas pressure [Pa] for some temperature [K] and mass density
    [kg/m^3]. The default values are specified as those for the sun.
    """
    return (
        mass_density
        * BOLTZMANN_CONSTANT
        * temperature
        / MEAN_MOLECULAR_WEIGHT_SUN
        / ATOMIC_MASS_UNIT
    )

def gas_pressure_gradient(
    mass_density = MASS_DENSITY_SUN_CORE, 
    temperature_gradient = TEMPERATURE_GRADIENT_SUN,
    mean_molecular_weight = MEAN_MOLECULAR_WEIGHT_SUN):
    """
    compute the pressure gradient by gas [Pa] in a star with some temperature 
    [K], mass density [kg/m^3] and some mean molecular weight. Default values
    are specified as those for the sun.
    """
    return (
        mass_density
        * BOLTZMANN_CONSTANT
        / mean_molecular_weight
        / ATOMIC_MASS_UNIT
        * temperature_gradient
    )


print(
f"""
gas pressure in sun: {gas_pressure()}Pa
gas pressure gradient: {gas_pressure_gradient()}Pa/m
"""
)