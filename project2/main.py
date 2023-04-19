"""
Author: Mads S. Balto
Description:
Models radiative and convective energy transport of a Sun-like star.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from project1 import EnergyProduction
STEFAN_BOLTZMANN_CONSTANT = 5.6704e-8  # W / (m^2 K^4)
GRAVITATIONAL_CONSTANT = 6.6742e-11  # N m^2 / kg^2
BOLTZMANN_CONSTANT = 1.3806e-23  # m^2 kg / (s^2 K)
ATOMIC_MASS_UNIT = 1.6605e-27  # kg
ADIABATIC_TEMPERATURE_GRADIENT = 2/5
DELTA = 1


def compute_mean_molecular_mass():
    """
    Compute the mean molecular mass.

    Returns:
        mean molecular mass.       
    """
    MASS_FRACTION_HYDROGEN = 0.7
    MASS_FRACTION_HELIUM_4 = 0.29
    MASS_FRACTION_HELIUM_3 = 1e-10
    MASS_FRACTION_LITHIUM_7 = 1e-7
    MASS_FRACTION_BERYLLIUM_7 = 1e-7
    MASS_FRACTION_NITROGEN_14 = 1e-11
    return 1/np.sum(
        1/2*MASS_FRACTION_HYDROGEN,
        2/3*MASS_FRACTION_HELIUM_3,
        2/4*MASS_FRACTION_HELIUM_4,
        3/7*MASS_FRACTION_LITHIUM_7,
        4/7*MASS_FRACTION_BERYLLIUM_7,
        7/14*MASS_FRACTION_NITROGEN_14
    )


MEAN_MOLECULAR_MASS = compute_mean_molecular_mass()


def compute_specific_heat_capacity():
    """
    Computes the specific heat capacity.

    Returns:
        (float) Specific heat capacity.
    """
    return 5*BOLTZMANN_CONSTANT/2/MEAN_MOLECULAR_MASS/ATOMIC_MASS_UNIT


SPECIFIC_HEAT_CAPACITY = compute_specific_heat_capacity()


def rmo_to_mass_density(temperature, rosseland_mean_opacity):
    """
    Calculates the mass density given the temperature and Rosseland mean opacity.

    Args:
        temperature (float): The temperature in units of Kelvin.
        rosseland_mean_opacity (array-like of dtype float): The Rosseland mean opacity 
            in units of cm^2/g.

    Returns:
        mass_density (float): The mass density in units of g/cm^3
    """
    return rosseland_mean_opacity*(temperature*1e-6)**2


def compute_radial_coordinate_differential(radial_position, mass_density):
    """
    Computes the differential of the radial coordinate with respect to mass,
    for a point at a given radial position within a sun-like star.

    Args:
        radial_position (float): The radial position of the point within the 
        star, measured in units of distance from the star's center.
        mass_density (float): The local mass density at the point within the
        star, measured in units of mass per unit volume.

    Returns:
        float: The differential of the radial coordinate with respect to mass,
            measured in units of distance per unit mass.
    """
    return 1 / (4 * np.pi * radial_position**2 * mass_density)


def compute_opacity(temperature, mass_density):
    """
    Computes mass density as functions of the rmo and 
    temperature using data from "opacity.txt". Interpolates opacity as
    functions of temperature and mass density.

    Args:
        temperature (float) Temperature in Kelvin.
        mass_density (float) Mass density in kg/m^3

    Returns:
        opacity (float) Opacity [m^2/kg] of sun-like star, as a function
        of temperature and pressure, per interpolation of data
        from 'opacity.txt'.
    """
    data = np.genfromtxt("opacity.txt")
    LOG_10_ROSSELAND_MEAN_OPACITY = data[0][1:]  # g/cm^3
    ROSSELAND_MEAN_OPACITY = 10**(LOG_10_ROSSELAND_MEAN_OPACITY)
    LOG_10_TEMPERATURES = data[1:][0]  # K
    TEMPERATURES = 10**(LOG_10_TEMPERATURES)
    LOG_10_OPACITY = data[1:][1:]  # g/cm^2
    OPACITY = 10**(LOG_10_OPACITY)
    mass_densities = np.empty_like(LOG_10_ROSSELAND_MEAN_OPACITY)
    for i, rmo in enumerate(ROSSELAND_MEAN_OPACITY):
        mass_densities[i] = rmo_to_mass_density(TEMPERATURES[i], rmo)
    try:
        opacity_interp = RectBivariateSpline(
            TEMPERATURES, mass_densities, OPACITY, bounds_error=True)
    except:
        opacity_interp = RectBivariateSpline(
            TEMPERATURES, mass_densities, OPACITY, bounds_error=False)
        print("""
WARNING: compute_opacity accessed with parameter(s) exceeding interpolation bounds
""")
    return opacity_interp(temperature, mass_density)


def pressure_to_density(temperature, pressure):
    """
    Compute the mass density given some temperature and pressure.
    Args:
        temperature (float): The temperature [K].
        pressure (float): The pressure in units of Pa. [N/m^2]

    Returns:
        (float) mass density in units of kg/m^3
    """
    return (MEAN_MOLECULAR_MASS*ATOMIC_MASS_UNIT*pressure
            / BOLTZMANN_CONSTANT/temperature)


def density_to_pressure(temperature, mass_density):
    """
    Compute the pressure given some temperature and mass density.

    Args:
        temperature (float): The temperature [K]
        mass_density (float): The mass density [kg/m^3]
    returns:
        (float) pressure [?]
    """
    return (mass_density*BOLTZMANN_CONSTANT*temperature
            / MEAN_MOLECULAR_MASS/ATOMIC_MASS_UNIT)


def compute_pressure_differential(mass, radial_position):
    """
    Computes the differential of pressure with respect to mass.

    Args:
        mass (float): Mass
        radial_position (float): The radial position relative to center of a
            sun-like star. [m]
    
    Returns:
        (float): Differential of pressure with respect to mass.
    """
    return GRAVITATIONAL_CONSTANT*mass/4/np.pi/radial_position**4


def total_energy_production_rate(temperature, mass_density):
    """
    Computes the total energy production rate. (dL/dm)
    Args:
        temperature (float): Temperature [K].
        mass_density (float): Mass density [kg/m^3],

    returns:
        (float): The total energy production rate. []
    """
    ENERGY_PRODUCTION_INSTANCE = EnergyProduction(mass_density, temperature)
    return ENERGY_PRODUCTION_INSTANCE.total_energy_production_rate()


def compute_actual_temperature_gradient(temperature, mass_density, mass,
                                        radial_position, pressure, luminosity):
    """
    Computes the actual temperature gradient, by sequentially computing the
    following quantities:

    1. gravitational acceleration,

    2. pressure scale height,

    3. geometric factor,

    4. Computes the unnamed but useful quantity U,
    
    6. temperature gradient needed for all the energy to be carried
    by radiation,

    then solves a third order polynomial numerically for xi, and then
    computes the actual temperature gradient.

    Args:
        temperature (float): Temperature [K]
        mass_density (float): mass density [kg/m^3]
        mass (float): mass of ? [kg]
        radial_position (float): radial position relative to center of a
            sun-like star. [m]
        pressure (float): pressure [?]
        luminosity (float): ? [?]
    
    Returns:
        (float): Actual temperature gradient
    """
    GRAVITATIONAL_ACCELERATION = (
        mass*GRAVITATIONAL_CONSTANT/radial_position**2)
    PRESSURE_SCALE_HEIGHT = (
        -pressure*GRAVITATIONAL_CONSTANT*mass/mass_density/radial_position**2
    )
    GEOMETRIC_FACTOR = 4/DELTA/PRESSURE_SCALE_HEIGHT
    OPACITY = compute_opacity(temperature, mass_density)
    u = (
        64*STEFAN_BOLTZMANN_CONSTANT*temperature**3
        * SPECIFIC_HEAT_CAPACITY
        * np.sqrt(PRESSURE_SCALE_HEIGHT/GRAVITATIONAL_ACCELERATION/DELTA)
        / 3/mass_density**2/SPECIFIC_HEAT_CAPACITY/OPACITY
    )
    TEMPERATURE_GRADIENT_STABLE = (
        3*luminosity*OPACITY*mass_density*PRESSURE_SCALE_HEIGHT/4/np.pi /
        radial_position**2/16/STEFAN_BOLTZMANN_CONSTANT/temperature**4)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    coeffs = np.asarray([
        1,
        u/MIXING_LENGTH**2,
        u**2*GEOMETRIC_FACTOR/MIXING_LENGTH**3,
        u/MIXING_LENGTH**2*(TEMPERATURE_GRADIENT_STABLE - ADIABATIC_TEMPERATURE_GRADIENT)])
    roots = np.roots(coeffs)
    xi = np.real_if_close(roots)
    assert len(xi) == 1, "1 element expected, condition not met."
    actual_temperature_gradient = xi**2 - ADIABATIC_TEMPERATURE_GRADIENT
    return actual_temperature_gradient


class EnergyTransport:
    """ 
    Models central parts of a Sun-like star.
    """

    def __init__(self, ):
        """
        
        """
        INIT_LUMINOSITY = 3.846e26  # kg m^2 /s^3  (W)
        INIT_RADIUS = 6.96e8  # m
        INIT_MASS = 1.989e30  # kg
        AVERAGE_MASS_DENSITY_SUN = 1.408e3  # kg/m^3
        INIT_MASS_DENSITY = 1.42e-7*AVERAGE_MASS_DENSITY_SUN  # kg/m^3
        INIT_TEMPERATURE = 5770  # K
        INIT_TOTAL_ENERGY_PRODUCTION_RATE = total_energy_production_rate(
            INIT_TEMPERATURE, INIT_MASS_DENSITY)

    def compute_convective_flux(self, mass_density, temperature):
        """
        Computes convective flux:

        Args:
            mass_density (float): mass density [kg/m^3],
            temperature (float): temperature [K].

        returns:
            (float) Convective flux [?]
        """
        pass


instance = EnergyTransport()
instance.compute_opacity
