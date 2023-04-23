"""
Author: Mads S. Balto
Description:
Models radiative and convective energy transport of a Sun-like star.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from project1 import EnergyProduction
import os
import matplotlib.pyplot as plt
from cross_section import cross_section
import sys
from tabulate import tabulate

STEFAN_BOLTZMANN_CONSTANT = 5.6704e-8  # W / (m^2 K^4)
GRAVITATIONAL_CONSTANT = 6.6742e-11  # N m^2 / kg^2
BOLTZMANN_CONSTANT = 1.3806e-23  # m^2 kg / (s^2 K)
ATOMIC_MASS_UNIT = 1.6605e-27  # kg
ADIABATIC_TEMPERATURE_GRADIENT = 2 / 5
DELTA = 1
LUMINOSITY_SUN = 3.846e26  # kg m^2 /s^3  (W)
MASS_SUN = 1.989e30  # kg
RADIUS_SUN = 6.96e8  # m
AVERAGE_MASS_DENSITY_SUN = 1.408e3  # kg/m^3


def mean_molecular_mass():
    """
    Computes the mean molecular mass.

    Returns:
        mean molecular mass.
    """
    free_particles = {"hydrogen_1": 2, "helium_3": 3, "helium_4": 3, "lithium_7": 4,
                      "beryllium_7": 5, "nitrogen_14": 8}
    mass_fraction = {"hydrogen_1": 0.7, "helium_3": 1e-10, "helium_4": 0.29,
                     "lithium_7": 1e-7, "beryllium_7": 1e-7, "nitrogen_14": 1e-11}
    nucleons = {"hydrogen_1": 1, "helium_3": 3, "helium_4": 4, "lithium_7": 7,
                "beryllium_7": 7, "nitrogen_14": 14}
    isotopes = free_particles.keys()
    sum = 0
    for isotope in isotopes:
        sum += free_particles[isotope]/nucleons[isotope]*mass_fraction[isotope]
    return 1/sum


MEAN_MOLECULAR_MASS = mean_molecular_mass()


def specific_heat_capacity():
    """
    Computes the specific heat capacity.

    Returns:
        (float) Specific heat capacity.
    """
    return 5 * BOLTZMANN_CONSTANT / 2 / MEAN_MOLECULAR_MASS / ATOMIC_MASS_UNIT


SPECIFIC_HEAT_CAPACITY = specific_heat_capacity()


def radial_coordinate_differential(radial_position, mass_density):
    """
    Computes the differential of the radial coordinate with respect to mass,
    for a point at a given radial position within a sun-like star.

    Args:
        radial_position (float): The radial position of the point relative  to
            the center of some sun-like star [m].
        mass_density (float): The local mass density at the point within the
        star, measured in units of mass per unit volume.

    Returns:
        float: The differential of the radial coordinate with respect to mass,
            measured in units of distance per unit mass.
    """
    return 1 / (4 * np.pi * radial_position**2 * mass_density)


def log_10_opacity(temperature, mass_density):
    """
    Computes opacity as functions of R and
    temperature using data from "opacity.txt".

    Args:
        temperature (float) Temperature in Kelvin.
        mass_density (float) Mass density in kg/m^3.

    Returns:
        opacity (float) Opacity [m^2/kg] of sun-like star, as a function
        of temperature and pressure, per interpolation of data
        from 'opacity.txt'.
    """
    data = np.genfromtxt("opacity.txt")
    LOG_10_R = data[0][1:]
    LOG_10_TEMPERATURES = data[1:, 0]
    LOG_10_OPACITIES = data[1:, 1:]
    instance = RectBivariateSpline(LOG_10_TEMPERATURES, LOG_10_R, LOG_10_OPACITIES)

    LOG_10_TEMPERATURE = np.log10(temperature)
    MASS_DENSITY_CGS = mass_density/1000
    R = MASS_DENSITY_CGS/(temperature/1e6)**3
    LOG_10_R = np.log10(R)
    LOG_10_OPACITY = instance.ev(LOG_10_TEMPERATURE, LOG_10_R)
    return LOG_10_OPACITY


def opacity(temperature, mass_density,sanity_check=False):
    LOG_10_OPACITY = log_10_opacity(temperature, mass_density)
    OPACITY_CGS = 10**LOG_10_OPACITY
    OPACITY_SI = OPACITY_CGS/10
    if sanity_check:
        return 3.98
    return OPACITY_SI


def check_relative_error(actual, approx, tol):
    """
    Checks if the relative error between the actual and approximate values is within the tolerance.

    Args:
        actual: The actual value.
        approx: The approximate value.
        tol: The tolerance value.

    Returns:
        None. Raises an AssertionError if the relative error is outside the tolerance.
    """
    error = abs(actual - approx)
    rel_error = error / abs(actual)
    assert rel_error < tol, f"{rel_error=} is larger than tolerance {tol}"


def sanity_check_opacity():
    """
    Verifies that opacity() works as intended.

    Returns:
        None.
    """
    LOG_10_TEMPERATURES = np.asarray([3.750, 3.755, 3.755, 3.755, 3.755,
                                      3.770, 3.780, 3.795, 3.770, 3.775,
                                      3.780, 3.795, 3.800])
    TEMPERATURES = 10**(LOG_10_TEMPERATURES) #K
    LOG_10_R = np.asfarray([-6.00, -5.95, -5.80, -5.70,
                                                 -5.55, -5.95, -5.95, -5.95,
                                                 -5.80, -5.75, -5.70, -5.55,
                                                 -5.50])
    R = 10**LOG_10_R
    MASS_DENSITIES_CGS = R*(TEMPERATURES/1e6)**3
    MASS_DENSITIES_SI = MASS_DENSITIES_CGS*1e3
    LOG_10_OPACITIES = log_10_opacity(TEMPERATURES, MASS_DENSITIES_SI)
    OPACITIES = opacity(TEMPERATURES, MASS_DENSITIES_SI)
    EXPECTED_OPACITIES = [2.84e-3, 3.11e-3, 2.68e-3, 2.46e-3, 2.12e-3,
                          4.70e-3, 6.25e-3, 9.45e-3, 4.05e-3, 4.43e-3,
                          4.94e-3, 6.89e-3, 7.69e-3]
    table = np.stack([LOG_10_TEMPERATURES, LOG_10_R, LOG_10_OPACITIES, OPACITIES], axis=-1)
    headers = ["log10 T", "log10 R", "log 10 opacity", "opacity"]
    print(tabulate(table, headers=headers))
    for index, opacity_ in enumerate(OPACITIES):
        check_relative_error(EXPECTED_OPACITIES[index], opacity_, tol=6e-2)


def pressure_to_mass_density(temperature, pressure):
    """
    Compute the mass density given some temperature and pressure.
    Args:
        temperature (float): The temperature [K].
        pressure (float): The pressure in units of Pa. [N/m^2]

    Returns:
        (float) mass density in units of kg/m^3
    """
    return (
        ATOMIC_MASS_UNIT*MEAN_MOLECULAR_MASS/BOLTZMANN_CONSTANT/temperature
        * pressure
    )


def mass_density_to_pressure(temperature, mass_density):
    """
    Compute the pressure given some temperature and mass density.

    Args:
        temperature (float): The temperature [K]
        mass_density (float): The mass density [kg/m^3]
    returns:
        (float) pressure [?]
    """
    return (
        BOLTZMANN_CONSTANT*temperature/MEAN_MOLECULAR_MASS/ATOMIC_MASS_UNIT
        * mass_density
    )


def pressure_differential(mass, radial_position):
    """
    Computes the differential of pressure with respect to mass.

    Args:
        mass (float): Mass inside volume V.
        radial_position (float): The radial position relative to center of a
            sun-like star. [m]

    Returns:
        (float): Differential of pressure with respect to mass.
    """
    return -GRAVITATIONAL_CONSTANT * mass / 4 / np.pi / radial_position**4


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
    return ENERGY_PRODUCTION_INSTANCE.true_total_energy_production_rate()


def gravitational_acceleration(mass, radial_position):
    """"
    Computes gravitational acceleration

    Returns:
        (float): Gravitational acceleration [m/s^2]
    """
    return mass * GRAVITATIONAL_CONSTANT / radial_position**2


def pressure_scale_height(mass, radial_position, temperature):
    """
    Computes pressure scale height.
    Args:

    Returns:
        (float): Pressure scale height. 
    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    return (
        BOLTZMANN_CONSTANT*temperature/ATOMIC_MASS_UNIT /
        MEAN_MOLECULAR_MASS/GRAVITATIONAL_ACCELERATION
    )


def stable_temperature_gradient(mass, radial_position, mass_density,
                                temperature, luminosity, sanity_check=False):
    """
    Compute the temperature gradient nescesary for all the energy to be
    carried by radiation.

    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): distance from center of sun-like star.
        mass_density (float): local mass density inside sun-like star.
        temperature (float): local temperature inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): The temperature gradient nescesary for all the energy to be
    carried by radiation.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)    
    return(
        3
        * luminosity
        * OPACITY
        * mass_density
        * PRESSURE_SCALE_HEIGHT
        / 64
        / np.pi
        / radial_position**2
        / STEFAN_BOLTZMANN_CONSTANT
        / temperature**4
    )


def u(mass, radial_position, mass_density, temperature, sanity_check=False):
    """
    Compute unnamed but useful quantity U.
    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): distance from center of sun-like star.
        mass_density (float): local mass density inside sun-like star.
        temperature (float): local temperature inside sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    """
    OPACITY = opacity(temperature, mass_density, sanity_check)
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    return (
        64
        * STEFAN_BOLTZMANN_CONSTANT
        * temperature**3
        * np.sqrt(PRESSURE_SCALE_HEIGHT / GRAVITATIONAL_ACCELERATION / DELTA)
        / 3
        / OPACITY
        / mass_density**2
        / SPECIFIC_HEAT_CAPACITY
    )


def xi(mass, radial_position, mass_density, temperature, luminosity, sanity_check=False):
    """
    computes xi.

    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): distance from center of sun-like star.
        mass_density (float): local mass density inside sun-like star.
        temperature (float): local temperature inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): xi.
    """
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    GEOMETRIC_FACTOR = 4 / DELTA / PRESSURE_SCALE_HEIGHT
    STABLE_TEMPERATURE_GRADIENT = stable_temperature_gradient(mass, radial_position, mass_density, temperature, luminosity,sanity_check=sanity_check)
    U = u(mass, radial_position, mass_density, temperature, sanity_check=sanity_check)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    coeffs = np.asarray(
        [
            1,
            U / MIXING_LENGTH**2,
            U**2 * GEOMETRIC_FACTOR / MIXING_LENGTH**3,
            U
            / MIXING_LENGTH**2
            *(ADIABATIC_TEMPERATURE_GRADIENT - STABLE_TEMPERATURE_GRADIENT),
        ]
    )
    roots = np.roots(coeffs)
    index = np.where(roots.imag == 0)[0][0]
    XI = roots[index].real
    return XI


def temperature_gradient(mass, radial_position, mass_density, temperature, pressure, luminosity, sanity_check=False):
    """
    Computes the temperature gradient.

    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): Distance from center of sun-like star.
        mass_density (float): Local mass density inside sun-like star.
        temperature (float): Local temperature inside sun-like star.
        pressure (float): Local pressure inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    
    Returns:
        (float): Temperature gradient.
    """
    XI = xi(mass, radial_position, mass_density, temperature, luminosity, sanity_check=sanity_check)
    return XI**2 + ADIABATIC_TEMPERATURE_GRADIENT

def temperature_differential(mass, radial_position, mass_density, temperature,
                             pressure, luminosity, dpdm, sanity_check=False):
    """
    computes the differential of temperature with respect to mass.

    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): Distance from center of sun-like star.
        mass_density (float): Local mass density inside sun-like star.
        temperature (float): Local temperature inside sun-like star.
        pressure (float): Local pressure inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        dpdm: (float): the differential of pressure with respect to mass.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): differential of temperature with respect to mass].
    """
    actual_temperature_gradient = temperature_gradient(mass, radial_position, mass_density, temperature, pressure, luminosity, sanity_check=sanity_check)
    return actual_temperature_gradient*temperature/pressure*dpdm


def parcel_velocity(mass, radial_position, mass_density, temperature,
                    luminosity, sanity_check=False):
    """
    Compute the velocity of parcel.
    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): Distance from center of sun-like star.
        mass_density (float): Local mass density inside sun-like star.
        temperature (float): Local temperature inside sun-like star.
        pressure (float): Local pressure inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): parcel velocity [m/s]
    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(mass, radial_position, temperature)
    PARCEL_DISTANCE_MOVED = PRESSURE_SCALE_HEIGHT/2
    XI = xi(mass, radial_position, mass_density, temperature, luminosity,
            sanity_check=sanity_check)
    return (np.sqrt(GRAVITATIONAL_ACCELERATION*DELTA/PRESSURE_SCALE_HEIGHT)
            *XI*PARCEL_DISTANCE_MOVED)


def convective_flux(mass, radial_position, mass_density, temperature, pressure, luminosity, sanity_check=False):
    """
    Computes convective flux.

    Args:
        mass (float): Mass inside volume for sphere with radius equal to
            radial position, centered at center of sun-like star [kg].
        temperature (float): Temperature [K].

    Returns:

    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    XI = xi(mass, radial_position, mass_density,
            temperature, luminosity, sanity_check=sanity_check)
    return (mass_density*SPECIFIC_HEAT_CAPACITY*temperature
            * np.sqrt(GRAVITATIONAL_ACCELERATION*DELTA/PRESSURE_SCALE_HEIGHT)
            * (MIXING_LENGTH/2)**2*XI**3)


def radiative_flux(mass, radial_position, mass_density, temperature, pressure, luminosity, sanity_check=False):
    """
    Computes radiative flux.
    Args:
        mass (float): mass inside volume contained in sphere inside sun-like
            star with radius equal to radial_postion,
        radial_position (float): Distance from center of sun-like star.
        mass_density (float): Local mass density inside sun-like star.
        temperature (float): Local temperature inside sun-like star.
        pressure (float): Local pressure inside sun-like star.
        luminosity (float): luminosity emitted by volume contained inside
            sun-like star with radius equal to radial_position.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): Radiative flux.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check=sanity_check)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(mass, radial_position, temperature)
    TEMPERATURE_GRADIENT = temperature_gradient(mass, radial_position, mass_density, temperature, pressure, luminosity, sanity_check=sanity_check)
    return (16*STEFAN_BOLTZMANN_CONSTANT*temperature**4*TEMPERATURE_GRADIENT/3/OPACITY/mass_density/PRESSURE_SCALE_HEIGHT)


def append_line_to_data_file(data, filename="stellar_parameters.txt"):
    """
    Appends data to new line in specified file
    """
    data = np.reshape(data, (1, len(data)))
    with open(filename, 'a') as f:
        np.savetxt(f, data, delimiter=',', fmt='%f')


def initialize_file(filename="stellar_parameters.txt"):
    """
    Clears the contents of the file specified by filename, if it exists, or
    creates the file if it doesn't exist.

    Args:
        Filename (str): name of file to be initialized.

    Returns:
        None. 
    """
    if os.path.exists(filename):
        with open(filename, 'w') as f:
            f.truncate(0)


def compute_temperature_differential_radiative_only(
    temperature, mass_density, luminosity, radial_position,sanity_check=False
):
    """
    Computes the differential of temperature with respect to mass, with
    radiative only.
    """
    OPACITY = opacity(temperature, mass_density,sanity_check=sanity_check)
    return (
        3
        * OPACITY
        * luminosity
        / 256
        / np.pi**2
        / STEFAN_BOLTZMANN_CONSTANT
        / radial_position**4
        / temperature**3
    )


def sanity_check_gradients():
    """
    Verifies quantity computing functions by comparing with expected values.

    Returns:
        None.
    """
    TEMPERATURE = 0.9e6 #K
    MASS_DENSITY = 55.9 #kg/m^3
    PRESSURE = mass_density_to_pressure(TEMPERATURE, MASS_DENSITY)
    RADIAL_POSITION = 0.84*RADIUS_SUN
    MASS = 0.99*MASS_SUN
    LUMINOSITY = LUMINOSITY_SUN
    STABLE_TEMPERATURE_GRADIENT = stable_temperature_gradient(MASS,
                                                              RADIAL_POSITION,
                                                              MASS_DENSITY,
                                                              TEMPERATURE,
                                                              LUMINOSITY,
                                                              sanity_check=True)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(MASS, RADIAL_POSITION, TEMPERATURE)
    U = u(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, sanity_check=True)
    XI = xi(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, LUMINOSITY,
            sanity_check=True)
    TEMPERATURE_GRADIENT = temperature_gradient(MASS, RADIAL_POSITION,
                                                MASS_DENSITY, TEMPERATURE,
                                                PRESSURE, LUMINOSITY,
                                                sanity_check=True)
    PARCEL_VELOCITY = parcel_velocity(MASS, RADIAL_POSITION, MASS_DENSITY,
                                      TEMPERATURE, LUMINOSITY,
                                      sanity_check=True) 
    RADIATIVE_FLUX = radiative_flux(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, PRESSURE, LUMINOSITY, sanity_check=True)
    CONVECTIVE_FLUX = convective_flux(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, PRESSURE, LUMINOSITY, sanity_check=True)
    FLUX_SUM = RADIATIVE_FLUX + CONVECTIVE_FLUX
    CONVECTIVE_FLUX_RATIO = CONVECTIVE_FLUX/FLUX_SUM
    RADIATIVE_FLUX_RATIO = RADIATIVE_FLUX/FLUX_SUM
    check_relative_error(0.6, MEAN_MOLECULAR_MASS, 1e-1 )
    check_relative_error(32.4e6, PRESSURE_SCALE_HEIGHT, 3e-2)
    check_relative_error(3.26, STABLE_TEMPERATURE_GRADIENT, 4e-2)
    check_relative_error(5.94e5, U, 4e-2)
    check_relative_error(1.173e-3, XI, 2e-2)
    check_relative_error(0.400, TEMPERATURE_GRADIENT, 1e-2)
    check_relative_error(65.50, PARCEL_VELOCITY, 2e-3)
    check_relative_error(0.88, CONVECTIVE_FLUX_RATIO, 1e-2)
    check_relative_error(0.12, RADIATIVE_FLUX_RATIO, 1e-2)

    print("success")



class EnergyTransport:
    """
    Models central parts of a Sun-like star.
    """
    def __init__(self, init_temperature=5770, init_luminosity=LUMINOSITY_SUN,
                 init_mass=MASS_SUN, init_radial_position=RADIUS_SUN,
                 init_mass_density=1.42e-7*AVERAGE_MASS_DENSITY_SUN,
                 filename="stellar_parameters.txt"):
        """
        Initializes instance of EnergyTransport. 
        """
        self.filename = filename
        init_pressure = mass_density_to_pressure(init_temperature,
                                                 init_mass_density)
        self.init_radial_position = init_radial_position
        self.init_mass = init_mass

        self.parameters = np.asarray([init_mass, init_radial_position,
                                      init_luminosity, init_temperature,
                                      init_mass_density, init_pressure])

    def advance(self):
        """
        Evolves stellar parameter quantities 1 step with Euler's method, with
        variable step length.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.


        """
        mass = self.parameters[0]
        radial_position = self.parameters[1]
        luminosity = self.parameters[2]
        temperature = self.parameters[3]
        mass_density = self.parameters[4]
        pressure = self.parameters[5]
        permitted_change = 0.1
        drdm = radial_coordinate_differential(radial_position, mass_density)
        dpdm = pressure_differential(mass, radial_position)
        dldm = total_energy_production_rate(temperature, mass_density)
        dtdm = temperature_differential(temperature, mass_density,
                                        mass, radial_position,
                                        pressure, luminosity, dpdm)
        mass_differentials = np.asarray([
            permitted_change*radial_position/drdm,
            permitted_change*pressure/dpdm,
            permitted_change*luminosity/dldm,
            permitted_change*temperature/dtdm])
        index = np.argmin(np.abs(mass_differentials))
        step_size = np.abs(mass_differentials[index])
        mass = mass - step_size
        radial_position = radial_position - step_size*drdm
        pressure = pressure - step_size*dpdm
        luminosity = luminosity - step_size*dldm
        temperature = temperature - step_size*dtdm
        mass_density = pressure_to_mass_density(temperature, pressure)

        self.parameters = np.asarray([mass, radial_position, luminosity,
                                      temperature, mass_density, pressure])
        for index, param in enumerate(self.parameters):
            msg = f"param[{index}] is causing problems"
            assert isinstance(param, float), msg
            assert not np.isnan(param), msg
            assert not np.isinf(param), msg

    def compute_and_store_to_file(self):
        """
        numerically integrates the stellar parameter quantities with Euler's
        method, and stores values in file with name filename using CSV.
        """
        initialize_file(self.filename)
        append_line_to_data_file(self.parameters, self.filename)
        tolerance = 1e-5
        mass = self.parameters[0]
        while mass > self.init_mass*0.05:
            self.advance()
            append_line_to_data_file(self.parameters, self.filename)
        print("success")

    def read_file(self):
        """
        Reads numerical integral results stored in .txt, stores in
        numpy arrays.
        """
        data = np.loadtxt(self.filename, delimiter=",")
        self.masses = data[:, 0]
        self.radial_positions = data[:, 1]
        self.luminosities = data[:, 2]
        self.temperatures = data[:, 3]
        self.mass_densities = data[:, 4]
        self.pressures = data[:, 5]

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.masses, self.radial_positions, label="radius")
        ax.set_xlabel("mass [kg]")
        ax.set_ylabel("radial position [m]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()

    def plot_cross_section(self):
        convective_flux(self.masses, self.radial_positions, self.mass_densities,
                        self.temperatures, self.pressures, self.luminosities)
        cross_section(self.radial_positions, self.luminosities,
                      self.convective_fluxes)
        
    
sanity_check_opacity()
sanity_check_gradients()
instance = EnergyTransport(filename="stellar_parameters_0dot1.txt")
instance.compute_and_store_to_file()
instance.read_file()
instance.plot_cross_section()
instance.plot()
