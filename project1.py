"""
Script simulates the Energy production at the center of a star.
"""
import numpy as np


def evaluate(expected, computed, tolerance=1e-5):
    """
    verifies if known computed quantities match their expected values,
    used in sanity checks.
    """
    error_message = f"expected {expected}, computed {computed}."
    assert np.abs(expected - computed) < tolerance, error_message

def convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro):
    """
    converts the units of reaction rates from  [reactions*cm^3/s/mole]
    to [reactions*m^3/s]
    """
    avogadros_number = 6.0221e23  # 1/mol
    reaction_rate_cm = reaction_rate_cm_avogadro / avogadros_number
    reaction_rate_m = reaction_rate_cm / 1e6
    return reaction_rate_m

def idk(a,b):
    if a<b:
        return 

class EnergyProduction:
    """
    Calculates the energy production at the center of a star, given a
    temperature and mass density.
    """

    # The energy per PP-chain is constant, and therefore we define them as such,
    # in units of Joule. they are products of the conversion factor, aswell as
    # their output in MeV, sourced from Table 3.2. in lecture notes.
    ELECTRON_TO_JOULE_CONVERSION_FACTOR = 1.6022e-19 * 1e6  # Joule/MeV
    RELEASED_ENERGY_PP0_1ST = 1.177 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP0_2ND = 5.494 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP1 = 12.869 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP2_PP3_1ST = 1.586 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP2_2ND = 0.049 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP2_3RD = 17.346 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP3_2ND = 0.137 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PP3_3RD = 8.367 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_1ST = 1.944 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_2ND = 1.513 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_3RD = 7.551 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_4TH = 7.297 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_5TH = 1.757 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_CNO_6TH = 4.966 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    ATOMIC_MASS_UNIT = 1.6605e-27  # kg

    MASS_FRACTIONS = {
        "hydrogen_1": 0.7,
        "helium_3": 1e-10,
        "helium_4": 0.29,
        "lithium_7": 10e-7,
        "beryllium_7": 10e-7,
        "nitrogen_14": 10e-11,
    }

    def __init__(self, mass_density=1.62e5, temperature=1.57e7):
        """
        The mass density [kg/m^3] and temperature of the solar core [K] are specified by
        default, but may be overwritten.
        """
        self.mass_density = mass_density  # kg/m^3
        self.temperature = temperature  # K
        self.temperature9 = temperature * 1e-9  # 10^9K

    def number_density(self, atom_isotope):
        """
        atom_isotope must take the form "[atom]_[nucleon count]" e.g.
        "hydrogen_1", and is limited to those found as keys in the key
        pairs that live in the mass_fraction dictionary.
        """
        nucleus_mass = int(atom_isotope[-1]) * self.ATOMIC_MASS_UNIT
        return self.mass_density * self.MASS_FRACTIONS[atom_isotope] / nucleus_mass

    def number_density_electron(self):
        """
        computes the total number density of electrons, assuming each element as fully ionized.
        """
        return (
            self.number_density("hydrogen_1")
            + 2 * self.number_density("helium_3")
            + 2 * self.number_density("helium_4")
            + 3 * self.number_density("lithium_4")
            + 4 * self.number_density("beryllium_7")
            + self.number_density("nitrogen_14")
        )

    def scale_factor_helium_3(self):
        """
        scales consumption of helium 3 according to the rate of production.
        if consumption exceeds production, has no effect otherwise.
        """
        production_helium_3 = self.reaction_rate_per_unit_mass_pp()
        consumption_helium_3 = 2*self.reaction_rate_per_unit_mass_33() + self.reaction_rate_per_unit_mass_34()
        if production_helium_3 < consumption_helium_3:
            return production_helium_3 / consumption_helium_3
        return 1

    def reaction_rate_pp(self):
        """
        computes reaction rate  for the fusion of two protons, i.e. the first
        step in PP0. [reactions*m^3/s]
        """
        reaction_rate_cm_avogadro = (
            4.01e-15
            * self.temperature9 ** (-2 / 3)
            * np.exp(-3.380 * self.temperature9 ** (-1 / 3))
            * (
                1
                + 0.123 * self.temperature9 ** (1 / 3)
                + 1.09 * self.temperature9 ** (2 / 3)
                + 0.938 * self.temperature9
            )
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_33(self):
        """
        computes reaction rate  for the fusion of deuterium and hydrogen,
        i.e. the first step in the PP1 branch. [reactions/s/m^3]
        """
        reaction_rate_cm_avogadro = (
            6.04e10
            * self.temperature9 ** (-2 / 3)
            * np.exp(-12.276 * self.temperature9 ** (-1 / 3))
            * (
                1
                + 0.034 * self.temperature9 ** (1 / 3)
                - 0.522 * self.temperature9 ** (2 / 3)
                - 0.124 * self.temperature9
                + 0.353 * self.temperature9 ** (4 / 3)
                + 0.213 * self.temperature9 ** (5 / 3)
            )
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_34(self):
        """
        computes reaction rate  for the fusion of Helium 3 and Helium 4,
        i.e. first step of the PP2  and PP3 branches. [reactions/s/m^3]
        """
        temperature9_ = self.temperature9 / (1 + 4.95e-2 * self.temperature9)
        reaction_rate_cm_avogadro = (
            5.61e6
            * temperature9_ ** (5 / 6)
            * self.temperature9 ** (-3 / 2)
            * np.exp(-12.826 * temperature9_ ** (-1 / 3))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_e7(self):
        """
        computes the reaction rate  for the decay of Berylium 7, i.e. the
        second step of the PP2 branch. [reactions/s/m^3]
        """
        reaction_rate_cm_avogadro = (
            1.34e-10
            * self.temperature9 ** (-1 / 2)
            * (
                1
                - 0.537 * self.temperature9 ** (1 / 3)
                + 3.86 * self.temperature9 ** (2 / 3)
                + 0.0027
                * self.temperature9 ** (-1)
                * np.exp(2.515e-3 * self.temperature9 ** (-1))
            )
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_17_(self):
        """
        computes the reaction rate for the fusion of Lithium and hydrogen,
        i.e. the third step of the PP2 branch. [reactions/s/m^3]
        """
        temperature9__ = self.temperature9 / (1 + 0.759 * self.temperature9)
        reaction_rate_cm_avogadro = (
            1.096e9
            * self.temperature9 ** (-2 / 3)
            * np.exp(-8.472 * self.temperature9 ** (-1 / 3))
            - 4.830e-8
            * temperature9__ ** (5 / 6)
            * self.temperature9 ** (-3 / 2)
            * np.exp(-8.472 * temperature9__ ** (-1 / 3))
            + 1.06e10
            * self.temperature9 ** (-3 / 2)
            * np.exp(-30.442 * self.temperature9 ** (-1))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_17(self):
        """
        computes the reaction rate for the fusion of Beryllium and hydrogen,
        i.e. the second step of the PP3 branch. [reactions/s/m^3]
        """
        reaction_rate_cm_avogadro = 3.11e5 * self.temperature9 ** (-2 / 3) * np.exp(
            -10.262 * self.temperature9
        ) + 2.53e3 * self.temperature9 ** (-3 / 2) * np.exp(
            -7.306 * self.temperature9 ** (-1)
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_p14(self):
        """
        computes the reaction rate for the fusion of nitrogen and hydrogen,
        i.e. the fourth and only significant of the CNO cycle, in regards
        to reaction rates. [reactions/s/m^3]
        """
        reaction_rate_cm_avogadro = (
            4.90e7
            * self.temperature9 ** (-2 / 3)
            * np.exp(
                -15.228 * self.temperature9 ** (-1 / 3) - 0.092 * self.temperature9**2
            )
            * (
                1
                + 0.027 * self.temperature9 ** (1 / 3)
                - 0.778 * self.temperature9 ** (2 / 3)
                - 0.149 * self.temperature9
                + 0.261 * self.temperature9 ** (4 / 3)
                + 0.127 * self.temperature9 ** (5 / 3)
            )
            + 2.37e3
            * self.temperature9 ** (-3 / 2)
            * np.exp(-3.011 * self.temperature9 ** (-1))
            + 2.19e4 * np.exp(-12.53 * self.temperature9 ** (-1))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_per_unit_mass_pp(self):
        """
        computes reaction rate per unit mass for the fusion of two protons,
        i.e. the first step in PP0. [reactions/s/kg]
        """
        return (
            self.reaction_rate_pp()
            * self.number_density("hydrogen_1") ** 2
            / self.mass_density
            / 2
        )

    def reaction_rate_per_unit_mass_33(self):
        """
        computes reaction rate per unit mass for the fusion of deuterium and
        hydrogen, i.e. the first step in the PP1 branch. [reactions/s/kg]
        """
        return (
            self.reaction_rate_33()
            * self.number_density("helium_3") ** 2
            / self.mass_density
            / 2
        )

    def reaction_rate_per_unit_mass_34(self):
        """
        computes reaction rate per unit mass for the fusion of Helium 3 and
        Helium 4, i.e. first step of the PP2  and PP3 branches.
        [reactions/s/kg]
        """
        return (
            self.reaction_rate_34()
            * self.number_density("helium_3")
            * self.number_density("helium_4")
            / self.mass_density
        )

    def reaction_rate_per_unit_mass_e7(self):
        """
        computes the reaction rate per unit mass for the decay of Beryllium 7,
        i.e.
        """
        return (
            self.reaction_rate_e7()
            * self.number_density("beryllium_7")
            * self.number_density_electron()
            / self.mass_density
        )

    def reaction_rate_per_unit_mass_17_(self):
        """
        computes the reaction rate per unit mass for the fusion of Lithium and
        hydrogen 1, i.e. the third step of the PP2 branch. [reactions/s/m^3]
        """
        return (
            self.reaction_rate_17_()
            * self.number_density("hydrogen_1")
            * self.number_density("lithium_7")
            / self.mass_density()
        )

    def reaction_rate_per_unit_mass_17(self):
        """
        computes the reaction rate per unit mass for the fusion of hydrogen 1
        and Beryllium 7, i.e. the second step of the PP3 branch.
        [reactions/s/kg]
        """
        return (
            self.reaction_rate_17()
            * self.number_density("hydrogen_1")
            * self.number_density("lithium_7")
            / self.mass_density()
        )

    def reaction_rate_per_unit_mass_p14(self):
        """
        computes the reaction rate per unit mass for the fusion of hydrogen 1
        and nitrogen 14, i.e. the fourth, and only significant
        step in the CNO-cycle with regards to reaction rates. [reactions/s/kg]
        """
        return (
            self.reaction_rate_p14()
            * self.number_density("hydrogen_1")
            * self.number_density("nitrogen_14")
            / self.mass_density()
        )

    def energy_rate_pp0(self):
        """
        computes energy production rate per unit volume per second [J/m^3/s]
        for the fusion of two protons, aswell as the fusion of deuterium and
        a proton, i.e. the first and second step of the PP 0 branch.
        """
        return (
            self.reaction_rate_per_unit_mass_pp()
            * (self.RELEASED_ENERGY_PP0_1ST + self.RELEASED_ENERGY_PP0_2ND)
            * self.mass_density
        )

    def energy_rate_pp1(self):
        """
        Computes energy production rate per unit volume per second [J/m^3/s]
        for the PP 1 branch, i.e. the fusion of two helium-3 isotopes,
        accounting for whether or not PP 0 can sustain the hydrogen 1
        required, and limiting the reaction rate per unit mass accordingly
        if not.
        """
        return (
            self.reaction_rate_per_unit_mass_33()
            * self.RELEASED_ENERGY_PP1 
            * self.mass_density
            * self.scale_factor_helium_3()
        )
    def energy_rate_pp2_pp3_1st(self):
        """
        computes energy production rate per unit volume per second [j/m^3/s]
        for the fusion of helium 3 and helium 4, i.e. the first step of both
        the PP 2 and PP 3 branches. 
        """ 
        return (
            self.reaction_rate_per_unit_mass_34()
            * self.RELEASED_ENERGY_PP2_PP3_1ST 
            * self.mass_density
            * self.scale_factor_helium_3()
        )

    
    def full_energy_generation_per_unit_mass(self):
        pass

    def sanity_check(self):
        """
        Verifies the that the various methods work as intended, and calculate the correct values.
        """
        assert (
            self.temperature == 1.57e7
        ), f"expected T = {1.57e7}K, actual T = {self.temperature}K."
        assert (
            self.mass_density == 1.62e5
        ), f"expected rho =  {1.62e5}kg/m^3, actual rho =  {self.mass_density}kg/m^3."
        evaluate(4.04e2, self.energy_rate_pp0(),tolerance=1)
        evaluate(8.68e-9, self.energy_rate_pp1())
        evaluate(4.86e-5, self.energy_rate_pp2_pp3_1st())

    def debug(self):
        print("DEBUG TIME!")
        print(f"temperature9: {self.temperature9}*10^9 K")
        print(f"reaction rate: {self.reaction_rate_pp()} m^3/s")
        print(f'number density: {self.number_density("hydrogen_1")} 1/m^3')


instance = EnergyProduction()
# instance.debug()
instance.sanity_check()
# mass for each atomic species are provided in kg


# Mass fractions for each atomic species are assumed independent of radius, and are unitless.
