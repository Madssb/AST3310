"""
Script simulates the Energy production at the center of a star.
"""
import numpy as np


def evaluate(expected, computed, tolerance=1e-5):
    """
    verifies if known computed quantities match their expected values,
    used in sanity checks.
    """
    error = np.abs(expected - computed)
    error_message = (
        f"expected {expected:.4g}, computed {computed:.4g}, error: {error:.4g} "
    )
    assert error < tolerance, error_message


def convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro):
    """
    converts the units of reaction rates from  [reactions*cm^3/s/mole]
    to [reactions*m^3/s]
    """
    avogadros_number = 6.0221e23  # 1/mol
    reaction_rate_cm = reaction_rate_cm_avogadro / avogadros_number  # reactions*cm^3/s
    reaction_rate_m = reaction_rate_cm / 1e6  # reactions*m^3/s
    return reaction_rate_m

class EnergyProduction:
    """
    Calculates the energy production at the center of a star, given a
    temperature and mass density.
    """

    # The energy per PP-chain is constant, and therefore we define them as such,
    # in units of Joule. they are products of the conversion factor, aswell as
    # their output in MeV, sourced from Table 3.2. in lecture notes.
    # here p and numbers denote which process is the origin of the released
    # energy.
    ELECTRON_TO_JOULE_CONVERSION_FACTOR = 1.6022e-19 * 1e6  # Joule/MeV
    ATOMIC_MASS_UNIT = 1.6605e-27  # kg
    MASS_FRACTIONS = {
        "hydrogen_1": 0.7,
        "helium_3": 1e-10,
        "helium_4": 0.29,
        "lithium_7": 1e-7,
        "beryllium_7": 1e-7,
        "nitrogen_14": 1e-11,
    }

    def __init__(self, mass_density=1.62e5, temperature=1.57e7):
        """
        The mass density [kg/m^3] and temperature [K] are by default specified
        to the values of that of the solar core, and may be overwritten.
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

    def reaction_rate_p14(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        nitrogen 14 and hydrogen 1, forming oxygen 15.
        This is thethe fourth step of the CNO-cycle.
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

    def reaction_rate_per_unit_mass_p14(self):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of hydrogen 1 and nitrogen 14, forming oxygen 15.
        this is the fourth, and only step within the CNO-cycle that is not
        nearly instantaneous.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_p14()
            * self.number_density("hydrogen_1")
            * self.number_density("nitrogen_14")
            / self.mass_density
        )
        return reaction_rate_per_unit_mass

    def energy_production_rate_p14(self):
        """
        computes the energy production rate per unit volume [J/m^3/s] for the
        entire CNO cycle, gated by the reaction rate per unit mass for the
        fusion of nitrogen 14 and hydrogen 1, as other fusions in the CNO-
        cycle are demmed near instantaneous.
        """
        released_energy_p12 = 1.944 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_13 = 1.513 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p13 = 7.551 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p14 = 7.297 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_15 = 1.757 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p15 = 4.966 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_cno = sum(
            [
                released_energy_p12,
                released_energy_13, 
                released_energy_p13,
                released_energy_p14,
                released_energy_15,
                released_energy_p15,
            ]
        )
        return (
            self.reaction_rate_per_unit_mass_p14()
            * released_energy_cno
            * self.mass_density
        )

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
        evaluate(9.18e-8, self.energy_production_rate_p14(), tolerance=1e-10)

    def debug(self):
        """
        used for debugging, mainly prints values.
        """
        epsilon = 9.18e-8
        released_energy_p12 = 1.944 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_13 = 1.513 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p13 = 7.551 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p14 = 7.297 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_15 = 1.757 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        released_energy_p15 = 4.966 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        Q = sum(
            [
                released_energy_p12,
                released_energy_13, 
                released_energy_p13,
                released_energy_p14,
                released_energy_15,
                released_energy_p15,
            ]
        )
        expected_reaction_rate_per_unit_mass = epsilon / Q / self.mass_density
        expected_reaction_rate = (
            expected_reaction_rate_per_unit_mass
            * self.mass_density
            / self.number_density("hydrogen_1")
            / self.number_density("nitrogen_14")
        )
        print(
            f"""
            Reaction rate per unit mass. Computed: {self.reaction_rate_per_unit_mass_p14():.4g},  expected: {expected_reaction_rate_per_unit_mass:.4g}
            reaction rate. Computed: {self.reaction_rate_p14():.4g}, expected: {expected_reaction_rate:.4g}
            production rate. Computed: {self.energy_production_rate_p14():.4g}, expected: {epsilon:.4g}
            mass density: {self.mass_density:.4g}
            number density hydrogen 1: {self.number_density('hydrogen_1'):.4g}
            number density nitrogen 14: {self.number_density('nitrogen_14'):.4g}
                """
        )


instance = EnergyProduction()
instance.debug()
instance.sanity_check()