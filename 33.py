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
    error_message = f"expected {expected:.4g}, computed {computed:.4g}, error: {error:.4g} "
    assert error < tolerance, error_message


def convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro):
    """
    converts the units of reaction rates from  [reactions*cm^3/s/mole]
    to [reactions*m^3/s]
    """
    avogadros_number = 6.0221e23  # 1/mol
    reaction_rate_cm = reaction_rate_cm_avogadro / avogadros_number #reactions*cm^3/s
    reaction_rate_m = reaction_rate_cm / 1e6 #reactions*m^3/s
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
        "nitrogen_14": 1e-11
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

    def number_density_electron(self):
        """
        Computes the total number density of electrons, assuming each element is fully ionized.
        """ 
        return np.sum([
            self.number_density("hydrogen_1"),
            1 * self.number_density("helium_3"),
            2 * self.number_density("helium_4"),
            3 * self.number_density("lithium_7"),
            4 * self.number_density("beryllium_7"),
            7 * self.number_density("nitrogen_14"),
        ])

    def scale_factor_helium_3(self):
        """
        computes the scaling factor which normalizes the consumption of
        helium 3, such that it does not exceed the production of helium 3,
        and returns 1 if the consumption does not exceed production.
        """
        production_helium_3 = self.reaction_rate_pp()
        consumption_helium_3 = np.sum([
            2 * self.reaction_rate_33(apply_scale_factor=False),
            self.reaction_rate_34(apply_scale_factor=False)
        ])
        if production_helium_3 < consumption_helium_3:
            return production_helium_3 / consumption_helium_3
        return 1

    def reaction_rate_pp(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the
        the fusion of hydrogen 1 nuclei.
        this is the first step of the PP 0 branch.
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

    def reaction_rate_33(self,apply_scale_factor=True):
        """ 
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        helium 3 nuclei, forming helium 4 and hydrogen 1.
        Allows for computing the reaction rate unrestricted by the rate at
        which helium 3 is produced, for the sake of computing the factor which
        is required for adjusting the consumption of helium 3 to the rate at
        which it is produced. 
        This is the first and only step within the PP1 branch.
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
        if apply_scale_factor:
            reaction_rate_m *= self.scale_factor_helium_3()
        return reaction_rate_m

    def reaction_rate_34(self,apply_scale_factor=True):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        Helium 3 and Helium 4, forming beryllium 7.
        Allows for computing the reaction rate unrestricted by the rate at
        which helium 3 is produced, for the sake of computing the factor which
        is required for adjusting the consumption of helium 3 to the rate at
        which it is produced. 
        This is the first step within the PP2 and PP3 branches.
        """
        temperature9_ = self.temperature9 / (1 + 4.95e-2 * self.temperature9)
        reaction_rate_cm_avogadro = (
            5.61e6
            * temperature9_ ** (5 / 6)
            * self.temperature9 ** (-3 / 2)
            * np.exp(-12.826 * temperature9_ ** (-1 / 3))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        if apply_scale_factor:
            reaction_rate_m *= self.scale_factor_helium_3()
        return reaction_rate_m

    def reaction_rate_per_unit_mass_33(self):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of helium 3 nuclei, forming helium 4 and hydrogen 1.
        This is the first and only step within the pp 1 branch.
        """
        return (
            self.reaction_rate_33(apply_scale_factor=False)
            * self.number_density("helium_3") ** 2
            / self.mass_density
            / 2
        )

    def energy_production_rate_33(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of helium 3 nuclei, forming helium 4 and hydrogen 1,
        adjusting the consumption of helium 3 according to the rate at
        which it is produced.
        This is the first and only step within the PP 1 branch.
        """
        released_energy_33 = 12.860 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        return (
            self.reaction_rate_per_unit_mass_33()
            * released_energy_33
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
        evaluate(8.68e-9, self.energy_production_rate_33(),tolerance=1e-22)


    def debug(self):
        """
        used for debugging, mainly prints values.
        """
        expected_energy_production_rate_33 = 8.68e-9
        number_density_helium_3 = self.number_density("helium_3")
        released_energy_33 = 12.860 * self.ELECTRON_TO_JOULE_CONVERSION_FACTOR
        expected_reaction_rate_per_unit_mass_33 = expected_energy_production_rate_33/released_energy_33/self.mass_density
        expected_reaction_rate_33 = expected_reaction_rate_per_unit_mass_33*2*self.mass_density/number_density_helium_3/number_density_helium_3
        print(f"scale factor helium 3: {self.scale_factor_helium_3()}")
        print("\n\n\n\n\n\n")
        print("various values computed for computing the energy production rate for helium 3 fusion:")
        print(f"energy production rates, expected: {expected_energy_production_rate_33:.4g}, computed: {self.energy_production_rate_33():.4g}")
        print(f"reaction rates per unit mass, expected: {expected_reaction_rate_per_unit_mass_33:.4g}, computed: {self.reaction_rate_per_unit_mass_33()}")
        print(f"reaction rates, expected: {expected_reaction_rate_33}, computed: {self.reaction_rate_33()}")

instance = EnergyProduction()
instance.debug()
instance.sanity_check()
# mass for each atomic species are provided in kg


# Mass fractions for each atomic species are assumed independent of radius, and are unitless.
