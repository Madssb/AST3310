"""
Script simulates the energy production at the center of a star, in the 
processes that generate helium 4. For now, energy production rates associated
with excess helium 3 are not accounted for. 
"""
import numpy as np
import matplotlib.pyplot as plt
# all masses are in atomic units [u], and sourced from https://ciaaw.org/
# unless otherwise specified
MASS = {  # u
    "hydrogen_1": 1.0078250322,
    "deuterium_2": 2.0141017781,
    "helium_3": 3.016029322,
    "helium_4": 4.0026032545,
    "lithium_7": 7.01600344,
    "beryllium_7": 7.0169287,  # Pubchem 2.1
    "nitrogen_14": 14.003074004,
    "carbon_12": 12,
    "carbon_13": 13.003354835,
    "nitrogen_15": 15.000108899,
    "electron": 5.486e-4  # nist
}


def converted_mass_to_energy(mass_difference):
    """
    Applies the mass-energy equivalency principle E=mc^2 to convert mass
    defiency [u] into the corresponding energy [MeV], utilizing a conversion factor for
    atomic mass units to MeV.
    """
    u_to_mev_per_c_squared = 931.4941  # MeV/c^2
    return mass_difference * u_to_mev_per_c_squared  # Mev


def energy_pp0():
    """
    Applies the mass-energy equivalency principle to compute the energy freed
    in the PP 0 chain, where three hydrogen 1 nuclei are converted to a
    single helium 3 nucleus. The annihilation of protons and electrons are
    not included.
    """
    input_mass = 3 * MASS["hydrogen_1"]  # u
    output_mass = MASS["helium_3"]  # u
    converted_mass = input_mass - output_mass  # u
    return converted_mass_to_energy(converted_mass)  # MeV


def energy_pp1():
    """
    Applies the mass-energy equivalency principle to compute the energy freed
    in the PP 1 branch, where two helium 3 nuclei fuse, forming helium 4 and
    two hydrogen 1 nuclei. Energy also includes the energy produced by
    2 iterations of PP0 because we require two helium 3 nuclei.
    """
    input_mass = 2 * MASS["helium_3"]  # u
    output_mass = MASS["helium_4"] + 2 * MASS["hydrogen_1"]  # u
    converted_mass = input_mass - output_mass  # u
    return converted_mass_to_energy(converted_mass) + 2 * energy_pp0()  # MeV


def energy_pp2_pp3():
    """
    Applies the mass-energy equivalency principle to compute the energy freed
    in the PP 2 and 3 branches, where through three and four steps
    respectively, helium 3, helium 4 and hydrogen 1 form two helium 4 nuclei.
    Energy produced by PP0 is also included due to helium 3 nucleon required.
    """
    input_mass = MASS["helium_3"] + MASS["helium_4"] + MASS["hydrogen_1"]  # u
    output_mass = 2 * MASS["helium_4"]  # u
    converted_mass = input_mass - output_mass  # u
    return converted_mass_to_energy(converted_mass) + energy_pp0()  # MeV


def energy_cno():
    """
    Applies the mass-energy equivalency principle to compute the energy freed in
    the CNO-cycle, where through 6 steps, 4 hydrogen 1 nuclei form a helium 4
    nucleon.
    """
    input_mass = 4*MASS["hydrogen_1"]
    output_mass = MASS["helium_4"]
    converted_mass = input_mass - output_mass
    return converted_mass_to_energy(converted_mass)


def energies():
    """
    Compute the energy gained from mass conversion per iterations for the PP
    chains aswell as for the CNO-cycle. Compares this energy to the energy
    which is lost by neutrinos escaping. Also prints all of this
    """
    neutrino_energy_lost_pp0 = 0.265  # MeV
    neutrino_energy_lost_pp1 = 2 * neutrino_energy_lost_pp0
    neutrino_energy_lost_pp2 = 0.815 + neutrino_energy_lost_pp0  # MeV
    neutrino_energy_lost_pp3 = 6.711 + neutrino_energy_lost_pp0  # MeV
    neutrino_energy_lost_cno = 0.707 + 0.997
    print(
        f"""
PP0:
mass converted: {(3*MASS['hydrogen_1'] - MASS['helium_3']):.4g}u
energy output: {energy_pp0():.4g}MeV

PP1:
Energy output: {energy_pp1():.4g}MeV,
lost to neutrino: {neutrino_energy_lost_pp1:.4g}MeV,
percentage of energy lost: {neutrino_energy_lost_pp0/energy_pp0()*1e2:.4g}%.

PP2
Energy output: {energy_pp2_pp3():.4g}MeV,
lost to neutrino: {neutrino_energy_lost_pp2:.4g}MeV,
percentage of energy lost: {neutrino_energy_lost_pp2/energy_pp2_pp3()*1e2:.4g}%.

PP3:
Energy output: {energy_pp2_pp3():.4g}MeV,
lost to neutrino: {neutrino_energy_lost_pp3:.4g}MeV,
percentage of energy lost: {neutrino_energy_lost_pp3/energy_pp2_pp3()*1e2:.4g}%.

CNO:
Energy output: {energy_cno():.4g}MeV,
lost to neutrino: {neutrino_energy_lost_cno:.4g}Mev,
percentage of energy lost: {neutrino_energy_lost_cno/energy_cno()*1e2:.4g}%.
"""
    )


def evaluate(expected, computed, tolerance=1e-5):
    """
    Verifies if computations match their expected value, used exclusively
    by sanity_check() within EnergyProduction.
    """
    error = np.abs(expected - computed)
    error_message = (
        f"expected {expected:.4g}, computed {computed:.4g}, error: {error:.4g}."
    )
    assert error < tolerance, error_message


def convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro):
    """
    Convert the units of reaction rates from  [reactions*cm^3/s/mole]
    to [reactions*m^3/s]. Applied inside reaction_rate_xx() methods within
    EnergyProduction, where xx denotes which process the reaction rate
    corresponds to.
    """
    avogadros_number = 6.0221e23  # 1/mol
    reaction_rate_cm = reaction_rate_cm_avogadro / \
        avogadros_number  # reactions*cm^3/s
    reaction_rate_m = reaction_rate_cm / 1e6  # reactions*m^3/s
    return reaction_rate_m


class ReactionRate:
    """
    Computes the reaction rate [reactions*m^3/s] for the fusion
    processes found in the solar core, as part of the Proton Proton chain, in
    addition to the CNO-cycle, as functions of temperature [K] and mass
    density [kg/m^3].
    """

    MEV_TO_JOULE_CONVERSION_FACTOR = 1.6022e-19 * 1e6  # Joule/MeV

    def __init__(self, mass_density=1.62e5, temperature=1.57e7):
        """
        Initializes a ReactionRate object for a given
        The mass density [kg/m^3] and temperature [K].
        """
        self.mass_density = mass_density  # kg/m^3
        self.temperature = temperature  # K
        self.temperature9 = temperature * 1e-9  # 10^9K

    def reaction_rate_pp(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the
        the fusion of hydrogen 1 nuclei, forming deuterium.

        this is the first step within the PP 0 branch.
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_33(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        helium 3 nuclei, forming helium 4 and hydrogen 1.
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_34(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        Helium 3 and Helium 4, forming beryllium 7.
        The reaction rate is unscaled, meaning that the computation is wrong
        for conditions where the rate of consumption for helium 3 exceeds the
        rate at which it is produced.
        This is the first step within both the PP2 and PP3 branches.
        """
        temperature9_ = self.temperature9 / (1 + 4.95e-2 * self.temperature9)
        reaction_rate_cm_avogadro = (
            5.61e6
            * temperature9_ ** (5 / 6)
            * self.temperature9 ** (-3 / 2)
            * np.exp(-12.826 * temperature9_ ** (-1 / 3))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_e7(self):
        """
        Computes the reaction rate [reactions/s/m^3] for the electron capture
        by beryllium 7, forming lithium 7.
        For temperatures below 1M Kelvin, the reaction rate is restricted to
        a known upper limit.
        The reaction rate is unscaled, meaning that the computation is wrong
        for conditions where the rate of consumption for beryllium 7 exceeds
        the rate at which it is produced.
        This is the the second step within the PP2 branch.
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
        upper_limit_cm_avogadro = 1.57e-7 / self.number_density_electron()
        if self.temperature9 < 1e6*1e-9:
            reaction_rate_m = convert_cm_to_m_reaction_rate(
                upper_limit_cm_avogadro)
            return reaction_rate_m
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_17_(self):
        """
        Computes the reaction rate for [reactions*m^3/s/] the fusion of
        Lithium and hydrogen, forming helium 4.
        The reaction rate is unscaled, meaning that the computation is wrong
        for conditions where the rate of consumption for lithium 7 exceeds
        the rate at which it is produced.
        This is the third and final step within the PP2 branch.
        """
        temperature9__ = self.temperature9 / (1 + 0.759 * self.temperature9)
        reaction_rate_cm_avogadro = (
            1.096e9
            * self.temperature9 ** (-2 / 3)
            * np.exp(-8.472 * self.temperature9 ** (-1 / 3))
            - 4.830e8
            * temperature9__ ** (5 / 6)
            * self.temperature ** (-3 / 2)
            * np.exp(-8.472 * temperature9__ ** (-1 / 3))
            + 1.06e10
            * self.temperature9 ** (-3 / 2)
            * np.exp(-30.442 * self.temperature9 ** (-1))
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_17(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        Beryllium 7 and hydrogen, forming boron 8.
        The reaction rate is unscaled, meaning that the compuatation is wrong
        for conditions where the rate of consumption for beryllium 7 exceeds
        the rate at which it is produced.
        This is the second step of the PP3 branch.
        """
        reaction_rate_cm_avogadro = 3.11e5 * self.temperature9 ** (-2 / 3) * np.exp(
            -10.262 * self.temperature9 ** (-1 / 3)
        ) + 2.53e3 * self.temperature9 ** (-3 / 2) * np.exp(
            -7.306 * self.temperature9 ** (-1)
        )
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m

    def reaction_rate_p14(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        nitrogen 14 and hydrogen 1, forming oxygen 15.
        This is the fourth step of the CNO-cycle.
        """
        reaction_rate_cm_avogadro = (
            4.90e7
            * self.temperature9 ** (-2 / 3)
            * np.exp(
                -15.228 *
                self.temperature9 ** (-1 / 3) - 0.092 * self.temperature9**2
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(
            reaction_rate_cm_avogadro)
        return reaction_rate_m


class ReactionRatePerUnitMass(ReactionRate):
    """
    Computes the reaction rate per unit mass [reactions/kg/s] for the
    different fusion  processes found in the solar core, as part of the Proton
    Proton chain, in addition to the CNO-cycle, as functions of temperature
    [K] and mass density [kg/m^3].
    """

    ATOMIC_MASS_UNIT = 1.6605e-27  # kg
    MASS_FRACTIONS = {
        "hydrogen_1": 0.7,
        "helium_3": 1e-10,
        "helium_4": 0.29,
        "lithium_7": 1e-7,
        "beryllium_7": 1e-7,
        "nitrogen_14": 1e-11,
    }

    def __init__(self, mass_density=162000, temperature=15700000):
        """
        Initializes a ReactionRatePerUnitMass object for a given mass density
        [kg/m^3] and a given temperature [K].
        """
        super().__init__(mass_density, temperature)

    def number_density(self, atom_isotope):
        """
        atom_isotope must take the form "[atom]_[nucleon count]" e.g.
        "hydrogen_1", and is limited to those found as keys in the key
        pairs that live in the mass_fraction dictionary, otherwise throws
        an assertion error.
        """
        error_message = f"{atom_isotope=} not found in {self.MASS_FRACTIONS.keys()=}"
        assert atom_isotope in self.MASS_FRACTIONS.keys(), error_message
        nucleon_count = atom_isotope.split("_")[1]
        nucleon_count = int(nucleon_count)
        nucleus_mass = nucleon_count * self.ATOMIC_MASS_UNIT
        return self.mass_density * self.MASS_FRACTIONS[atom_isotope] / nucleus_mass

    def number_density_electron(self):
        """
        Computes the total number density of electrons.
        Each element is assumed each element fully ionized.
        """
        return np.sum(
            [
                self.number_density("hydrogen_1"),
                1 * self.number_density("helium_3"),
                2 * self.number_density("helium_4"),
                3 * self.number_density("lithium_7"),
                4 * self.number_density("beryllium_7"),
                7 * self.number_density("nitrogen_14"),
            ]
        )

    def scale_factor_helium_3(self):
        """
        Computes the factor which scales the reaction rate per unit mass for
        helium 3 consuming reactions, i.e. the fusion of helium 3 nuclei,
        aswell as the fusion of helium 3 and helium 4, such that their
        consumption doesn't exceed the rate at which helium 3 is produced by
        the fusion of hydrogen 1 nuclei, aswell as the fusion of hydrogen 1
        and deuterium.
        If consumption exceeds production, returns production/consumption.
        If consumption does not exceed production, returns 1.
        """
        production_helium_3 = self.reaction_rate_per_unit_mass_pp()
        consumption_helium_3 = np.sum(
            [
                2 *
                self.reaction_rate_per_unit_mass_33(apply_scale_factor=False),
                self.reaction_rate_per_unit_mass_34(apply_scale_factor=False),
            ]
        )
        if production_helium_3 < consumption_helium_3:
            return production_helium_3 / consumption_helium_3
        return 1

    def scale_factor_beryllium_7(self):
        """
        Computes the factor which scales the reaction rate per unit mass for
        beryllium 7 consuming reactions, i.e. the electron capture by
        beryllium 7, and the fusion of beryllium 7 and hydrogen 1, such that
        their consumption doesn't exceed the rate at which beryllium 7 is
        produced by the fusion of helium 3 and helium 4.
        If consumption exceeds production, returns production/consumption.
        If consumption does not exceed production, returns 1.
        """
        production_beryllium_7 = self.reaction_rate_per_unit_mass_34()
        consumption_beryllium_7 = np.sum(
            [
                self.reaction_rate_per_unit_mass_e7(apply_scale_factor=False),
                self.reaction_rate_per_unit_mass_17(apply_scale_factor=False),
            ]
        )
        if production_beryllium_7 < consumption_beryllium_7:
            return production_beryllium_7 / consumption_beryllium_7
        return 1

    def scale_factor_lithium_7(self):
        """
        Computes the factor which scales the reaction rate per unit mass for
        the lithium 7 consuming reaction, i.e. the fusion of lithium 7 and
        hydrogen 1, such that it's consumption does not exceed the rate at
        which lithium 7 is produced by the electron capture by beryllium 7.
        If consumption exceeds production, returns production/consumption.
        If consumption does not exceed production, returns 1.
        """
        production_lithium_7 = self.reaction_rate_per_unit_mass_e7(
            apply_scale_factor=True
        )
        consumption_lithium_7 = self.reaction_rate_per_unit_mass_17_(
            apply_scale_factor=False
        )
        if production_lithium_7 < consumption_lithium_7:
            return production_lithium_7 / consumption_lithium_7
        return 1

    def reaction_rate_per_unit_mass_pp(self):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of hydrogen 1 nuclei, forming deuterium. 
        this is the first step of the PP chain.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_pp()
            * self.number_density("hydrogen_1") ** 2
            / self.mass_density
            / 2
        )
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_33(self, apply_scale_factor=True):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of helium 3 nuclei, forming helium 4 and hydrogen 1.
        By default, the reaction rate per unit mass is scaled such that
        the rate of consumption for helium 3 does not exceed the rate at
        which it is produced, but the unscaled rate may also be computed,
        such that the scaling factor itself can be computed too.
        This is the first and only step within the pp 1 branch.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_33()
            * self.number_density("helium_3") ** 2
            / self.mass_density
            / 2
        )
        if apply_scale_factor:
            reaction_rate_per_unit_mass *= self.scale_factor_helium_3()
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_34(self, apply_scale_factor=True):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of helium 3 and helium 4, forming lithium 6.
        By default, the reaction rate per unit mass is scaled such that the
        rate of consumption for helium 3 does not exceed the rate at which it
        is produced, but the unscaled rate may also be computed, such that the
        scaling factor itself can be computed too.
        this is the first step within the PP 2 and PP 3 branches.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_34()
            * self.number_density("helium_3")
            * self.number_density("helium_4")
            / self.mass_density
        )
        if apply_scale_factor:
            reaction_rate_per_unit_mass *= self.scale_factor_helium_3()
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_e7(self, apply_scale_factor=True):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        electron capture by beryllium 7, forming lithium 7.
        By default, the reaction rate per unit mass is scaled such that the
        rate of consumption for helium 3 does not exceed the rate at which it
        is produced, but the unscaled rate may also be computed, such that the
        scaling factor itself can be computed too.
        This is the second step within the PP 2 branch.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_e7()
            * self.number_density("beryllium_7")
            * self.number_density_electron()
            / self.mass_density
        )
        if apply_scale_factor:
            reaction_rate_per_unit_mass *= self.scale_factor_beryllium_7()
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_17_(self, apply_scale_factor=True):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of lithium and hydrogen 1, forming helium 4.
        By default, the reaction rate per unit mass is scaled such that the
        rate of consumption for lithium 7 does not exceed the rate at which it
        is produced, but the unscaled rate may also be computed, such that the
        scaling factor itself can be computed too.
        This is the third and final step within the PP 2 branch.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_17_()
            * self.number_density("hydrogen_1")
            * self.number_density("lithium_7")
            / self.mass_density
        )
        if apply_scale_factor:
            reaction_rate_per_unit_mass *= self.scale_factor_lithium_7()
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_17(self, apply_scale_factor=True):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of hydrogen 1 and Beryllium 7, forming boron 8.
        Assuming near instantaneous decay of boron 8, forming beryllium 8, and
        the near instantaneous decay of beryllium 8, forming helium 4 nuclei,
        this is also the reaction rate per unit mass for both of these.
        By default, the reaction rate per unit mass is scaled such that the
        rate of consumption for helium 3 does not exceeds the rate at which it
        is procued, but the unscaled rate may also be computed, such that the
        scaling factor itself can be computed too.
        This is the second step within the PP3 branch.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_17()
            * self.number_density("hydrogen_1")
            * self.number_density("beryllium_7")
            / self.mass_density
        )
        if apply_scale_factor:
            reaction_rate_per_unit_mass *= self.scale_factor_beryllium_7()
        return reaction_rate_per_unit_mass

    def reaction_rate_per_unit_mass_p14(self):
        """
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of hydrogen 1 and nitrogen 14, forming oxygen 15.
        assuming all other reaction rates near instantaneous, this is the
        reaction rate per unit mass for all reactions within the CNO-cycle.
        this is the fourth step of the CNO-cycle.
        """
        reaction_rate_per_unit_mass = (
            self.reaction_rate_p14()
            * self.number_density("hydrogen_1")
            * self.number_density("nitrogen_14")
            / self.mass_density
        )
        return reaction_rate_per_unit_mass


class EnergyProduction(ReactionRatePerUnitMass):
    """
    computes the energy production rate for the PP branches, aswell as for the
    CNO-cycle.
    """

    ELECTRON_TO_JOULE_CONVERSION_FACTOR = 1.6022e-19 * 1e6  # Joule/MeV
    RELEASED_ENERGY_PP = 1.177 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_PD = 5.494 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_33 = 12.860 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_34 = 1.586 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_E7 = 0.049 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_17_ = 17.346 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_17 = 0.137 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_8 = 8.367 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_8_ = 2.995 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_P12 = 1.944 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_13 = 1.513 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_P13 = 7.551 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_P14 = 7.297 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_15 = 1.757 * ELECTRON_TO_JOULE_CONVERSION_FACTOR
    RELEASED_ENERGY_P15 = 4.966 * ELECTRON_TO_JOULE_CONVERSION_FACTOR

    def __init__(self, mass_density=162000, temperature=15700000):
        """
        initializes an object of EnergyProduction.
        """
        super().__init__(mass_density, temperature)

    def energy_production_rate_pp_1(self):
        """
        Computes the energy production rate for the PP 1 branch.
        
        The fusion of a proton and deuterium is nearly instantaneous. As such,
        it's reaction rate per unit mass is limited to, and therefore the same
        as the reaction rate per unit mass for the fusion of hydrogen 1 nuclei.

        For every iteration of the PP 1 branch, there are 2 iterations of PP 0
        hence the factor 2r_33 for the released energy by PP 0.
        
        """
        return (
            (self.RELEASED_ENERGY_PP + self.RELEASED_ENERGY_PD)
            * self.reaction_rate_per_unit_mass_33() * 2
            + self.RELEASED_ENERGY_33 * self.reaction_rate_per_unit_mass_33()
        )

    def energy_production_rate_pp_2(self):
        """
        Computes the energy production rate for the PP 2 branch.

        The fusion of a proton and deuterium is nearly instantaneous. As such,
        it's reaction rate per unit mass is limited to, and therefore the same
        as the reaction rate per unit mass for the fusion of hydrogen 1 nuclei.

        For the energy production rate for the PP 2 + PP 3, the contribution
        from the fusion of protons, aswell as the fusion of protons and 
        deuterium is (Q'_pp + Q'_pd)r_34. The contribution from the fusion of
        helium 3 and helium 4 is Q'_34*r_34.
        
        The contributions are both shared among the PP 2 and PP 3 branches
        through a weighted distribution, adjusting the reaction rates per unit
        mass according to the demand in each branch. for PP 2 we add a factor
        r_e7/(r_e7 + r_17)
        """
        normalizing_factor = 1 / (
            self.reaction_rate_per_unit_mass_e7()
            + self.reaction_rate_per_unit_mass_17()
        )
        return (
            (self.RELEASED_ENERGY_PP + self.RELEASED_ENERGY_PD)
            * self.reaction_rate_per_unit_mass_34()
            * self.reaction_rate_per_unit_mass_e7()
            * normalizing_factor


            + self.RELEASED_ENERGY_34 * self.reaction_rate_per_unit_mass_34()
            * self.reaction_rate_per_unit_mass_e7()
            * normalizing_factor
            #exclusive to PP 2 so we safe
            + self.RELEASED_ENERGY_E7 * self.reaction_rate_per_unit_mass_e7()
            + self.RELEASED_ENERGY_17_ * self.reaction_rate_per_unit_mass_17_()
        )

    def energy_production_rate_pp_3(self):
        """
        Computes the energy production rate for the PP 3 branch.

        The fusion of a proton and deuterium is nearly instantaneous. As such,
        it's reaction rate per unit mass is limited to, and therefore the same
        as the reaction rate per unit mass for the fusion of hydrogen 1 nuclei.

        For the energy production rate for the PP 2 + PP 3, the contribution
        from the fusion of protons, aswell as the fusion of protons and 
        deuterium is (Q'_pp + Q'_pd)r_34. The contribution from the fusion of
        helium 3 and helium 4 is Q'_34*r_34.

        The contributions are both shared among the PP 2 and PP 3 branches
        through a weighted distribution, adjusting the reaction rates per unit
        mass according to the demand in each branch. for PP 3 we add a factor
        r_17/(r_e7 + r_17) 

        The decay of boron 8, forming beryllium 8, along with the decay of
        beryllium 8, forming 2 helium 4 nuclei are nearly instantaneous. As
        such, their reaction rate per unit mass is  limited to, and therefore
        the same as the reaction rate per unit mass for the fusion of hydrogen
        1 and beryllium 7. 
        """
        normalizing_factor = 1 / (
            self.reaction_rate_per_unit_mass_e7()
            + self.reaction_rate_per_unit_mass_17()
        )

        return (
            (self.RELEASED_ENERGY_PP + self.RELEASED_ENERGY_PD)
            * self.reaction_rate_per_unit_mass_34()
            * self.reaction_rate_per_unit_mass_17()
            * normalizing_factor
            + self.RELEASED_ENERGY_34 * self.reaction_rate_per_unit_mass_34()
            * self.reaction_rate_per_unit_mass_17()
            * normalizing_factor
            + (
                self.RELEASED_ENERGY_17
                + self.RELEASED_ENERGY_8
                + self.RELEASED_ENERGY_8_
            )
            * self.reaction_rate_per_unit_mass_17()
        )

    def energy_production_rate_cno(self):
        """
        Compute the energy production rate for the CNO-cycle.

        The reaction rate per unit mass for all fusions in the CNO-cycle
        except for the fusion of nitrogen 14 and hydrogen 1 are near
        instantaneous. As such, their reaction rate per unit mass is limited
        to, and therefore the same as the reaction rate per unit mass for the
        fusion of nitrogen 14 and hydrogen 1.
        
        """
        return (
            np.sum(
                [
                    self.RELEASED_ENERGY_P12,
                    self.RELEASED_ENERGY_13,
                    self.RELEASED_ENERGY_P13,
                    self.RELEASED_ENERGY_P14,
                    self.RELEASED_ENERGY_15,
                    self.RELEASED_ENERGY_P15
                ]
            )
            * self.reaction_rate_per_unit_mass_p14()
        )

    def total_energy_production_rate(self):
        """
        Compute the total energy production rate in the scope of helium 4
        production.
        """
        return (
            np.sum(
                [
                    self.energy_production_rate_pp_1(),
                    self.energy_production_rate_pp_2(),
                    self.energy_production_rate_pp_3(),
                    self.energy_production_rate_cno()
                ]
            )
        )

    def print_energy_production_rates(self):
        print(
            f"""
Parameters: temperature = {self.temperature:.4g}, mass density = {self.mass_density:.4g}.
Energy production rate for PP 1 branch: {self.energy_production_rate_pp_1():.4g}J
Energy production rate for PP 2 branch: {self.energy_production_rate_pp_2():.4g}J
Energy production rate for PP 3 branch: {self.energy_production_rate_pp_3():.4g}J
Energy production rate for CNO-cycle: {self.energy_production_rate_cno():.4g}J
Total energy production rate: {self.total_energy_production_rate()}
            """
        )

    def compute_energy_production_rate_arrays(self, temperature_array):
        """
        computes a energy production rate matrix of size N x 5, where N is the
        # of temperatures in temperature array argument. this matrix is
        normalized, such that the sum of column elements equal 1.
        """
        temperature_array_size = len(temperature_array)
        data = np.empty((temperature_array_size, 5))
        for i in range(temperature_array_size):
            self.temperature9 = temperature_array[i]*1e-9
            data[i, 0] = self.energy_production_rate_pp_1()
            data[i, 1] = self.energy_production_rate_pp_2()
            data[i, 2] = self.energy_production_rate_pp_3()
            data[i, 3] = self.energy_production_rate_cno()
            data[i, :] /= np.sum(data[i, :])
        return data


class SanityCheck(EnergyProduction):
    """
    Ensures the validity of reaction rates aswell as reaction rates per unit
    mass inside EnergyProduction by comparing their output with known values.
    """

    def __init__(self):
        super().__init__()
        self.sanity_check()

    def _sanity_check_pp(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of hydrogen 1 nuclei, forming deuterium,
        aswell as the fusion of deuterium and hydrogen 3, forming helium 3.
        Useless for computing the energy production rate for PP branches,
        but used within the sanity check.
        """
        return (
            self.reaction_rate_per_unit_mass_pp()
            * (self.RELEASED_ENERGY_PP + self.RELEASED_ENERGY_PD)
            * self.mass_density
        )

    def _sanity_check_33(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of helium 3 nuclei, forming helium 4 and hydrogen 1,
        adjusting the consumption of helium 3 according to the rate at
        which it is produced.
        This is the first and only step within the PP 1 branch.
        """
        return (
            self.reaction_rate_per_unit_mass_33()
            * self.RELEASED_ENERGY_33
            * self.mass_density
        )

    def _sanity_check_34(self):
        """
        Computes energy production rate per unit volume [J/m^3/s] for the
        fusion of helium 3 and helium 4, forming beryllium 7 and hydrogen 1.
        adjusting the consumption of helium 3 according to rate at
        which it is produced.
        This is the first step within the PP 2 and PP 3 branches.
        """
        return (
            self.reaction_rate_per_unit_mass_34()
            * self.RELEASED_ENERGY_34
            * self.mass_density
        )

    def _sanity_check_e7(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        decay of beryllium 7, forming lithium 7.
        This is the second step within the PP 2 branch.
        """
        return (
            self.reaction_rate_per_unit_mass_e7()
            * self.RELEASED_ENERGY_E7
            * self.mass_density
        )

    def _sanity_check_17_(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of lithium 7 and hydrogen 1, forming helium 4.
        This is the third and final step within the PP 2 branch.
        """
        return (
            self.reaction_rate_per_unit_mass_17_()
            * self.RELEASED_ENERGY_17_
            * self.mass_density
        )

    def _sanity_check_17(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of beryllium 7 and hydrogen 1, forming boron 8, aswell as the
        near instantaneous decay of boron 8, forming beryllium 9, in addition
        to the the near instantaneous  decay of beryllium 8, forming helium 4
        nuclei, which are steps 2, 3 and 4 of the PP 3 branch, respectively.
        """
        sum_released_energies = np.sum(
            [self.RELEASED_ENERGY_17, self.RELEASED_ENERGY_8, self.RELEASED_ENERGY_8_]
        )
        return (
            self.reaction_rate_per_unit_mass_17()
            * sum_released_energies
            * self.mass_density
        )

    def _sanity_check_p14(self):
        """
        computes the energy production rate per unit volume [J/m^3/s] for the
        entire CNO-cycle, gated by the reaction rate per unit mass for the
        fusion of nitrogen 14 and hydrogen 1, as other fusions in the CNO-
        cycle are deemed near instantaneous.
        """
        sum_released_energies = sum(
            [
                self.RELEASED_ENERGY_P12,
                self.RELEASED_ENERGY_13,
                self.RELEASED_ENERGY_P13,
                self.RELEASED_ENERGY_P14,
                self.RELEASED_ENERGY_15,
                self.RELEASED_ENERGY_P15,
            ]
        )
        return (
            self.reaction_rate_per_unit_mass_p14()
            * sum_released_energies
            * self.mass_density
        )

    def sanity_check(self):
        """
        Verifies the computation of the various energy production rates by
        comparing the outputs with their respective known values at a mass
        density of the solar core, and temperatures of that of the solar core,
        aswell as 10^8 Kelvin.
        """
        self.temperature9 = 1.57e7 * 1e-9
        self.mass_density = 1.62e5
        assert (
            self.temperature == 1.57e7
        ), f"expected T = {1.57e7}K, actual T = {self.temperature}K."
        assert (
            self.mass_density == 1.62e5
        ), f"expected rho =  {1.62e5}kg/m^3, actual rho =  {self.mass_density}kg/m^3."

        evaluate(4.04e2, self._sanity_check_pp(), tolerance=1)
        evaluate(8.68e-9, self._sanity_check_33(), tolerance=1e-11)
        evaluate(4.86e-5, self._sanity_check_34(), tolerance=1e-7)
        evaluate(1.49e-6, self._sanity_check_e7(), tolerance=1e-8)
        evaluate(5.29e-4, self._sanity_check_17_(), tolerance=1e-6)
        evaluate(1.63e-6, self._sanity_check_17(), tolerance=1e-8)
        evaluate(9.18e-8, self._sanity_check_p14(), tolerance=1e-10)
        self.temperature9 = 1e8 * 1e-9
        evaluate(7.34e4, self._sanity_check_pp(), 1e2)
        evaluate(1.09, self._sanity_check_33(), 1e-2)
        evaluate(1.74e4, self._sanity_check_34(), 1e2)
        evaluate(1.22e-3, self._sanity_check_e7(), 1e-1)
        evaluate(4.35e-1, self._sanity_check_17_(), 1e-3)
        evaluate(1.26e5, self._sanity_check_17(), 1e3)
        evaluate(3.45e4, self._sanity_check_p14(), 1e2)


class GamowPeaks(ReactionRatePerUnitMass):
    """
    Compute gamow peaks for all processes in PP and CNO-cycles, excluding
    decays, (i.e. fusions and electron captures). Gamow peaks make no sense
    for processes which don't involve overcoming the coloumb barrier.
    """

    def __init__(self, energy_array=np.logspace(-17, -13, 1000), mass_density=162000, temperature=15700000):
        super().__init__(mass_density, temperature)
        self.energy_array = energy_array

    def gamow_peak(self, nucleon_1, nucleon_2):
        """ 
        generalized gamow peak method for computing the gamow peaks for
        the various processes in PP and CNO.
        
        k is the boltzmann constant (from revised SI)
        e is the elementary charge (from revised SI)
        eps_0 is the vacuum permittivity
        h is Plancks constant.
        """
        k = 1.380649e-23  # J/K
        e = 1.602176634e-19  # C
        eps_0 = 8.8541878128e-13  # F/m
        h = 6.62607015e-34
        mass1_kg = MASS[nucleon_1]*self.ATOMIC_MASS_UNIT
        mass2_kg = MASS[nucleon_2]*self.ATOMIC_MASS_UNIT
        reduced_mass = (
            mass1_kg*mass2_kg
            / (mass1_kg + mass2_kg)
        )
        lambda_exponent = np.exp(
            -self.energy_array / k / self.temperature
        )
        cross_section_exponent = np.exp(
            -np.sqrt(reduced_mass/2/self.energy_array)
            * int(nucleon_1.split("_")[1])
            * int(nucleon_2.split("_")[1])
            * e**2 * np.pi / eps_0 / h
        )
        probabilities = lambda_exponent*cross_section_exponent
        normalizing_factor = np.sum(probabilities)
        relative_probabilitis = probabilities/normalizing_factor
        return relative_probabilitis

    def gamow_peak_pp(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 nuclei, i.e. the
        first step of the PP cycle
        """
        return self.gamow_peak('hydrogen_1', 'hydrogen_1')

    def gamow_peak_pd(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and deuterium,
        i.e. the second step of the PP cycle.
        """
        return self.gamow_peak('hydrogen_1', 'deuterium_2')

    def gamow_peak_33(self):
        """
        compute the gamow peak for the fusion of helium 3 nuclei, i.e.
        the the first and only step of the PP 1 branch.
        """
        return self.gamow_peak('helium_3', 'helium_3')

    def gamow_peak_34(self):
        """
        compute the gamow peak for the fusion of helium 3 and helium 4 nuclei,
        the first step of the shared PP2/PP3 branch.
        """
        return self.gamow_peak('helium_3', 'helium_4')

    def gamow_peak_17_(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and lithium 7,
        i.e. the third and final step of the PP 2 branch.
        """
        return self.gamow_peak('hydrogen_1', 'lithium_7')

    def gamow_peak_17(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and beryllium 7,
        i.e. the second step of the PP 3 branch.
        """
        return self.gamow_peak('hydrogen_1', 'beryllium_7')

    def gamow_peak_p12(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and carbon 12,
        i.e. the first step in the CNO-cycle.
        """
        return self.gamow_peak('hydrogen_1', 'carbon_12')

    def gamow_peak_p13(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and carbon 13,
        i.e. the third step in the CNO-cycle.
        """
        return self.gamow_peak('hydrogen_1', 'carbon_13')

    def gamow_peak_p14(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and nitrogen 14,
        i.e. the fourth step of the CNO-cycle.
        """
        return self.gamow_peak('hydrogen_1', 'nitrogen_14')

    def gamow_peak_p15(self):
        """
        compute the gamow peak for the fusion of hydrogen 1 and nitrogen 15,
        i.e. the sixth and final step of the CNO-cycle.
        """
        return self.gamow_peak('hydrogen_1', 'nitrogen_15')

    def plot_gamow(self):
        """
        plot all computed gamow peaks curves.
        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        ax[0, 0].set_title("PP 0 & PP 1")
        ax[0, 0].plot(self.energy_array, self.gamow_peak_pp(),
                      linestyle='-', color='black', label='pp')
        ax[0, 0].plot(self.energy_array, self.gamow_peak_pd(),
                      linestyle='--', color='black', label='pd')
        ax[0, 0].plot(self.energy_array, self.gamow_peak_33(),
                      linestyle = ':', color='black', label='33')
        ax[0, 0].legend()
        ax[0, 0].set_xscale('log')
        ax[0, 0].set_xticks([1e-17,1e-16,1e-15,1e-14,1e-13])
        ax[0, 0].set_xlim(1e-17, 1e-13)
        ax[0, 1].set_title("PP 2")
        ax[0, 1].plot(self.energy_array, self.gamow_peak_34(),
                      linestyle='-', color='black', label='34')
        ax[0, 1].plot(self.energy_array, self.gamow_peak_17_(),
                      linestyle='--', color='black', label="17'")
        ax[0, 1].legend()
        ax[0, 1].set_xscale('log')
        ax[0, 1].set_xticks([1e-17,1e-16,1e-15,1e-14,1e-13])
        ax[0, 1].set_xlim(1e-17, 1e-13)
        ax[1, 0].set_title("PP 3")
        ax[1, 0].plot(self.energy_array, self.gamow_peak_17(),
                      color='black', label='17')
        ax[1, 0].legend()
        ax[1, 0].set_xscale('log')
        ax[1, 0].set_xticks([1e-17,1e-16,1e-15,1e-14,1e-13])
        ax[1, 0].set_xlim(1e-17, 1e-13)
        ax[1, 1].set_title("CNO")
        ax[1, 1].plot(self.energy_array, self.gamow_peak_p12(),
                      linestyle='-', color='black', label='p12')
        ax[1, 1].plot(self.energy_array, self.gamow_peak_p13(),
                      linestyle='--', color='black', label='p13')
        ax[1, 1].plot(self.energy_array, self.gamow_peak_p14(),
                      linestyle=':', color='black', label='p14')
        ax[1, 1].plot(self.energy_array, self.gamow_peak_p15(),
                      linestyle='-.', color='black', label='p15')
        ax[1, 1].legend()
        ax[1, 1].set_xscale('log')
        ax[1, 1].set_xticks([1e-14,1e-13])
        ax[1, 1].set_xlim(1e-14, 1e-13)
        # Create a "parent" subplot that spans all other subplots
        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top=False,
                       bottom=False, left=False, right=False)
        # Set the x-label for the parent subplot)
        # Add some space between the parent subplot and the other subplots
        plt.subplots_adjust(hspace=0.4)
        ax.set_xlabel("Energy [J]")
        ax.set_ylabel("Relative probability")
        plt.savefig('gamow_peaks.pdf')

    def gamow_max(self):
        """
        find and print the energy corresponding to the gamow peak.
        """
        print(
f"""
Gamow peak peaks:
pp: {self.energy_array[np.argmax(self.gamow_peak_pp())]:.4g}J
pd: {self.energy_array[np.argmax(self.gamow_peak_pd())]:.4g}J
33: {self.energy_array[np.argmax(self.gamow_peak_33())]:.4g}J
34: {self.energy_array[np.argmax(self.gamow_peak_34())]:.4g}J
17_: {self.energy_array[np.argmax(self.gamow_peak_17_())]:.4g}J
17: {self.energy_array[np.argmax(self.gamow_peak_17())]:.4g}J
p12: {self.energy_array[np.argmax(self.gamow_peak_p12())]:.4g}J
p13: {self.energy_array[np.argmax(self.gamow_peak_p13())]:.4g}J
p14: {self.energy_array[np.argmax(self.gamow_peak_p14())]:.4g}J
p15: {self.energy_array[np.argmax(self.gamow_peak_p15())]:.4g}J

34 in eV:
{self.energy_array[np.argmax(self.gamow_peak_34())]/1.602e-19:.4g}eV

"""
        )

energies()
SanityCheck()
energies = EnergyProduction()
energies.print_energy_production_rates()
temperatures = np.logspace(4, 9, 1000)
data = energies.compute_energy_production_rate_arrays(temperatures)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, data[:, 0], linestyle='-', color='black', label='PP 1')
ax.plot(temperatures, data[:, 1], linestyle='--', color='black', label='PP 2')
ax.plot(temperatures, data[:, 2], linestyle=':', color='black', label='PP 3')
ax.plot(temperatures, data[:, 3], linestyle='-.', color='black', label='CNO')
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Relative energy production")
ax.set_xscale('log')
# set the x-ticks to follow the log scale
ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax.set_xlim(1e4, 1e9)
ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1))
plt.savefig("energy_production_rates.pdf")

gamow = GamowPeaks()
gamow.plot_gamow()
gamow.gamow_max()