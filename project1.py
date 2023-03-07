"""
Script simulates the Energy production at the center of a star.
"""
import numpy as np

MASS = {  # u
    "hydrogen_1": 1.007825,
    "helium_3": 3.0160,
    "helium_4": 4.0026,
}


def converted_mass_to_energy(mass_difference):
    """
    applies the mass-energy equivalency principle E=mc^2 to convert mass
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


def energies():
    """
    compute the energy lost via neutrinos for the PP branches, and compares
    with the total energy produced per PP branch.
    """
    neutrino_energy_lost_pp0 = 0.265  # MeV
    neutrino_energy_lost_pp1 = 2 * neutrino_energy_lost_pp0
    neutrino_energy_lost_pp2 = 0.815 + neutrino_energy_lost_pp0  # MeV
    neutrino_energy_lost_pp3 = 6.711 + neutrino_energy_lost_pp0   # MeV
    print(
        f"""
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
"""
    )


energies()


def evaluate(expected, computed, tolerance=1e-5):
    """
    Verifies if computations match their expected value, used exclusively
    by sanity_check() within EnergyProduction.
    """
    error = np.abs(expected - computed)
    error_message = (
        f"expected {expected:.4g}, computed {computed:.4g}, error: {error:.4g} "
    )
    assert error < tolerance, error_message


def convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro):
    """
    Converts the units of reaction rates from  [reactions*cm^3/s/mole]
    to [reactions*m^3/s]. Applied inside reaction_rate_xx() methods within
    EnergyProduction, where xx denotes which process the reaction rate
    corresponds to.
    """
    avogadros_number = 6.0221e23  # 1/mol
    reaction_rate_cm = reaction_rate_cm_avogadro / avogadros_number  # reactions*cm^3/s
    reaction_rate_m = reaction_rate_cm / 1e6  # reactions*m^3/s
    return reaction_rate_m


class EnergyProduction:
    """
    Calculates the energy production rate at the core of a star based on the
    input temperature [K] and mass density [kg/m^3].
    """

    MEV_TO_JOULE_CONVERSION_FACTOR = 1.6022e-19 * 1e6  # Joule/MeV
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
        to the values of that of the solar core, though may be overwritten.
        """
        self.mass_density = mass_density  # kg/m^3
        self.temperature = temperature  # K
        self.temperature9 = temperature * 1e-9  # 10^9K

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
                2 * self.reaction_rate_per_unit_mass_33(apply_scale_factor=False),
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

    def _reaction_rate_pp(self):
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_33(self):
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_34(self):
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_e7(self):
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
        if self.temperature < 1e6:
            reaction_rate_m = convert_cm_to_m_reaction_rate(upper_limit_cm_avogadro)
            return reaction_rate_m
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_17_(self):
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_17(self):
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
        reaction_rate_m = convert_cm_to_m_reaction_rate(reaction_rate_cm_avogadro)
        return reaction_rate_m

    def _reaction_rate_p14(self):
        """
        Computes the reaction rate [reactions*m^3/s] for the fusion of
        nitrogen 14 and hydrogen 1, forming oxygen 15.
        This is the fourth step of the CNO-cycle.
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
        Computes the reaction rate per unit mass [reactions/s/kg] for the
        fusion of hydrogen 1 nuclei, forming deuterium. Assuming near
        instantaneous fusion of hydrogen 1 and deuterium, this is also
        the reaction rate per unit mass for hydrogen 1 and deuterium,
        forming helium 3.
        this is the first step of the PP chain.
        """
        reaction_rate_per_unit_mass = (
            self._reaction_rate_pp()
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
            self._reaction_rate_33()
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
        self._reaction_rate_34()
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
            self._reaction_rate_e7()
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
            self._reaction_rate_17_()
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
            self._reaction_rate_17()
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
        reaction rate per unit mass for all reactions within the CNO cycle.
        this is the fourth step of the CNO-cycle.
        """
        reaction_rate_per_unit_mass = (
            self._reaction_rate_p14()
            * self.number_density("hydrogen_1")
            * self.number_density("nitrogen_14")
            / self.mass_density
        )
        return reaction_rate_per_unit_mass

    def _sanity_check_pp(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of hydrogen 1 nuclei, forming deuterium,
        aswell as the fusion of deuterium and hydrogen 3, forming helium 3.
        Useless for computing the energy production rate for PP branches,
        but used within the sanity check.
        """
        released_energy_pp = 1.177 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_pd = 5.494 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        reaction_rate_per_unit_mass = (
            self.reaction_rate_per_unit_mass_pp()
            * (released_energy_pp + released_energy_pd)
            * self.mass_density
        )
        return reaction_rate_per_unit_mass

    def _sanity_check_33(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of helium 3 nuclei, forming helium 4 and hydrogen 1,
        adjusting the consumption of helium 3 according to the rate at
        which it is produced.
        This is the first and only step within the PP 1 branch.
        """
        released_energy_33 = 12.860 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        return (
            self.reaction_rate_per_unit_mass_33()
            * released_energy_33
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
        released_energy_34 = 1.586 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        return (
            self.reaction_rate_per_unit_mass_34()
            * released_energy_34
            * self.mass_density
        )

    def _sanity_check_e7(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        decay of beryllium 7, forming lithium 7.
        This is the second step within the PP 2 branch.
        """
        released_energy_e7 = 0.049 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        return (
            self.reaction_rate_per_unit_mass_e7()
            * released_energy_e7
            * self.mass_density
        )

    def _sanity_check_17_(self):
        """
        Computes the energy production rate per unit volume [J/m^3/s] for the
        fusion of lithium 7 and hydrogen 1, forming helium 4.
        This is the third and final step within the PP 2 branch.
        """
        released_energy_17_ = 17.346 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        return (
            self.reaction_rate_per_unit_mass_17_()
            * released_energy_17_
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
        released_energy_17 = 0.137 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_8 = 8.367 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_8_ = 2.995 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy = sum(
            [released_energy_17, released_energy_8, released_energy_8_]
        )
        return (
            self.reaction_rate_per_unit_mass_17() 
            * released_energy * self.mass_density
        )

    def _sanity_check_p14(self):
        """
        computes the energy production rate per unit volume [J/m^3/s] for the
        entire CNO cycle, gated by the reaction rate per unit mass for the
        fusion of nitrogen 14 and hydrogen 1, as other fusions in the CNO-
        cycle are deemed near instantaneous.
        """
        released_energy_p12 = 1.944 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_13 = 1.513 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_p13 = 7.551 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_p14 = 7.297 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_15 = 1.757 * self.MEV_TO_JOULE_CONVERSION_FACTOR
        released_energy_p15 = 4.966 * self.MEV_TO_JOULE_CONVERSION_FACTOR
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

    def energy_production_rate_pp_1(self):
        """
        Computes the energy production rate for the PP 1 branch, by summing
        the energy production rates for all relevant steps.
        steps used by multiple PP branches are not normalized, and as such,
        this method may compute the wrong value.
        """
        return np.sum([
            self.energy_production_rate_pp(),
            self.energy_production_rate_33()
        ])
    
    def energy_production_rate_pp_2(self):
        return np.sum([
            self.energy_production_rate_pp(),
            self.energy_production_rate_34(),
            self.energy_production_rate_e7(),
            self.energy_production_rate_17_()
        ])

    def energy_production_rate_pp_3(self):
        return np.sum([
            self.energy_production_rate_pp(),
            self.energy_production_rate_34(),
            self.energy_production_rate_17()
        ])

    def sanity_check(self):
        """
        Verifies the computation of the various energy production rates by
        comparing the outputs with their respective known values at a mass
        density of the solar core, and temperatures of that of the solar core,
        aswell as 10^8 Kelvin.
        """
        temp = self.temperature9
        temp2 = self.mass_density
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
        self.temperature9 = temp
        self.mass_density = temp2


instance = EnergyProduction()
instance.sanity_check()
