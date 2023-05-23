import scipy.constants as const
import numpy as np
# Define the values of 1 Mpc and 1 kpc in meters
MPC_TO_M = const.parsec * 1e6
KPC_TO_M = const.parsec * 1e3

def knudsen_number(particle_quantity, diameter_system, height_system, diameter_particle):
    """
    Compute the Knudsen number for some system that is spherical or disc
        shaped.

    Args:
        particle_quantity (float): Number of "particles", e.g. stars,
            galaxies.
        diameter_system (float): Diameter of system we compute knudsen number
            for.
        height_system (float): Height of system if system is a disc. if 0,
            system assumed spherical.
    
    Returns:
        (float): Knudsen number.
    """
    # Calculate the volume of the system
    radius_system = diameter_system / 2
    # Assume disc if height != 0
    volume_system = np.pi * radius_system**2 * height_system
    # Assume sphere if height == 0
    if height_system == 0:
        volume_system = (4/3) * np.pi * radius_system**3

    # Calculate the number density
    number_density = particle_quantity / volume_system
    
    # Calculate the cross-sectional area
    cross_section_particle = np.pi * diameter_particle**2
    
    # Calculate the mean free path
    mean_free_path_particle = 1 / (cross_section_particle * number_density)
    
    # Calculate the Knudsen number
    knudsen_number = mean_free_path_particle / diameter_system
    
    return knudsen_number



def main():
    """ Galaxy knudsen number computation"""

    N_STARS = 300e9
    # Also the length scale
    DIAMETER_GALAXY = 35*KPC_TO_M #m
    HEIGHT_GALAXY = 0.5*KPC_TO_M #m
    RADIUS_SUN = 6.96e8*np.exp(1) #m
    DIAMETER_SUN = RADIUS_SUN*2


    KNUDSEN_NUMBER_GALAXY = knudsen_number(N_STARS, DIAMETER_GALAXY, HEIGHT_GALAXY, DIAMETER_SUN)

    GALAXY_IS_FLUID = KNUDSEN_NUMBER_GALAXY < 1
    """ Galaxy cluster knudsen number calc"""
    N_GALAXIES = 2000
    DIAMETER_GALAXY_CLUSTER = 2.2*MPC_TO_M


    KNUDSEN_NUMBER_GALAXY_CLUSTER = knudsen_number(N_GALAXIES, DIAMETER_GALAXY_CLUSTER, 0, DIAMETER_GALAXY)
    GALAXY_CLUSTER_IS_FLUID = KNUDSEN_NUMBER_GALAXY_CLUSTER < 1

    print(f"""
    knudsen number for galaxy is {KNUDSEN_NUMBER_GALAXY:.4g}
    Galaxy is fluid: {GALAXY_IS_FLUID}

    Knudsen number for galaxy cluster is {KNUDSEN_NUMBER_GALAXY_CLUSTER:.4g}
    Galaxy cluster is fluid: {GALAXY_CLUSTER_IS_FLUID}
    """)

if __name__ == "__main__":
    main()