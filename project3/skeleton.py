# visualiser
import FVis3

import numpy as np
import matplotlib.pyplot as plt

class TwoDConvection:
    x_max, y_max = 12e6, 4e6
    nx, ny = 300, 100
    dx = x_max/nx
    dy = y_max/ny
    temp_sun = 5778  # Kelvin
    mass_sun = 1.989e30  # kg
    radius_sun = 6.96e8  # m
    pressure_sun = 1.8e4 #Pa
    grav_const = 6.6742e-11  # N m^2 /kg^2
    grav_acc = grav_const*mass_sun/radius_sun**2  # m / s^2
    gamma = 5/3
    nabla = 4.1
    atomic_mass_unit = 1.6605e-27 #kg
    avg_atomic_weight = 0.61
    boltzmann_const = 1.3806e-23 # m^2 kg/(s^2 K)
    dTdr = -grav_acc*nabla*avg_atomic_weight*atomic_mass_unit/boltzmann_const

    def __init__(self):
        """
        Define variables
        """
        self.rho = np.empty((self.nx, self.ny), dtype=float)
        self.T = np.empty((self.nx, self.ny), dtype=float)
        self.P = np.empty((self.nx, self.ny), dtype=float)
        self.e = np.empty((self.nx, self.ny), dtype=float)
        self.u = np.zeros((self.nx, self.ny), dtype=float)
        self.w = np.zeros((self.nx, self.ny), dtype=float)   


    def initialise(self):
        """
        Initialise temperature, pressure, density and internal energy
        """
        self.T[:,0] = self.temp_sun
        self.T[:,1:] = self.T[:,:-1] - self.dTdr*self.dy
        self.P = self.pressure_sun*(self.T/self.temp_sun)**(1/self.nabla)
        self.rho = (self.P * self.atomic_mass_unit * self.avg_atomic_weight 
                    / (self.boltzmann_const * self.T))
        self.e = self.P/(self.gamma - 1)


    def timestep(self):
        """
        Calculate timestep
        """
        p = 0.1
        rel_phi = self.phi/self.dphidt
        np.max()
        self.dt = p/delta

    def boundary_conditions(self):
        """
        Boundary conditions for energy, density and velocity
        """
        self.u[:,0] = 0
        self.u[:,-1] = 0
        self.w[:,0] = 0
        self.w[:,-1] = 0 
        self.e[:,0] = ((3/(2*self.dy) - self.grav_acc*self.avg_atomic_weight*self.atomic_mass_unit/(self.boltzmann_const*self.T[:,0]))**(-1)*(self.e[:,2] + 4*self.e[:,1])/(2*self.dy))
        self.e[:,-1] = ((3/(2*self.dy) - self.grav_acc*self.avg_atomic_weight*self.atomic_mass_unit/(self.boltzmann_const*self.T[:,-1]))**(-1)*()/(2*self.dy))
        self.rho[:,0]
        self.rho[:,-1]
        
        

        

    def central_x(self, var):
        """
        Central difference scheme in x-direction.

        Args:
            (TwoDConvection): Instance of TwoDConvection. 
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                central difference scheme in x-direction.
        """
        var_ip1 = np.roll(var, -1, 0)
        var_im1 = np.roll(var, 1, 0)
        return (var_ip1 - var_im1)/(2*self.dx)

    def central_y(self, var):
        """
        Central difference scheme in y-direction

        Args:
            (TwoDConvection): Instance of TwoDConvection.
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                central difference scheme in y-direction.
        """
        var_jp1 = np.roll(var, -1, 1)
        var_jm1 = np.roll(var, 1, 0)
        dvardy_ = (var_jp1 - var_jm1)/(2*self.dy)
        dvardy = np.empty_like(var)
        dvardy[:, 1:-1] = dvardy_[:, 1:-1]
        return dvardy

    def upwind_x(self, var, v):
        """
        Upwind difference scheme in x-direction.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                upwind difference scheme in x-direction, e.grav_acc. rho_x, rho_u_x.
            v (numpy.ndarray[float]):  Horizontal velocity component, i.e. u.
        
        Returns:
            (numpy.ndarray[float]): d(var)/dx. (correct at boundaries)
        """
        var_ip1 = np.roll(var, -1, 0)
        var_im1 = np.roll(var, 1, 0)
        return np.where(
            v < 0,
            (var_ip1 - var)/self.dx,
            (var - var_im1)/self.dx
        )

    def upwind_y(self, var, v):
        """
        Upwind difference scheme in y-direction.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                upwind difference scheme in y-direction, e.grav_acc. rho_y, rho_u_y.
            v (numpy.ndarray[float]):  Vertical velocity component, i.e. w.
        
        Returns:
            (numpy.ndarray[float]): d(var)/dy. (wrong at boundaries)
        """
        var_jp1 = np.roll(var, -1, 1)
        var_jm1 = np.roll(var, 1, 1)
        dvardy = np.empty_like(var)
        dvardy_ = np.where(v < 0,
                           (var_jp1 - var)/self.dy,
                           (var - var_jm1)/self.dy
                           )
        dvardy[:, 1:-1] = dvardy_[:, 1:-1]
        return dvardy

    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        self.initialise()


    def plot_initialise(self):
        """
        Plots initials
        """
        self.initialise()
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        im1 = axs[0, 0].imshow(self.P, aspect='auto')
        axs[0, 0].set_xlabel('y')
        axs[0, 0].set_ylabel('x')
        axs[0, 0].set_title('Pressure (P)')
        fig.colorbar(im1, ax=axs[0, 0])
        
        im2 = axs[0, 1].imshow(self.T, aspect='auto')
        axs[0, 1].set_xlabel('y')
        axs[0, 1].set_ylabel('x')
        axs[0, 1].set_title('Temperature (T)')
        fig.colorbar(im2, ax=axs[0, 1])
        
        im3 = axs[1, 0].imshow(self.e, aspect='auto')
        axs[1, 0].set_xlabel('y')
        axs[1, 0].set_ylabel('x')
        axs[1, 0].set_title('Internal Energy (e)')
        fig.colorbar(im3, ax=axs[1, 0])
        
        im4 = axs[1, 1].imshow(self.rho, aspect='auto')
        axs[1, 1].set_xlabel('y')
        axs[1, 1].set_ylabel('x')
        axs[1, 1].set_title('Density (rho)')
        fig.colorbar(im4, ax=axs[1, 1])
        
        fig.tight_layout()
        plt.show()



if __name__ == '__main__':
    # Run your code here
    instance = TwoDConvection()
    instance.initialise()
    instance.plot_initialise()
