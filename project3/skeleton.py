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
    pressure_sun = 1.8e4  # Pa
    grav_const = 6.6742e-11  # N m^2 /kg^2
    grav_acc = grav_const*mass_sun/radius_sun**2  # m / s^2
    gamma = 5/3
    nabla = 4.1
    atomic_mass_unit = 1.6605e-27  # kg
    avg_atomic_weight = 0.61
    boltzmann_const = 1.3806e-23  # m^2 kg/(s^2 K)
    dTdr = -grav_acc*nabla*avg_atomic_weight*atomic_mass_unit/boltzmann_const
    y = np.linspace(4e6, 0, ny)

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
        self.rho_u = np.zeros((self.nx, self.ny), dtype=float)
        self.rho_w = np.zeros((self.nx, self.ny), dtype=float)

    def initialise(self):
        """
        Initialise temperature, pressure, density and internal energy
        """
        temp_1d = (self.grav_acc*self.atomic_mass_unit*self.avg_atomic_weight   
                   * self.y/self.boltzmann_const + self.temp_sun)
        self.T = np.tile(temp_1d[:, np.newaxis], (1, self.ny))
        self.P = self.pressure_sun*(self.T/self.temp_sun)**(1/self.nabla)
        self.rho = (self.P * self.atomic_mass_unit * self.avg_atomic_weight
                    / (self.boltzmann_const * self.T))
        self.e = self.P/(self.gamma - 1)

    def timestep(self):
        """
        Calculate timestep
        """
        p = 0.1
        rel_rho = self.rho_diff_t/self.rho
        rel_u = self.u_diff_t/self.u
        rel_w = self.w_diff_t/self.w
        rel_e = self.e_diff_t/self.e
        delta = np.max(np.asarray([np.max(rel_rho), np.max(rel_u), np.max(rel_w), np.max(rel_e)]))
        self.dt = p/delta
        rel_x = self.u/self.dx
        rel_y = self.w/self.dy



    def boundary_conditions(self):
        """
        Boundary conditions for energy, density and velocity
        """
        consts_factor = self.grav_acc*self.atomic_mass_unit*self.avg_atomic_weight/self.boltzmann_const
        self.T[:, 0] = ((4*self.T[:, 1] - self.T[:, 2])/3 + 2*self.dy/3*consts_factor*self.nabla)
        self.e[:, 0] = (2/(3 * self.dy) - consts_factor/self.T[:, 0])**(-1)*(4*self.e[:, 1] - self.e[:, 2])/(2*self.dy)
        self.rho[:, 0] = self.e[:, 0]*(self.gamma - 1)*consts_factor/(self.grav_acc*self.T[:, 0])
        self.T[:, -1] =  ((4*self.T[:, -2] - self.T[:, -3])/3 + 2*self.dy*consts_factor*self.nabla/3)
        self.e[:, -1] = (3/(2*self.dy) + consts_factor/self.T[:, -1])**(-1)*(4*self.e[:, -2] - self.e[:, -3])/(2*self.dy)
        self.rho[:, -1] = self.e[:, -1]*(self.gamma - 1)*consts_factor/(self.grav_acc*self.T[:, -1])
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        self.w[:, 0] = 0
        self.w[:, -1] = 0


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
        while True:
            #self.boundary_conditions()
            u_diff_x = self.central_x(self.u)
            u_diff_y = self.central_y(self.u)
            w_diff_x = self.central_x(self.w)
            w_diff_y = self.central_y(self.w)
            rho_diff_x = self.upwind_x(self.rho, self.u)
            rho_diff_y = self.upwind_y(self.rho, self.w)   
            rho_u_diff_x = self.upwind_x(self.rho_u, self.u)
            rho_u_diff_y = self.upwind_y(self.rho_u, self.w)
            rho_w_diff_x = self.upwind_x(self.rho_w, self.u)
            rho_w_diff_y = self.upwind_y(self.rho_w, self.w)
            P_diff_x = self.central_x(self.P)
            P_diff_y = self.central_y(self.P)
            self.rho_diff_t = -self.rho*(u_diff_x + w_diff_y) - self.u*rho_diff_x - self.w*rho_diff_y 
            rho_u_diff_t = -self.rho_u*(u_diff_x + w_diff_y) - self.u*rho_u_diff_x  - self.w*rho_u_diff_y - P_diff_x
            rho_w_diff_t = -self.rho_w(w_diff_x + u_diff_y) - self.w*rho_w_diff_x - self.u*rho_w_diff_y + self.rho*self.grav_acc
            self.u_diff_t = rho_u_diff_t/self.rho
            self.w_diff_t = rho_w_diff_t/self.rho
            e_diff_x = self.upwind_x(self.e, self.u)
            e_diff_y = self.upwind_y(self.e, self.w)
            self.e_diff_t = -self.u*e_diff_x -self.w*e_diff_y - (self.P + self.e)*(u_diff_x + w_diff_y)



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
