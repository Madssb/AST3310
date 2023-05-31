# visualiser
import FVis3 as FVis
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class TwoDConvection:
    """
    Simulate convection in photosphere of sun for two dimensions.
    """
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
    nabla = 0.44
    atomic_mass_unit = 1.6605e-27  # kg
    avg_atomic_weight = 0.61
    boltzmann_const = 1.3806e-23  # m^2 kg/(s^2 K)
    dTdy = grav_acc*nabla*avg_atomic_weight*atomic_mass_unit/boltzmann_const
    y = np.linspace(0, y_max, ny)

    def __init__(self):
        """
        Define & initialize arrays

        Args:
            (TwoDConvection): Instance of TwoDConvection. 
        """
        self.rho = np.empty((self.nx, self.ny), dtype=float)
        self.rho_diff_t = np.empty((self.nx, self.ny), dtype=float)
        self.rho_u_diff_t = np.empty((self.nx, self.ny), dtype=float)
        self.rho_w_diff_t = np.empty((self.nx, self.ny), dtype=float)
        self.e_diff_t = np.empty((self.nx, self.ny), dtype=float)
        self.T = np.empty((self.nx, self.ny), dtype=float)
        self.P = np.empty((self.nx, self.ny), dtype=float)
        self.e = np.empty((self.nx, self.ny), dtype=float)
        self.u = np.zeros((self.nx, self.ny), dtype=float)
        self.w = np.zeros((self.nx, self.ny), dtype=float)
        self.u = np.random.uniform(-1,1,size=(self.nx,self.ny))
        self.w = np.random.uniform(-1,1,size=(self.nx,self.ny))
        self.rho_u = np.zeros((self.nx, self.ny), dtype=float)
        self.rho_w = np.zeros((self.nx, self.ny), dtype=float)
        self.temperature_pertubation = np.zeros(
            (self.nx, self.ny), dtype=float)
        self.t = 0

    def initialise(self):
        """
        Initialise temperature, pressure, density, and internal energy.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
        """
        const1 = 1/(self.gamma - 1)
        const2 = (self.avg_atomic_weight*self.atomic_mass_unit
                  * (self.gamma - 1)/self.boltzmann_const)
        temp_1d = (self.dTdy*(self.y_max - self.y)) + self.temp_sun
        self.T = np.tile(temp_1d, (self.nx, 1))
        self.P = self.pressure_sun*(self.T/self.temp_sun)**(1/self.nabla)
        self.T += self.temperature_pertubation
        self.e = self.P * const1
        self.rho = const2 * self.e / self.T

        assert not np.any(np.isnan(self.T)), "Error: T contains NaN values"
        assert not np.any(np.isnan(self.P)), "Error: P contains NaN values"
        assert not np.any(np.isnan(self.rho)), "Error: rho contains NaN values"
        assert not np.any(np.isnan(self.e)), "Error: e contains NaN values"
        assert np.any(self.T > 0), "Error: T has zeros or negative vals"
        assert np.any(self.P > 0), "Error: P has zeros or negative vals"
        assert np.any(self.rho > 0), "Error: rho has zeros or negative vals"
        assert np.any(self.e > 0), "Error: e has zero or negative vals"

    def timestep(self):
        """
        Calculate timestep.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
        """
        p = 0.1

        indices_1 = np.where(np.abs(self.u) > 1e-10)[0]
        indices_2 = np.where(np.abs(self.w) > 1e-10)[0]
        combined = np.concatenate((indices_1, indices_2))
        indices = np.unique(combined)
        if len(indices) != 0:
            max_rel_rho_u = np.nanmax(
                np.abs(self.rho_u_diff_t[indices] / self.rho[indices]))
            max_rel_rho_w = np.nanmax(
                np.abs(self.rho_w_diff_t[indices] / self.rho[indices]))
            max_rel_rho = np.nanmax(
                np.abs(self.rho_diff_t[indices] / self.rho[indices]))
            max_rel_e = np.nanmax(
                np.abs(self.e_diff_t[indices] / self.e[indices]))
        else:
            max_rel_rho_u = 0
            max_rel_rho_w = 0
            max_rel_rho = 0
            max_rel_e = 0
        max_rel_x = np.nanmax(np.abs(self.u / self.dx))
        max_rel_y = np.nanmax(np.abs(self.w / self.dy))
        delta = np.nanmax([max_rel_rho, max_rel_rho_u,
                          max_rel_rho_w, max_rel_e, max_rel_x, max_rel_y])
        if delta == 0:
            delta = 1
        self.dt = p / delta
        if self.dt < 0.1:
            self.dt = 0.1
        assert not np.isnan(self.dt), "Error: self.dt is NaN"
        assert np.isfinite(self.dt), "Error: self.dt is Inf"

    def boundary_conditions(self):
        """
        Boundary conditions for energy, density, and velocity.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
        """
        consts = self.atomic_mass_unit * self.avg_atomic_weight * \
            (self.gamma - 1) / self.boltzmann_const

        dedy_top = -(self.e / self.T)[:, 0] * self.grav_acc * self.atomic_mass_unit * \
            self.avg_atomic_weight * (self.gamma - 1) / self.boltzmann_const
        dedy_term_top = -2 * self.dy * dedy_top / 3
        dedy_bot = -(self.e / self.T)[:, -1] * self.grav_acc * self.atomic_mass_unit * \
            self.avg_atomic_weight * (self.gamma - 1) / self.boltzmann_const
        dedy_term_bot = 2 * self.dy * dedy_bot / 3

        self.e[:, 0] = dedy_term_top - self.e[:, 2] / 3 + 4 * self.e[:, 1] / 3
        self.e[:, -1] = dedy_term_bot + 4 * self.e[:, -2] / 3 - self.e[:, -3] / 3
        self.rho[:, 0] = (self.e / self.T)[:, 0] * consts
        self.rho[:, -1] = (self.e / self.T)[:, -1] * consts
        self.u[:, 0] = 4 * self.u[:, 1] / 3 - self.u[:, 2] / 3
        self.w[:, 0] = 0
        self.rho_u[:, 0] = 0
        self.rho_w[:, 0] = 0
        self.u[:, -1] = 4 * self.u[:, -2] / 3 - self.u[:, -3] / 3
        self.w[:, -1] = 0



    def central_x(self, var):
        """
        Central difference scheme in x-direction.

        Args:
            self (TwoDConvection): Instance of TwoDConvection. 
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                central difference scheme in x-direction.
        """
        var_ip1 = np.roll(var, -1, 0)
        var_im1 = np.roll(var, 1, 0)
        var_diff_x = (var_ip1 - var_im1)/(2*self.dx)
        return var_diff_x[:, 1:-1]

    def central_y(self, var):
        """
        Central difference scheme in y-direction

        Args:
            (TwoDConvection): Instance of TwoDConvection.
            var (numpy.ndarray[float]): Array for quantity to be advanced with
                central difference scheme in y-direction.
        """
        var_jp1 = np.roll(var, -1, 1)
        var_jm1 = np.roll(var, 1, 1)
        var_diff_y = (var_jp1 - var_jm1)/(2*self.dy)
        return var_diff_y[:, 1:-1]

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
        var_diff_x = np.where(
            v < 0,
            (var_ip1 - var) / self.dx,
            (var - var_im1) / self.dx
        )
        return var_diff_x[:, 1:-1]

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
        var_diff_y = np.where(v < 0,
                          (var_jp1 - var)/self.dy,
                          (var - var_jm1)/self.dy
                          )
        return var_diff_y[:, 1:-1]

    def hydro_solver(self):
        """
        hydrodynamic equations solver.

        Args:
            (TwoDConvection): Instance of TwoDConvection. 
        """
        self.variable_checks()
        """Continuity eq"""
        rho_diff_x = self.upwind_x(self.rho, self.u)
        rho_diff_y = self.upwind_y(self.rho, self.w)
        u_diff_x = self.central_x(self.u)
        w_diff_y = self.central_y(self.w)
        self.rho_diff_t[:, 1:-1] = (-self.rho[:, 1:-1] * (u_diff_x + w_diff_y)
                                    - self.u[:, 1:-1] * rho_diff_x - self.w[:, 1:-1] * rho_diff_y)
        self.rho_diff_t[:, 0] = 0
        self.rho_diff_t[:, -1] = 0

        """Horizontal momentum eq"""
        u_diff_x = self.upwind_x(self.u, self.u)
        w_diff_y = self.central_y(self.w)
        rho_u_diff_x = self.upwind_x(self.rho_u, self.u)
        rho_u_diff_y = self.upwind_y(self.rho_u, self.w)
        P_diff_x = self.central_x(self.P)
        self.rho_u_diff_t[:, 1:-1] = -self.rho_u[:, 1:-1]*(
            u_diff_x + w_diff_y) - self.u[:, 1:-1]*rho_u_diff_x - self.w[:, 1:-1]*rho_u_diff_y - P_diff_x
        self.rho_u_diff_t[:, 0] = 0
        self.rho_u_diff_t[:, -1] = 0

        """Vertical momentum eq"""
        w_diff_x = self.upwind_x(self.w, self.u)
        u_diff_y = self.central_y(self.u)
        rho_w_diff_x = self.upwind_x(self.rho_w, self.w)
        rho_w_diff_y = self.upwind_y(self.rho_w, self.u)
        P_diff_y = self.central_y(self.P)
        self.rho_w_diff_t[:, 1:-1] = -self.rho_w[:, 1:-1]*(w_diff_x + u_diff_y) - self.w[:, 1:-1] * \
            rho_w_diff_x - self.u[:, 1:-1]*rho_w_diff_y - \
            P_diff_y - self.rho[:, 1:-1]*self.grav_acc
        self.rho_w_diff_t[:, 0] = 0
        self.rho_w_diff_t[:, -1] = 0

        """Energy eq"""
        e_diff_x = self.upwind_x(self.e, self.u)
        e_diff_y = self.upwind_y(self.e, self.w)
        u_diff_x = self.central_x(self.u)
        w_diff_y = self.central_y(self.w)
        self.e_diff_t[:, 1:-1] = -self.u[:, 1:-1]*e_diff_x - self.w[:, 1:-1] * \
            e_diff_y - (self.e[:, 1:-1] + self.P[:, 1:-1]) * \
            (u_diff_x + w_diff_y)
        self.e_diff_t[:, 0] = 0
        self.e_diff_t[:, -1] = 0

        """Assign timestep, and advance quantities forward in time"""
        self.timestep()
        self.rho[:] += self.rho_diff_t * self.dt
        self.u[:] = (self.rho_u + self.rho_u_diff_t * self.dt) / self.rho
        self.w[:] = (self.rho_w + self.rho_w_diff_t * self.dt) / self.rho
        self.rho_u[:] += self.rho_u_diff_t * self.dt
        self.rho_w[:] += self.rho_w_diff_t * self.dt
        self.e[:] += self.e_diff_t * self.dt

        self.boundary_conditions()

        factor = (self.avg_atomic_weight * self.atomic_mass_unit
                  * (self.gamma - 1) / self.boltzmann_const)
        self.T[:] = self.e * factor / self.rho
        self.P[:] = (self.gamma - 1) * self.e
        self.t += self.dt
        #print(f"{np.mean(self.rho)=:.4g},\t {np.mean(self.T)=:.4g},\t {np.mean(self.P)=:.4g},\t {np.mean(self.e)=:.4g}")
        return self.dt

    def variable_checks(self):
        """
        Verifies that various quantities behave as expected.

        Args:
            (TwoDConvection): Instance of TwoDConvection.
        """
        try:
            assert np.all(self.rho > 0), "Error: rho has zeros or negative vals"
            assert np.all(self.e > 0), "Error: e has zero or negative vals"
            assert np.all(self.T > 0), "Error: T has zeros or negative vals"
            assert np.all(self.P > 0), "Error: P has zeros or negative vals"
            assert not np.any(np.isnan(self.T)), "Error: T contains NaN values"
            assert not np.any(np.isnan(self.P)), "Error: P contains NaN values"
            assert not np.any(np.isnan(self.rho)), "Error: rho contains NaN values"
            assert not np.any(np.isnan(self.e)), "Error: e contains NaN values"
            assert not np.any(np.isnan(self.u)), "Error: u contains NaN values"
            assert not np.any(np.isnan(self.w)), "Error: w contains NaN values"
            assert not np.any(np.isnan(self.rho_u)
                            ), "Error: rho_u contains NaN values"
            assert not np.any(np.isnan(self.rho_w)
                            ), "Error: rho_w contains NaN values"
        except AssertionError:
            print(f"{self.t=}")
            assert np.all(self.rho > 0), "Error: rho has zeros or negative vals"
            assert np.all(self.e > 0), "Error: e has zero or negative vals"
            assert np.all(self.T > 0), "Error: T has zeros or negative vals"
            assert np.all(self.P > 0), "Error: P has zeros or negative vals"
            assert not np.any(np.isnan(self.T)), "Error: T contains NaN values"
            assert not np.any(np.isnan(self.P)), "Error: P contains NaN values"
            assert not np.any(np.isnan(self.rho)), "Error: rho contains NaN values"
            assert not np.any(np.isnan(self.e)), "Error: e contains NaN values"
            assert not np.any(np.isnan(self.u)), "Error: u contains NaN values"
            assert not np.any(np.isnan(self.w)), "Error: w contains NaN values"
            assert not np.any(np.isnan(self.rho_u)
                            ), "Error: rho_u contains NaN values"
            assert not np.any(np.isnan(self.rho_w)
                            ), "Error: rho_w contains NaN values"

    def plot_initialise(self):
        """
        Plots initial.
        """
        self.initialise()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        im1 = axs[0, 0].imshow(self.P.T, aspect='auto')
        axs[0, 0].set_xlabel('y')
        axs[0, 0].set_ylabel('x')
        axs[0, 0].set_title('Pressure (P)')

        fig.colorbar(im1, ax=axs[0, 0])

        im2 = axs[0, 1].imshow(self.T.T, aspect='auto')
        axs[0, 1].set_xlabel('y')
        axs[0, 1].set_ylabel('x')
        axs[0, 1].set_title('Temperature (T)')
        fig.colorbar(im2, ax=axs[0, 1])

        im3 = axs[1, 0].imshow(self.e.T, aspect='auto')
        axs[1, 0].set_xlabel('y')
        axs[1, 0].set_ylabel('x')
        axs[1, 0].set_title('Internal Energy (e)')
        fig.colorbar(im3, ax=axs[1, 0])

        im4 = axs[1, 1].imshow(self.rho.T, aspect='auto')
        axs[1, 1].set_xlabel('y')
        axs[1, 1].set_ylabel('x')
        axs[1, 1].set_title('Density (rho)')
        fig.colorbar(im4, ax=axs[1, 1])

        fig.tight_layout()
        return fig

    def add_temperature_pertubation(self, temperature_peak, x0, y0, spread_x, spread_y):
        """
        Returns 2Dim Gaussian function grid with specified parameters.

        Args:
            temperature_peak (float): Temperature pertubation peak value.
            x0 (float): X-position for pertubation peak
            y0 (float): Y-position for pertubation peak
            spread_x (float): spread in x-direction
            spread_y (float): spread in y-direction

        Returns:
            (numpy.ndarray[float]): Gaussian temperature pertubation
        """
        x_ = np.linspace(0, 12e6, self.nx)
        y_ = np.linspace(0, 4e6, self.ny)
        y, x = np.meshgrid(y_, x_)
        self.temperature_pertubation += temperature_peak * \
            np.exp(- ((x - x0)**2/(2*spread_x**2) + (y-y0)**2/(2*spread_y**2)))


def plot_initial():
    """
    Show initialized grids without temperature pertubations.
    """
    instance = TwoDConvection()
    instance.initialise()
    fig = instance.plot_initialise()
    fig.savefig("initial_grid.pdf")


def simulate(sim_duration):
    """
    Run simulation for sim_duration seconds.
    
    Args:
        sim_duration (int): duration for which to run simulation.
    """
    assert isinstance(sim_duration, int), "non-integer sim_duration provided"
    assert sim_duration > 0, "zero or non-positive sim_duration provided"
    instance = TwoDConvection()
    instance.initialise()
    vis = FVis.FluidVisualiser()
    vis.save_data(sim_duration, instance.hydro_solver,
                  rho=instance.rho.T,
                  T=instance.T.T,
                  u=instance.u.T,
                  w=instance.w.T,
                  P=instance.P.T,
                  e=instance.e.T,
                  sim_fps=30,
                  folder='{sim_duration}s')
    #vis.animate_2D('T', save=True)
    vis.animate_2D('rho', save=True)
    #vis.animate_2D('e', save=True)
    #vis.animate_2D('P', save=True)


def sanity_check():
    """
    Run simulation for 60s with no pertubations. Benchmark for if code works or not
    is if the .mp4 generated causes too much problems.
    """
    instance = TwoDConvection()
    instance.initialise()
    vis = FVis.FluidVisualiser()
    vis.save_data(60, instance.hydro_solver,
                  rho=instance.rho.T,
                  T=instance.T.T,
                  u=instance.u.T,
                  w=instance.w.T,
                  P=instance.P.T,
                  e=instance.e.T,
                  sim_fps=30,
                  folder='sanity_check')
    vis.animate_2D('T', save=True, video_name="sanity_check")


def plot_initial_single_pertubation():
    """
    show initialized grid with temperature pertubation.
    """
    instance = TwoDConvection()
    instance.add_temperature_pertubation(
        temperature_peak=10000,
        x0=6e6,
        y0=2e6,
        spread_x=1e6,
        spread_y=1e6
    )
    instance.initialise()
    fig = instance.plot_initialise()
    fig.savefig("initial_grid_with_pertubation.pdf")


def animate_quantity(vis, quantity, sim_duration):
    """
    Generates 2D animation for quantity
    
    Args:
        vis (FVis.FluidVisualizer): Instance of FluidVisualizer
        quantity (str): Name of quantity to be simulated.
        sim_duration (float): Duration of simulation.
    """
    vis.animate_2D(quantity, save=True, video_name=f"{sim_duration}s_{quantity}_with_single_pertubation")


def simulate_single_pertubation(sim_duration):
    """
    Run simulation for sim_duration s with single pertubation.

    Args:
        sim_duration (int): duration for which to run simulation
    """
    assert isinstance(sim_duration, int), "non-integer sim_duration provided"
    assert sim_duration > 0, "zero or non-positive sim_duration provided"
    instance = TwoDConvection()
    instance.add_temperature_pertubation(
        temperature_peak=10000,
        x0=6e6,
        y0=2e6,
        spread_x=1e6,
        spread_y=1e6
    )
    instance.initialise()
    vis = FVis.FluidVisualiser()
    vis.save_data(sim_duration, instance.hydro_solver,
                  rho=instance.rho.T,
                  T=instance.T.T,
                  u=instance.u.T,
                  w=instance.w.T,
                  P=instance.P.T,
                  e=instance.e.T,
                  sim_fps=5,
                  folder=f'{sim_duration}s_single_pertubation')
    quantities = ["T", "rho", "P", "e"]
    for quantity in quantities:
        animate_quantity(vis, quantity, sim_duration)


if __name__ == '__main__':
    """
    uncomment to decide what the code does, refer to the docstrings.
    """
    # Run your code here
    #plot_initial()
    sanity_check()
    simulate_single_pertubation(200)
    #plot_initial_single_pertubation()
    #simulate()
