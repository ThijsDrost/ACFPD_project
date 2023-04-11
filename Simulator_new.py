import math
import warnings
import time

import numpy as np
import numba
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, t, steps, particles, *, masses=None, save_every=1, verlet_type='basic', cutoff=None,
                 boundary_condition=False, gaussian_mass=False, seed=None, temperature=None, thermostat=False,
                 thermostat_time=None, equilibration_steps=0):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        if cutoff is None:
            self.cutoff = np.inf
        else:
            self.cutoff = cutoff
        self.cutoff2 = self.cutoff ** 2

        if verlet_type not in ['basic', 'velocity', 'euler']:
            raise ValueError(rf"`verlet_type` should be either 'basic', 'velocity' or 'euler' not {verlet_type}")
        self.verlet_type = verlet_type

        if masses is None:
            # The shape of (particles, 1) is for making the multiplication with the velocity vector easier
            if gaussian_mass:
                self.particle_mass = np.random.normal(1, 0.1, (particles, 1))
                self.particle_mass[self.particle_mass < 0] *= -1
            else:
                self.particle_mass = np.ones((particles, 1))
        else:
            try:
                self.particle_mass = np.array(masses).reshape((particles, 1))
            except ValueError as e:
                raise ValueError(
                    'Something is wrong with the given particle masses, NumPy gave the following error:\n' +
                    str(e))

        self.steps = steps
        self.particles = particles
        self.boundary_condition = boundary_condition
        self.dt = t / (steps - 1)
        self.save_every = save_every
        self.box_size = 1.2 * (math.ceil(particles ** (1 / 3)))
        self._flat_mass = self.particle_mass.flatten()

        self.thermostat = thermostat
        self.thermostat_steps = None
        if thermostat:
            if thermostat_time is None:
                warnings.warn('No thermostat time constant is given, time constant is set to ten timesteps')
                self.thermostat_steps = 10
            else:
                self.thermostat_steps = thermostat_time / self.dt

            if temperature is None:
                raise ValueError('When using the thermostat `temperature` needs to be given')

        if temperature is None:
            self._temperatures = (0 for _ in range(steps))
        elif isinstance(temperature, (tuple, list, np.ndarray)):
            if equilibration_steps > self.steps:
                warnings.warn('Equilibration time is longer than simulation time, simulation will remain at initial'
                              ' temperature')

            self._temperatures = self._temperature_generator(steps, temperature, equilibration_steps)
        else:
            self._temperatures = (temperature for _ in range(steps))
        self._input_temperature = temperature
        self.equilibration_steps = equilibration_steps

        save_num = (steps // save_every)
        save_num += 1 if steps % save_every != 0 else 0

        self.times = np.linspace(0, t, steps)
        self.kinetic = np.zeros(save_num)
        self.potential = np.zeros(save_num)
        self.temperature_arr = np.zeros(save_num)
        self.total_particle_velocities = np.zeros((save_num, particles, 3))
        self.total_particle_locations = np.zeros((save_num, particles, 3))
        self.particle_acceleration = np.zeros((particles, 3))
        self.boundary_transitions = np.zeros((particles, 3))

        self.temperature = next(self._temperatures)
        self.particle_velocities = self._initial_vel(particles, self.particle_mass, self.temperature)
        self.particle_locations = self._initial_pos(particles)
        self.total_particle_locations[0] = self.particle_locations
        self.total_particle_velocities[0] = self.particle_velocities
        boxsize = self.box_size if boundary_condition else -1
        self.kinetic[0], self.potential[0] = self.calc_energies(self.particle_locations, self.particle_velocities,
                                                                self._flat_mass, self.particles, boxsize, self.cutoff2)
        self.last_particle_locations = self.particle_locations
        self.runtime = None

    def execute(self):
        if self.boundary_condition:
            box_size = self.box_size
        else:
            box_size = -1

        if self.verlet_type == 'basic':
            def stepper():
                self._do_step_verlet(box_size)
        elif self.verlet_type == 'velocity':
            def stepper():
                self._do_step_velocity_verlet(box_size)
        elif self.verlet_type == 'euler':
            def stepper():
                self._do_step_euler(box_size)
        else:
            raise ValueError("`verlet_type` not recognized")

        # First step
        forces = self.calc_forces(self.particle_locations, self.particles, box_size, self.cutoff2)
        acc = forces / self.particle_mass
        self.particle_locations = self.last_particle_locations + self.particle_velocities * self.dt \
                                  + 0.5 * acc * self.dt * self.dt
        self._save_data(1)

        # Further steps
        start_time = time.time()
        for index, time_step in enumerate(self.times[2:], start=2):
            stepper()
            self.temperature = next(self._temperatures)
            if self.thermostat:
                self._Berensen_thermostat()
            if (index % self.save_every) == 0:
                self._save_data(index // self.save_every)
            if (index % int(self.steps / 100)) == 0:
                print(f'\r{index // int(self.steps / 100)}% done', end='')
        self.runtime = time.time() - start_time
        print(f'\r100% done', end='')
        print('')

    def _Berensen_thermostat(self):
        temperature = self._temperature()
        lambda_factor = 1 + (1 / self.thermostat_steps) * (self.temperature / temperature - 1)
        self.particle_velocities = np.sqrt(lambda_factor) * self.particle_velocities

    @staticmethod
    def _temperature_generator(steps, temperatures, equilibration_steps):
        for step in range(steps):
            if step < equilibration_steps:
                yield temperatures[0]
            else:
                relative_step = step - equilibration_steps
                dT = temperatures[1]-temperatures[0]
                ramp_steps = steps-1-equilibration_steps
                yield temperatures[0] + relative_step*dT/ramp_steps

    def _do_step_verlet(self, box_size):
        forces = self.calc_forces(self.particle_locations, self.particles, box_size, self.cutoff2)
        self.particle_acceleration = forces / self.particle_mass
        new_particle_locations = 2 * self.particle_locations - self.last_particle_locations \
                                 + self.particle_acceleration * self.dt * self.dt
        self.particle_velocities = (new_particle_locations - self.particle_locations) / self.dt

        if box_size > 0:
            mask = new_particle_locations > box_size
            self.particle_locations[mask] = self.particle_locations[mask] - box_size
            new_particle_locations[mask] = new_particle_locations[mask] - box_size
            self.boundary_transitions[mask] += 1

            mask = new_particle_locations < 0
            self.particle_locations[mask] = self.particle_locations[mask] + box_size
            new_particle_locations[mask] = new_particle_locations[mask] + box_size
            self.boundary_transitions[mask] -= 1

        self.last_particle_locations = self.particle_locations
        self.particle_locations = new_particle_locations

    def _do_step_velocity_verlet(self, box_size):
        self.last_particle_locations = self.particle_locations
        self.particle_velocities += 0.5 * self.particle_acceleration * self.dt
        self.particle_locations = self.particle_locations + self.particle_velocities * self.dt

        forces = self.calc_forces(self.particle_locations, self.particles, box_size, self.cutoff2)
        self.particle_acceleration = forces / self.particle_mass
        self.particle_velocities += 0.5 * self.particle_acceleration * self.dt

        if box_size > 0:
            self._boundary_condition(box_size)

    def _do_step_euler(self, box_size):
        self.last_particle_locations = self.particle_locations
        forces = self.calc_forces(self.particle_locations, self.particles, box_size, self.cutoff2)
        self.particle_acceleration = forces / self.particle_mass
        self.particle_velocities += self.particle_acceleration * self.dt
        self.particle_locations = self.particle_locations + self.particle_velocities * self.dt
        if box_size > 0:
            self._boundary_condition(box_size)

    def _boundary_condition(self, box_size):
        mask = self.particle_locations > box_size
        self.particle_locations[mask] = self.particle_locations[mask] - box_size
        self.boundary_transitions[mask] += 1

        mask = self.particle_locations < 0
        self.particle_locations[mask] = self.particle_locations[mask] + box_size
        self.boundary_transitions[mask] -= 1

    def _temperature(self):
        return np.average(self._flat_mass * np.average(self.particle_velocities ** 2, axis=1))
    def _save_data(self, index):
        self.total_particle_locations[index] = self.particle_locations
        self.total_particle_velocities[index] = self.particle_velocities
        boxsize = self.box_size if self.boundary_condition else -1
        self.kinetic[index], self.potential[index] = self.calc_energies(self.particle_locations,
                                                                        self.particle_velocities,
                                                                        self._flat_mass, self.particles, boxsize,
                                                                        self.cutoff2)
        self.temperature_arr[index] = self._temperature()

    def plot(self, start=0, stop=-1, every=1):
        ax = plt.figure().add_subplot(projection='3d')
        for i in range(len(self.total_particle_locations[0])):
            ax.plot(self.total_particle_locations[start:stop:every, i, 0],
                    self.total_particle_locations[start:stop:every, i, 1],
                    self.total_particle_locations[start:stop:every, i, 2])

    def plot_energy(self, start=0, stop=-1, every=1):
        plt.figure()
        plt.plot(self.times[::self.save_every][start:stop:every], self.kinetic[start:stop:every], label='kinetic')
        plt.plot(self.times[::self.save_every][start:stop:every], self.potential[start:stop:every], label='potential')
        plt.plot(self.times[::self.save_every][start:stop:every], self.total_energy[start:stop:every], label='total')
        plt.ylabel('Energy')  # TODO: unit?
        plt.xlabel('Time')  # TODO: unit?
        plt.legend()

    def save(self, loc):
        thermostat_time = None if self.thermostat_steps is None else self.thermostat_steps*self.dt
        np.savez_compressed(loc, time=self.save_times, loc=self.total_particle_locations, masses=self.particle_mass,
                            velocity=self.total_particle_velocities, save_every=self.save_every, seed=[self.seed],
                            boundary_transitions=self.boundary_transitions, boundary=[self.boundary_condition],
                            boxsize=[self.box_size], verlet_type=[self.verlet_type], runtime=[self.runtime],
                            thermostat=[self.thermostat], temperature=[self._input_temperature], cutoff=[self.cutoff],
                            thermostat_time=[thermostat_time], equilibration_steps=[self.equilibration_steps])

    @property
    def total_energy(self):
        return self.potential + self.kinetic

    @property
    def save_times(self):
        return self.times[::self.save_every]

    @staticmethod
    def read(loc):
        read = np.load(loc, allow_pickle=True)
        times = read['time']
        total_particle_locations = read['loc']
        particle_mass = read['masses']
        total_particle_velocities = read['velocity']
        save_every = read['save_every']
        boundary_condition = read['boundary'][0]
        box_size = read['boxsize'][0]
        verlet_type = read['verlet_type'][0]
        seed = read['seed'][0]
        cutoff = read['cutoff'][0]
        boundary_transitions = read['boundary_transitions']
        thermostat = read['thermostat'][0]
        temperature = read['temperature'][0]
        thermostat_time = read['thermostat_time'][0]
        equilibration_steps = read['equilibration_steps'][0]
        runtime = read['runtime'][0]

        simulation = Simulation(t=times[-1], steps=len(times) * save_every, particles=total_particle_locations.shape[1],
                                masses=particle_mass, save_every=save_every, boundary_condition=boundary_condition,
                                verlet_type=verlet_type, seed=seed, cutoff=cutoff, thermostat=thermostat,
                                temperature=temperature, thermostat_time=thermostat_time,
                                equilibration_steps=equilibration_steps)

        simulation.box_size = box_size
        simulation.total_particle_locations = total_particle_locations
        simulation.total_particle_velocities = total_particle_velocities
        simulation.boundary_transitions = boundary_transitions
        simulation.runtime = runtime

        boxsize = box_size if boundary_condition else -1
        for index in range(len(times)):
            simulation.kinetic[index], simulation.potential[index] = Simulation.calc_energies(
                total_particle_locations[index], total_particle_velocities[index], simulation._flat_mass,
                simulation.particles, boxsize, cutoff ** 2)
        return simulation

    @staticmethod
    def load(loc):
        return Simulation.read(loc)

    @staticmethod
    def run(t, steps, particles, **kwargs):
        simulation = Simulation(t, steps, particles, **kwargs)
        simulation.execute()
        return simulation

    @staticmethod
    @numba.njit
    def calc_energies(particle_locations, particle_velocity, masses, particle_num, boxsize, cutoff2):
        kinetic = np.sum(0.5 * masses * np.sum(particle_velocity ** 2, axis=1))
        potential = 0
        inv_box_size = 1 / boxsize
        for i in range(particle_num):
            for j in range(i + 1, particle_num):
                xij = particle_locations[i, 0] - particle_locations[j, 0]
                yij = particle_locations[i, 1] - particle_locations[j, 1]
                zij = particle_locations[i, 2] - particle_locations[j, 2]

                if boxsize > 0:
                    if xij < -0.5 * boxsize:
                        xij += boxsize
                    elif xij > 0.5 * boxsize:
                        xij -= boxsize
                    if yij < -0.5 * boxsize:
                        yij += boxsize
                    elif yij > 0.5 * boxsize:
                        yij -= boxsize
                    if zij < -0.5 * boxsize:
                        zij += boxsize
                    elif zij > 0.5 * boxsize:
                        zij -= boxsize
                    # xij = xij - boxsize * round(xij * inv_box_size)
                    # yij = yij - boxsize * round(yij * inv_box_size)
                    # zij = zij - boxsize * round(zij * inv_box_size)

                rij2 = xij * xij + yij * yij + zij * zij
                if cutoff2 < rij2:
                    continue
                potential += 4.0 * (1.0 / (rij2 ** 6) - 1.0 / (rij2 ** 3))
        return kinetic, potential

    @staticmethod
    def _initial_pos(particles):
        particle_locations = np.zeros((particles, 3), dtype=float)
        particles_per_dimension = math.ceil(particles ** (1 / 3))
        value = 1.2 * np.linspace(0, particles_per_dimension, particles_per_dimension, endpoint=False)
        x, y, z = np.meshgrid(value, value, value, indexing='ij')
        particle_locations[:, 0] = x.flatten()[:particles]
        particle_locations[:, 1] = y.flatten()[:particles]
        particle_locations[:, 2] = z.flatten()[:particles]
        return particle_locations

    @staticmethod
    def _initial_vel(particles, particle_mass, temperature=0):
        initial_particle_velocities = np.random.uniform(-1, 1, (particles, 3))
        velocities = np.sum(particle_mass * initial_particle_velocities, axis=0)
        initial_particle_velocities -= velocities / np.sum(particle_mass)
        if temperature > 0:
            init_temp = np.average(particle_mass * np.average(initial_particle_velocities ** 2, axis=1))
            change_factor = temperature / init_temp
            initial_particle_velocities *= np.sqrt(change_factor)
        if np.any(np.sum(particle_mass * initial_particle_velocities, axis=0) > 1e-5):
            raise ValueError('Centre of mass moves')
        return initial_particle_velocities

    @staticmethod
    @numba.njit()
    def calc_forces(particle_locations, particles, boxsize, cutoff2):
        forces = np.zeros((particles, 3), dtype=float)
        for i in range(particles):
            for j in range(i + 1, particles):
                xij = particle_locations[i, 0] - particle_locations[j, 0]
                yij = particle_locations[i, 1] - particle_locations[j, 1]
                zij = particle_locations[i, 2] - particle_locations[j, 2]

                if boxsize > 0:
                    if xij < -0.5 * boxsize:
                        xij += boxsize
                    elif xij > 0.5 * boxsize:
                        xij -= boxsize
                    if yij < -0.5 * boxsize:
                        yij += boxsize
                    elif yij > 0.5 * boxsize:
                        yij -= boxsize
                    if zij < -0.5 * boxsize:
                        zij += boxsize
                    elif zij > 0.5 * boxsize:
                        zij -= boxsize
                    # xij = xij - boxsize * round(xij * inv_box_size)
                    # yij = yij - boxsize * round(yij * inv_box_size)
                    # zij = zij - boxsize * round(zij * inv_box_size)

                rij2 = xij * xij + yij * yij + zij * zij

                if cutoff2 < rij2:
                    continue

                factor = 4.0 * (12.0 / (rij2 ** 7) - 6.0 / (rij2 ** 4))

                Fijx = factor * xij
                Fijy = factor * yij
                Fijz = factor * zij

                forces[i, 0] += Fijx
                forces[i, 1] += Fijy
                forces[i, 2] += Fijz
                forces[j, 0] -= Fijx
                forces[j, 1] -= Fijy
                forces[j, 2] -= Fijz
        return forces


if __name__ == '__main__':
    np.random.seed(123456)
    sim = Simulation.run(1e1, int(1e5), 2, save_every=100, verlet_type='basic', boundary_condition=False)
    sim.plot()
    plt.show()
    sim.plot_energy()
    plt.show()

    plt.figure()
    plt.plot(sim.total_energy[1:])
    plt.show()
