# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm, Andreas Kraemer

import torch
import numpy as np

from torch_geometric.data import Data, DataLoader

import os
import time
import warnings

from .utils import tqdm


class Simulation(object):
    """Simulate an artificial trajectory from a CGnet.

    If friction and masses are provided, Langevin dynamics are used (see also
    Notes). The scheme chosen is BAOA(F)B, where
        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
        F = force calculation (i.e., from the cgnet)

    Where we have chosen the following implementation so as to only calculate
    forces once per timestep:
        F = - grad( U(X_t) )
        [BB] V_(t+1) = V_t + dt * F/m
        [A] X_(t+1/2) = X_t + V * dt/2
        [O] V_(t+1) = V_(t+1) * vscale + dW_t * noisescale
        [A] X_(t+1) = X_(t+1/2) + V * dt/2

    Where:
        vscale = exp(-friction * dt)
        noisecale = sqrt(1 - vscale * vscale)

    The diffusion constant D can be back-calculated using the Einstein relation
        D = 1 / (beta * friction)

    Initial velocities are set to zero with Gaussian noise.

    If friction is None, this indicates Langevin dynamics with *infinite*
    friction, and the system evolves according to overdamped Langevin
    dynamics (i.e., Brownian dynamics) according to the following stochastic
    differential equation:

        dX_t = - grad( U( X_t ) ) * D * dt + sqrt( 2 * D * dt / beta ) * dW_t

    for coordinates X_t at time t, potential energy U, diffusion D,
    thermodynamic inverse temperature beta, time step dt, and stochastic Weiner
    process W. The choice of Langevin dynamics is made because CG systems
    possess no explicit solvent, and so Brownian-like collisions must be
    modeled indirectly using a stochastic term.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        Trained model used to generate simulation data
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Units are determined
        by the frame striding of the original training data simulation
    beta : float (default=1.0)
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
    friction : float (default=None)
        If None, overdamped Langevin dynamics are used (this is equivalent to
        "infinite" friction). If a float is given, Langevin dynamics are
        utilized with this (finite) friction value (sometimes referred to as
        gamma)
    masses : list of floats (default=None)
        Only relevant if friction is not None and (therefore) Langevin dynamics
        are used. In that case, masses must be a list of floats where the float
        at mass index i corresponds to the ith CG bead.
    diffusion : float (default=1.0)
        The constant diffusion parameter D for overdamped Langevin dynamics
        *only*. By default, the diffusion is set to unity and is absorbed into
        the dt argument. However, users may specify separate diffusion and dt
        parameters in the case that they have some estimate of the CG
        diffusion
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.

    Notes
    -----
    Long simulation lengths may take a significant amount of time.

    Langevin dynamics simulation velocities are currently initialized from
    zero. You should probably remove the beginning part of the simulation.

    Any output files will not be overwritten; the presence of existing files
    will cause an error to be raised.

    Langevin dynamics code based on:
    https://github.com/choderalab/openmmtools/blob/master/openmmtools/integrators.py

    Checks are broken into two methods: one (_input_model_checks()) that deals
    with checking components of the input models and their architectures and another
    (_input_option_checks()) that deals with checking options pertaining to the
    simulation, such as logging or saving/exporting options. This is done for the
    following reasons:

		1) If cgnet.network.Simluation is subclassed for multiple or specific
        model types, the _input_model_checks() method can be overridden without
        repeating a lot of code. As an example, see
        cgnet.network.MultiModelSimulation, which uses all of the same
        simulation options as cgnet.network.Simulation, but overides
        _input_model_checks() in order to perform the same model checks as
        cgnet.network.Simulation but for more than one input model.

		2) If cgnet.network.Simulation is subclassed for different simulation
        schemes with possibly different/additional simulation
        parameters/options, the _input_option_checks() method can be overriden
        without repeating code related to model checks. For example, one might
        need a simulation scheme that includes an external force that is
        decoupled from the forces predicted by the model.

    """

    def __init__(self, model, initial_coordinates, embeddings, dt=5e-4,
                 beta=1.0, friction=None, masses=None, diffusion=1.0,
                 save_forces=False, save_potential=False, length=100,
                 save_interval=10, random_seed=None,
                 device=torch.device('cpu'),
                 export_interval=None, log_interval=None,
                 log_type='write', filename=None, batch_size=10):

        self.initial_coordinates = initial_coordinates
        self.embeddings = embeddings

        model.eval()
        self.model = model

        self.friction = friction
        self.masses = masses

        self.n_sims = self.initial_coordinates.shape[0]
        self.n_beads = self.initial_coordinates.shape[1]
        self.n_dims = self.initial_coordinates.shape[2]

        self.save_forces = save_forces
        self.save_potential = save_potential
        self.length = length
        self.save_interval = save_interval

        self.dt = dt
        self.diffusion = diffusion
        self.beta = beta

        self.device = device
        self.export_interval = export_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        if log_type not in ['print', 'write']:
            raise ValueError(
                "log_type can be either 'print' or 'write'"
            )
        self.log_type = log_type
        self.filename = filename

        # Here, we check to make sure input options for the simulation
        # are acceptable. Note that these checks are separated from
        # the input model checks in _input_model_checks() for ease in
        # subclassing. See class notes for more information.
        self._input_option_checks()

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed

        self._simulated = False


    def _input_option_checks(self):
        """Method to catch any problems before starting a simulation:
        - Make sure the save_interval evenly divides the simulation length
        - Checks shapes of starting coordinates and embeddings
        - Ensures masses are provided if friction is not None
        - Warns if diffusion is specified but won't be used
        - Checks compatibility of arguments to save and log
        - Sets up saving parameters for numpy and log files, if relevant

        Notes
        -----
        This method is meant to check the acceptability/compatibility of
        options pertaining to simulation details, saving/output settings,
        and/or log settings. For checks related to model structures/architectures
        and input compatibilities (such as using embeddings in models
        with SchnetFeatures), see the _input_model_checks() method.

        """

        # if there are embeddings, make sure their shape is correct
        if self.embeddings is not None:
            if len(self.embeddings.shape) != 2:
                raise ValueError('embeddings shape must be [frames, beads]')

            if self.initial_coordinates.shape[:2] != self.embeddings.shape:
                raise ValueError('initial_coordinates and embeddings '
                                 'must have the same first two dimensions')

        # make sure save interval is a factor of total length
        if self.length % self.save_interval != 0:
            raise ValueError(
                'The save_interval must be a factor of the simulation length'
            )


        # make sure initial coordinates are in the proper format
        if len(self.initial_coordinates.shape) != 3:
            raise ValueError(
                'initial_coordinates shape must be [frames, beads, dimensions]'
            )

        # set up initial coordinates
        if type(self.initial_coordinates) is not torch.Tensor:
            initial_coordinates = torch.tensor(self.initial_coordinates)

        self._initial_x = self.initial_coordinates.detach().requires_grad_(
            True).to(self.device)

        # set up simulation parameters
        if self.friction is not None:  # langevin
            if self.masses is None:
                raise RuntimeError(
                    'if friction is not None, masses must be given'
                )
            if len(self.masses) != self.initial_coordinates.shape[1]:
                raise ValueError(
                    'mass list length must be number of CG beads'
                )
            self.masses = torch.tensor(self.masses, dtype=torch.float32
                                       ).to(self.device)

            self.vscale = np.exp(-self.dt * self.friction)
            self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

            self.kinetic_energies = []

            if self.diffusion != 1:
                warnings.warn(
                    "Diffusion other than 1. was provided, but since friction "
                    "and masses were given, Langevin dynamics will be used "
                    "which do not incorporate this diffusion parameter"
                )

        else:  # Brownian dynamics
            self._dtau = self.diffusion * self.dt

            self.kinetic_energies = None

            if self.masses is not None:
                warnings.warn(
                    "Masses were provided, but will not be used since "
                    "friction is None (i.e., infinte)."
                )

        # everything below has to do with saving logs/numpys

        # check whether a directory is specified if any saving is done
        if self.export_interval is not None and self.filename is None:
            raise RuntimeError(
                "Must specify filename if export_interval isn't None"
            )
        if self.log_interval is not None:
            if self.log_type == 'write' and self.filename is None:
                raise RuntimeError(
                    "Must specify filename if log_interval isn't None and log_type=='write'"
                )

        # saving numpys
        if self.export_interval is not None:
            if self.length // self.export_interval >= 1000:
                raise ValueError(
                    "Simulation saving is not implemented if more than 1000 files will be generated"
                )

            if os.path.isfile("{}_coords_000.npy".format(self.filename)):
                raise ValueError(
                    "{} already exists; choose a different filename.".format(
                        "{}_coords_000.npy".format(self.filename))
                )

            if self.export_interval is not None:
                if self.export_interval % self.save_interval != 0:
                    raise ValueError(
                        "Numpy saving must occur at a multiple of save_interval"
                    )
                self._npy_file_index = 0
                self._npy_starting_index = 0

        # logging
        if self.log_interval is not None:
            if self.log_interval % self.save_interval != 0:
                raise ValueError(
                    "Logging must occur at a multiple of save_interval"
                )

            if self.log_type == 'write':
                self._log_file = self.filename + '_log.txt'

                if os.path.isfile(self._log_file):
                    raise ValueError(
                        "{} already exists; choose a different filename.".format(
                            self._log_file)
                    )

    def _set_up_simulation(self, overwrite):
        """Method to initialize helpful objects for simulation later
        """
        if self._simulated and not overwrite:
            raise RuntimeError('Simulation results are already populated. '
                               'To rerun, set overwrite=True.')

        self._save_size = int(self.length/self.save_interval)

        self.simulated_coords = torch.zeros((self._save_size, self.n_sims, self.n_beads,
                                             self.n_dims))
        if self.save_forces:
            self.simulated_forces = torch.zeros((self._save_size, self.n_sims,
                                                 self.n_beads, self.n_dims))
        else:
            self.simulated_forces = None

        # the if saved, the simulated potential shape is identified in the first
        # simulation time point in self._save_timepoint
        self.simulated_potential = None

        if self.friction is not None:
            self.kinetic_energies = torch.zeros((self._save_size, self.n_sims))

    def _timestep(self, x_old, v_old, forces):
        """Shell method for routing to either Langevin or overdamped Langevin
        dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None or torch.Tensor
            None if overdamped Langevin; velocities before propagation
            otherwise
        forces: torch.Tensor
            forces at x_old
        """
        if self.friction is None:
            assert v_old is None
            return self._overdamped_timestep(x_old, v_old, forces)
        else:
            return self._langevin_timestep(x_old, v_old, forces)

    def _langevin_timestep(self, x_old, v_old, forces):
        """Heavy lifter for Langevin dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : torch.Tensor
            velocities before propagation
        forces: torch.Tensor
            forces at x_old
        """

        # BB (velocity update); uses whole timestep
        v_new = v_old + self.dt * forces / self.masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt / 2.

        # O (noise)
        noise = torch.sqrt(1. / self.beta / self.masses[:, None])
        noise = noise * torch.randn(size=x_new.size(),
                                    generator=self.rng).to(self.device)
        v_new = v_new * self.vscale
        v_new = v_new + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt / 2.

        return x_new, v_new

    def _overdamped_timestep(self, x_old, v_old, forces):
        """Heavy lifter for overdamped Langevin (Brownian) dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None
            Placeholder
        forces: torch.Tensor
            forces at x_old
        """
        noise = torch.randn(size=x_old.size(),
                            generator=self.rng).to(self.device)
        x_new = (x_old.detach() + forces*self._dtau +
                 np.sqrt(2*self._dtau/self.beta)*noise)
        return x_new, None

    def _save_timepoint(self, x_new, v_new, forces, potential, t):
        """Utilities to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy

        Parameters
        ----------
        x_new : torch.Tensor
            current coordinates
        v_new : None or torch.Tensor
            current velocities, if Langevin dynamics are used
        forces: torch.Tensor
            current forces
        potential : torch.Tensor
            current potential
        t : int
            Timestep iteration index
        """
        save_ind = t // self.save_interval

        self.simulated_coords[save_ind, :, :] = x_new
        if self.save_forces:
            self.simulated_forces[save_ind, :, :] = forces

        if self.save_potential:
            # The potential will look different for different network
            # structures, so determine its dimensionality at the first
            # timepoint (as opposed to in self._set_up_simulation)
            if self.simulated_potential is None:
                assert potential.shape[0] == self.n_sims
                potential_dims = ([self._save_size, self.n_sims] +
                                  [potential.shape[j]
                                   for j in range(1,
                                                  len(potential.shape))])
                self.simulated_potential = torch.zeros((potential_dims))

            self.simulated_potential[t//self.save_interval] = potential

        if v_new is not None:
            kes = 0.5 * torch.sum(torch.sum(self.masses[:, None]*v_new**2,
                                            dim=2), dim=1)
            self.kinetic_energies[save_ind, :] = kes

    def _log_progress(self, iter_):
        """Utility to print log statement or write it to an text file"""
        printstring = '{}/{} time points saved ({})'.format(
            iter_, self.length // self.save_interval, time.asctime())

        if self.log_type == 'print':
            print(printstring)

        elif self.log_type == 'write':
            printstring += '\n'
            file = open(self._log_file, 'a')
            file.write(printstring)
            file.close()

    def _get_numpy_count(self):
        """Returns a string 000-999 for appending to numpy file outputs"""
        if self._npy_file_index < 10:
            return '00{}'.format(self._npy_file_index)
        elif self._npy_file_index < 100:
            return '0{}'.format(self._npy_file_index)
        else:
            return '{}'.format(self._npy_file_index)

    def _save_numpy(self, iter_):
        """Utility to save numpy arrays"""
        key = self._get_numpy_count()

        coords_to_export = self.simulated_coords[self._npy_starting_index:iter_]
        coords_to_export = self._swap_and_export(coords_to_export)
        np.save("{}_coords_{}.npy".format(
            self.filename, key), coords_to_export)

        if self.save_forces:
            forces_to_export = self.simulated_forces[self._npy_starting_index:iter_]
            forces_to_export = self._swap_and_export(forces_to_export)
            np.save("{}_forces_{}.npy".format(
                self.filename, key), forces_to_export)

        if self.save_potential:
            potentials_to_export = self.simulated_potential[self._npy_starting_index:iter_]
            potentials_to_export = self._swap_and_export(potentials_to_export)
            np.save("{}_potential_{}.npy".format(
                self.filename, key), potentials_to_export)

        if self.friction is not None:
            kinetic_energies_to_export = self.kinetic_energies[self._npy_starting_index:iter_]
            kinetic_energies_to_export = self._swap_and_export(
                kinetic_energies_to_export)
            np.save("{}_kineticenergy_{}.npy".format(self.filename, key),
                    kinetic_energies_to_export)

        self._npy_starting_index = iter_
        self._npy_file_index += 1

    def _swap_and_export(self, data, axis1=0, axis2=1):
        """Helper method to exchange the zeroth and first axes of tensors that
        will be output or exported as numpy arrays

        Parameters
        ----------
        data : torch.Tensor
            Tensor to perform the axis swtich upon. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        axis1 : int (default=0)
            Zero-based index of the first axis to swap
        axis2 : int (default=1)
            Zero-based index of the second axis to swap

        Returns
        -------
        swapped_data : torch.Tensor
            Axes-swapped tensor. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        """
        axes = list(range(len(data.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = data.permute(*axes)
        return swapped_data.cpu().detach().numpy()


    def calculate_potential_and_forces(self, x_old):
        """Method to calculated predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates from the previous timestep

        Returns
        -------
        potential : torch.Tensor
            scalar potential predicted by the model
        forces : torch.Tensor
            vector forces predicted by the model

        Notes
        -----
        This method has been isolated from the main simulation update scheme
        for ease in overriding in a subclass that may preprocess or modify the
        forces or potential returned from a model. For example, see
        cgnet.network.MultiModelSimulation, where this method is overridden
        in order to average forces and potentials over more than one model.
        """
        n_frames, n_beads, _ = x_old.shape
        # batch = torch.zeros((n_frames*n_beads,), device=x_old.device, dtype=torch.int64)
        # pos = x_old.reshape((-1, 3))
        # z = self.embeddings.reshape((-1))
        # idx = torch.zeros((n_frames,), device=x_old.device, dtype=torch.long)
        # n_atoms = n_beads * torch.ones((n_frames,), device=x_old.device, dtype=torch.long)
        # ii = 0
        data_list = []
        for iframe in range(n_frames):
            # batch[ii:ii+n_beads] = iframe
            # idx[iframe] = iframe
            # ii += n_beads

            data_list.append(Data(z=self.embeddings[iframe], pos=x_old[iframe], idx=iframe, n_atoms=n_beads).to(device=x_old.device))

        potential = torch.zeros(n_frames, device=x_old.device, dtype=x_old.dtype)
        forces = torch.zeros_like(x_old)
        ii = 0
        dataloader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        for data in tqdm(dataloader, desc='Batch', leave=False):
            batch_size = data.n_atoms.shape[0]
            # data.to(device=x_old.device)
            pp, ff = self.model(data)
            potential[ii:ii+batch_size] = pp.flatten()
            forces[ii:ii+batch_size] = ff.view((batch_size, n_beads, 3))
            ii += batch_size
        return potential, forces.reshape((-1, n_beads, 3))


    def simulate(self, overwrite=False):
        """Generates independent simulations.

        Parameters
        ----------
        overwrite : Bool (default=False)
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords : np.ndarray
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
            calculated for each frame in the simulation at the
            save_interval
        simulated_potential : np.ndarray or None
            Shape [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation at the save_interval
        kinetic_energies : np.ndarray or None
            If friction is not None, stores the kinetic energy calculated
            for each frame in the simulation at the save_interval

        """
        self._set_up_simulation(overwrite)

        if self.log_interval is not None:
            printstring = "Generating {} simulations of length {} saved at {}-step intervals ({})".format(
                self.n_sims, self.length, self.save_interval, time.asctime())
            if self.log_type == 'print':
                print(printstring)

            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        x_old = self._initial_x

        # for each simulation step
        if self.friction is None:
            v_old = None
        else:
            # initialize velocities at zero
            v_old = torch.tensor(np.zeros(x_old.shape),
                                 dtype=torch.float32).to(self.device)
            # v_old = v_old + torch.randn(size=v_old.size(),
            #                             generator=self.rng).to(self.device)

        for t in tqdm(range(self.length), desc='Simulation timestep'):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(x_old)
            potential = potential.detach()
            forces = forces.detach()

            # step forward in time
            x_new, v_new = self._timestep(x_old, v_old, forces)

            # save to arrays if relevant
            if (t+1) % self.save_interval == 0:
                self._save_timepoint(x_new, v_new, forces, potential, t)

                # save numpys if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self._save_numpy((t+1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self._log_progress((t+1) // self.save_interval)

            # prepare for next timestep
            x_old = x_new.detach().requires_grad_(True).to(self.device)
            v_old = v_new

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t+1) % self.export_interval > 0:
                self._save_numpy(t+1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            printstring = 'Done simulating ({})'.format(time.asctime())
            if self.log_type == 'print':
                print(printstring)
            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(
            self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(
                self.simulated_forces)

        if self.save_potential:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential)

        if self.friction is not None:
            self.kinetic_energies = self._swap_and_export(
                self.kinetic_energies)

        self._simulated = True

        return self.simulated_coords


class MultiModelSimulation(Simulation):
    """Simulation that integrates CG coordinates forward in time using
    the average forces predicted from more than one CGnet model. For
    thoeretical details on (overdamped) Langevin integration schemes,
    see help(cgnet.network.Simulation)

    Parameters
    ----------
    models: list of cgnet.network.CGNet() instances
        The list of models from which predicted forces will be averaged over.
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Units are determined
        by the frame striding of the original training data simulation
    beta : float (default=1.0)
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
    friction : float (default=None)
        If None, overdamped Langevin dynamics are used (this is equivalent to
        "infinite" friction). If a float is given, Langevin dynamics are
        utilized with this (finite) friction value (sometimes referred to as
        gamma)
    masses : list of floats (default=None)
        Only relevant if friction is not None and (therefore) Langevin dynamics
        are used. In that case, masses must be a list of floats where the float
        at mass index i corresponds to the ith CG bead.
    diffusion : float (default=1.0)
        The constant diffusion parameter D for overdamped Langevin dynamics
        *only*. By default, the diffusion is set to unity and is absorbed into
        the dt argument. However, users may specify separate diffusion and dt
        parameters in the case that they have some estimate of the CG
        diffusion
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.

    Notes
    -----
    This class inherits from cgnet.network.Simulation. The same checks for
    model mode ('train' vs. 'eval'), embedding/model compatibility for each input
    model. The checks concerning simulation options are handled through superclassed
    __init__(). For a full list of these checks, see
    cgnet.network.Simulation._input_option_checks
    """

    def __init__(self, models, initial_coordinates, **kwargs):

        super(MultiModelSimulation, self).__init__(models, initial_coordinates,
                                                   **kwargs)

        self._input_model_list_check(models)
        self.models = models

    def _input_model_list_check(self, models):
        for idx, individual_model in enumerate(models):
            self._input_model_checks(individual_model, idx=idx)

    def calculate_potential_and_forces(self, x_old):
        """Method to calculate average predicted potentials and forces from
        forwarding the x_old through each of the models in self.models

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates from the previous timestep

        Returns
        -------
        potential : torch.Tensor
            scalar potential averaged over all the models in self.models
        forces : torch.Tensor
            vector forces averaged over all the models in self.models
        """
        potential_list = []
        forces_list = []
        for model in self.models:
            potential, forces = model(x_old, self.embeddings)
            potential_list.append(potential)
            forces_list.append(forces)
        mean_potential =  sum(potential_list) / len(potential_list)
        mean_forces =  sum(forces_list) / len(forces_list)

        # make sure the mean did not alter the shapes of potential/forces
        # using the last potential/force
        assert mean_potential.size() == potential.size()
        assert mean_forces.size() == forces.size()

        return mean_potential, mean_forces
