# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang, Simon Olsson

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """Physical prior to the CGNet model class

    Parameters
    ----------
    geom_feature : nn.Module() instance
        geometry features.
    priors : list of nn.Module() instances (default=None)
        list of prior layers that provide energy contributions external to
        the hidden architecture of the CGnet.

    """

    def __init__(self, geom_feature, priors, n_beads, beta=1/300):
        super(BaselineModel, self).__init__()
        self.geom_feature = geom_feature
        self.beta = beta
        self.priors = nn.Sequential(*priors)
        harm_idx = 1
        repul_idx = 1
        for layer in self.priors:
            if isinstance(layer, HarmonicLayer):
                self.register_buffer('harmonic_params_{}'.format(harm_idx),
                                     layer.harmonic_parameters)
                harm_idx += 1
            if isinstance(layer, RepulsionLayer):
                self.register_buffer('repulsion_params_{}'.format(repul_idx),
                                     layer.repulsion_parameters)
                repul_idx += 1

        self.n_beads = n_beads

    def forward(self, pos, n_frames):
        coordinates = pos.reshape((-1, self.n_beads, 3))
        geometry_features = self.geom_feature(coordinates)
        # n_frames = data.idx.shape[0]
        energy = torch.zeros((n_frames,1), device=pos.device, dtype=pos.dtype)
        for ip,prior in enumerate(self.priors):
            energy += prior(geometry_features[:, prior.callback_indices])
        return energy

class _AbstractPriorLayer(nn.Module):
    """Abstract Layer for definition of priors, which only imposes the minimal
    functional constraints to enable model estimation and inference.
    """
    def __init__(self):
        super(_AbstractPriorLayer, self).__init__()
        self.callback_indices = slice(None, None)

    def forward(self, x):
        """Forward method to compute the prior energy contribution.

        Notes
        -----
        This must be explicitly implemented in a child class that inherits from
        _AbstractPriorLayer(). The details of this method should encompass the
        mathematical steps to form each specific energy contribution to the
        potential energy.;
        """
        raise NotImplementedError(
            'forward() method must be overridden in \
            custom classes inheriting from _AbstractPriorLayer()'
                                  )

class _PriorLayer(_AbstractPriorLayer):
    """Layer for adding prior energy computations external to CGnet hidden
    output

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The strucutre of interaction_parameters
        is the following:

            [ {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... },
              {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... },
                                     .
                                     .
                                     .
              {'parameter_1' : 1.24, 'parameter_2' : 2.21, ... }]

        In this way, _PriorLayer may be subclassed to make arbitray prior
        layers based on arbitrary interactions between bead tuples.

    Attributes
    ----------
    interaction_parameters: list of dict
        each list element contains a dictionary of physical parameters that
        characterizxe the interaction of the associated beads. The order of
        this list proceeds in the same order as self.callback_indices
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    Examples
    --------
    To assemble the feat_dict input for a HarmonicLayer prior for bonds from an
    instance of a stats = GeometryStatistics():

    bonds_interactions, _ = stats.get_prior_statistics('Bonds', as_list=True)
    bonds_idx = stats.return_indices('Bonds')
    bond_layer = HarmonicLayer(bonds_idx, bonds_interactions)

    Notes
    -----
    callback_indices and interaction_parameters MUST share the same order for
    the prior layer to produce correct energies. Using
    GeometryStatistics.get_prior_statistics() with as_list=True together with
    GeometryStatistics.return_indices() will ensure this is True for the same
    list of features.

    The units of the interaction paramters for priors must correspond with the
    units of the input coordinates and force labels used to train the CGnet.

    """

    def __init__(self, callback_indices, interaction_parameters):
        super(_PriorLayer, self).__init__()
        if len(callback_indices) != len(interaction_parameters):
            raise ValueError(
                "callback_indices and interaction_parameters must have the same length"
                )
        self.interaction_parameters = interaction_parameters
        self.callback_indices = callback_indices

class RepulsionLayer(_PriorLayer):
    """Layer for calculating pairwise repulsion energy prior. Pairwise repulsion
    energies are calculated using the following formula:

        U_repulsion_ij = (sigma_{ij} / r_{ij}) ^ exp_{ij}

    where U_repulsion_ij is the repulsion energy contribution from
    coarse grain beads i and j, sigma_ij is the excluded volume parameter
    between the pair (in units of distance), r_ij is the pairwise distance
    (in units of distance) between coarse grain beads i and j, and exp_ij
    is the repulsion exponenent (dimensionless) that characterizes the
    asymptotics of the interaction.

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The parameters for RepulsionLayer
        dictionaries are 'ex_vol', the excluded volume (in length units), and
        'exp', the (positive) exponent characterizing the repulsion strength
        decay with distance.

    Attributes
    ----------
    repulsion_parameters : torch.Tensor
        tensor of shape [2, num_interactions]. The first row contains the
        excluded volumes, the second row contains the exponents, and each
        column corresponds to a single interaction in the order determined
        by self.callback_indices

    Notes
    -----
    This prior energy should be used for longer molecules that may possess
    metastable states in which portions of the molecule that are separated by
    many CG beads in sequence may nonetheless adopt close physical proximities.
    Without this prior, it is possilbe for the CGnet to learn energies that do
    not respect proper physical pairwise repulsions. The interaction is modeled
    after the VDW interaction term from the classic Leonard Jones potential.

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(RepulsionLayer, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('ex_vol', 'exp')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
        repulsion_parameters = torch.tensor([])
        for param_dict in self.interaction_parameters:
            repulsion_parameters = torch.cat((
                repulsion_parameters,
                torch.tensor([[param_dict['ex_vol']],
                              [param_dict['exp']]])), dim=1)
        self.register_buffer('repulsion_parameters', repulsion_parameters)

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.repulsion_parameters[0, :]/in_feat)
                           ** self.repulsion_parameters[1, :],
                           1).reshape(n, 1) / 2
        return energy


class HarmonicLayer(_PriorLayer):
    """Layer for calculating bond/angle harmonic energy prior. Harominc energy
    contributions have the following form:

        U_harmonic_{ij} = 0.5 * k_{ij} * ((r_{ij} - r_0_{ij}) ^ 2)

    where U_harmonic_ij is the harmonic energy contribution from
    coarse grain beads i and j, k_ij is the harmonic spring constant
    (in energy/distance**2) that characterizes the strength of the harmonic
    interaction between coarse grain beads i and j, r_{ij} is the pairwise
    distance (in distance units) between coarse grain beads i and j, and r_0_ij
    is the equilibrium/average pairwise distance (in distance units) between
    coarse grain beads i and j.

    Parameters
    ----------
    callback_indices: list of int
        indices used to access a specified subset of outputs from the feature
        layer through a residual connection

    interaction_parameters : list of python dictionaries
        list of dictionaries that specify the constants characterizing
        interactions between beads. Each list element corresponds to a single
        interaction using a dictionary of parameters keyed to corresponding
        numerical values. The order of these dictionaries follows the same order
        as the callback indices specifying which outputs from the feature layer
        should pass through the prior. The parameters for HarmonicLayer
        dictionaries are 'mean', the center of the harmonic interaction
        (in length or angle units), and 'k', the (positive) harmonic spring
        constant (in units of energy / length**2 or 1 / length**2).

    Attributes
    ----------
    harmonic_parameters : torch.Tensor
        tensor of shape [2, num_interactions]. The first row contains the
        harmonic spring constants, the second row contains the mean positions,
        and each column corresponds to a single interaction in the order
        determined by self.callback_indices

    Notes
    -----
    This prior energy is useful for constraining the CGnet potential in regions
    of configuration space in which sampling is normally precluded by physical
    harmonic constraints associated with the structural integrity of the protein
    along its backbone. The harmonic parameters are also easily estimated from
    all atom simulation data because bond and angle distributions typically have
    Gaussian structure, which is easily intepretable as a harmonic energy
    contribution via the Boltzmann distribution.

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913
    """

    def __init__(self, callback_indices, interaction_parameters):
        super(HarmonicLayer, self).__init__(
            callback_indices, interaction_parameters)
        for param_dict in self.interaction_parameters:
            if (key in param_dict for key in ('k', 'mean')):
                if torch.isnan(param_dict['k']).any():
                    raise ValueError(
                    'Harmonic spring constant "k" contains NaNs.' \
                    'Check your parameters.'
                        )
                if torch.isnan(param_dict['mean']).any():
                    raise ValueError(
                    'Center of the harmonic interaction "mean" contains NaNs.'\
                    'Check your parameters.'
                        )
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
        harmonic_parameters = torch.tensor([])
        for param_dict in self.interaction_parameters:
            harmonic_parameters = torch.cat((harmonic_parameters,
                                             torch.tensor([[param_dict['k']],
                                                           [param_dict['mean']]])),
                                                           dim=1)
        self.register_buffer('harmonic_parameters', harmonic_parameters)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(self.harmonic_parameters[0, :] *
                           (in_feat - self.harmonic_parameters[1, :]) ** 2,
                           1).reshape(n, 1) / 2
        return energy


class ZscoreLayer(nn.Module):
    """Layer for Zscore normalization. Zscore normalization involves
    scaling features by their mean and standard deviation in the following
    way:

        X_normalized = (X - X_avg) / sigma_X

    where X_normalized is the zscore-normalized feature, X is the original
    feature, X_avg is the average value of the orignal feature, and sigma_X
    is the standard deviation of the original feature.

    Parameters
    ----------
    zscores: torch.Tensor
        [2, n_features] tensor, where the first row contains the means
        and the second row contains the standard deviations of each
        feature

    Notes
    -----
    Zscore normalization can accelerate training convergence if placed
    after a GeometryFeature() layer, especially if the input features
    span different orders of magnitudes, such as the combination of angles
    and distances.

    For more information, see the documentation for
    sklearn.preprocessing.StandardScaler

    """

    def __init__(self, zscores):
        super(ZscoreLayer, self).__init__()
        self.register_buffer('zscores', zscores)

    def forward(self, in_feat):
        """Normalizes each feature by subtracting its mean and dividing by
           its standard deviation.

        Parameters
        ----------
        in_feat: torch.Tensor
            input data of shape [n_frames, n_features]

        Returns
        -------
        rescaled_feat: torch.Tensor
            Zscore normalized features. Shape [n_frames, n_features]

        """
        rescaled_feat = (in_feat - self.zscores[0, :])/self.zscores[1, :]
        return rescaled_feat
