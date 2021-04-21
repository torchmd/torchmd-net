# Authors: Brooke Husic, Nick Charron
# Contributors: Jiang Wang


import copy
import numpy as np
import torch
import scipy.spatial
import warnings

from .geometry import Geometry
g = Geometry(method='numpy')

KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184


class GeometryStatistics(object):
    """Calculation of statistics for geometric features; namely
   distances, angles, and dihedral cosines and sines.

    Parameters
    ----------
    data : torch.Tensor or np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    custom_feature_tuples : list of tuples (default=[])
        List of 2-, 3-, and 4-element tuples containing arbitrary distance,
        angle, and dihedral features to be calculated.
    backbone_inds : 'all', list or np.ndarray, or None (default=None)
        Which bead indices correspond to consecutive beads along the backbone
    get_all_distances : Boolean (default=False)
        Whether to calculate all pairwise distances
    get_backbone_angles : Boolean (default=False)
        Whether to calculate angles among adjacent beads along the backbone
    get_backbone_dihedrals : Boolean (default=False)
        Whether to calculate dihedral cosines and sines among adjacent beads
        along the backbone
    temperature : float or None (default=300.0)
        Temperature of system. Use None for dimensionless calculations.
    get_redundant_distance_mapping : Boolean (default=True)
        If true, creates a redundant_distance_mapping attribute
    bond_pairs : list of tuples (default=[])
        List of 2-element tuples containing bonded pairs
    adjacent_backbone_bonds : Boolean, (default=True)
        Whether adjacent beads along the backbone should be considered
        as bonds

    Attributes
    ----------
    beta : float
        1/(Boltzmann constant)/(temperature) if temperature is not None in
        units of kcal per mole; otherwise 1.0
    descriptions : dictionary
        List of indices (value) for each feature type (key)
    redundant_distance_mapping
        Redundant square distance matrix
    feature_tuples : list of tuples
        List of tuples for non-redundant feature descriptions in order

    Example
    -------
    stats = GeometryStatistics(data, n_beads = 10)
    prior_stats_dict = ds.get_prior_statistics()
    """

    def __init__(self, data, custom_feature_tuples=None, backbone_inds=None,
                 get_all_distances=False, get_backbone_angles=False,
                 get_backbone_dihedrals=False, temperature=300.0,
                 get_redundant_distance_mapping=False, bond_pairs=[],
                 adjacent_backbone_bonds=True):
        if torch.is_tensor(data):
            self.data = data.detach().numpy()
        else:
            self.data = data

        if custom_feature_tuples is None:
            self.custom_feature_tuples = []
        else:
            self.custom_feature_tuples = custom_feature_tuples

        self.n_frames = self.data.shape[0]
        self.n_beads = self.data.shape[1]
        assert self.data.shape[2] == 3  # dimensions
        self.temperature = temperature
        if self.temperature is not None:
            self.beta = JPERKCAL/KBOLTZMANN/AVOGADRO/self.temperature
        else:
            self.beta = 1.0

        if custom_feature_tuples is None:
            if backbone_inds is None:
                raise RuntimeError("You must specify either custom_feature_tuples ' \
                                   'or backbone_inds='all'")
            if type(backbone_inds) is str:
                if backbone_inds == 'all':
                    if get_all_distances + get_backbone_angles + get_backbone_dihedrals == 0:
                        raise RuntimeError('Without custom feature tuples, you must specify ' \
                                           'any of get_all_distances, get_backbone_angles, or ' \
                                           'get_backbone_dihedrals.')
        self._process_backbone(backbone_inds)
        self._process_custom_feature_tuples()

        if get_redundant_distance_mapping and not get_all_distances:
            raise ValueError(
                "Redundant distance mapping can only be returned "
                "if get_all_distances is True."
            )
        self.get_redundant_distance_mapping = get_redundant_distance_mapping

        if not get_all_distances:
            if np.any([bond_ind not in self.custom_feature_tuples
                       for bond_ind in bond_pairs]):
                raise ValueError(
                    "All bond_pairs must be also in custom_feature_tuples "
                    "if get_all_distances is False."
                )
        if np.any([len(bond_ind) != 2 for bond_ind in bond_pairs]):
            raise RuntimeError(
                "All bonds must be of length 2."
            )
        self._bond_pairs = bond_pairs
        self.adjacent_backbone_bonds = adjacent_backbone_bonds

        self.order = []

        self.distances = []
        self.angles = []
        self.dihedral_cosines = []
        self.dihedral_sines = []

        self.descriptions = {
            'Distances': [],
            'Angles': [],
            'Dihedral_cosines': [],
            'Dihedral_sines': []
        }

        self._stats_dict = {}

        # # # # # # #
        # Distances #
        # # # # # # #
        if get_all_distances:
            (self._pair_order,
             self._adj_backbone_pairs) = g.get_distance_indices(self.n_beads,
                                                                self.backbone_inds,
                                                                self._backbone_map)
            if len(self._custom_distance_pairs) > 0:
                warnings.warn(
                    "All distances are already being calculated, so custom distances are meaningless."
                )
                self._custom_distance_pairs = []
            self._distance_pairs = self._pair_order

            if self.adjacent_backbone_bonds:
                if np.any([bond_ind in self._adj_backbone_pairs
                           for bond_ind in self._bond_pairs]):
                    warnings.warn(
                        "Some bond indices were already on the backbone."
                    )
                    # We weed out already existing backbone pairs from the
                    # bond pairs provided by the user. We will append them
                    # to all of our bond pairs below.
                    self._bond_pairs = [bond_ind for bond_ind
                                        in self._bond_pairs if bond_ind
                                        not in self._adj_backbone_pairs]

            # This attribute starts our list of "master" bond pairs
            # when we've done some automatic calculations on the distances
            # because we specified get_all_distances.
            # Note also that we force the user to put backbone bonds in
            # their custom_feature_tuples list if get_all_distances is
            # False, which is why we're still inside the case where
            # get_all_distances is True.
            self.bond_pairs = copy.deepcopy(self._adj_backbone_pairs)

        else:
            self._distance_pairs = []
            # If we haven't specified get_all_distances, our "master"
            # bond list starts out empty
            self.bond_pairs = []
        self._distance_pairs.extend(self._custom_distance_pairs)

        # Extend our master list of bond pairs by the user-defined, possibly
        # filtered bond pairs
        self.bond_pairs.extend(self._bond_pairs)

        if len(self._distance_pairs) > 0:
            self._get_distances()

        # # # # # #
        # Angles  #
        # # # # # #
        if get_backbone_angles:
            self._angle_trips = [(self.backbone_inds[i], self.backbone_inds[i+1],
                                  self.backbone_inds[i+2])
                                 for i in range(len(self.backbone_inds) - 2)]
            if np.any([cust_angle in self._angle_trips
                       for cust_angle in self._custom_angle_trips]):
                warnings.warn(
                    "Some custom angles were on the backbone and will not be re-calculated."
                )
                self._custom_angle_trips = [cust_angle for cust_angle
                                            in self._custom_angle_trips
                                            if cust_angle not in self._angle_trips]
        else:
            self._angle_trips = []
        self._angle_trips.extend(self._custom_angle_trips)
        if len(self._angle_trips) > 0:
            self._get_angles()

        # # # # # # #
        # Dihedrals #
        # # # # # # #
        if get_backbone_dihedrals:
            self._dihedral_quads = [(self.backbone_inds[i], self.backbone_inds[i+1],
                                     self.backbone_inds[i+2], self.backbone_inds[i+3])
                                    for i in range(len(self.backbone_inds) - 3)]
            if np.any([cust_dih in self._dihedral_quads
                       for cust_dih in self._custom_dihedral_quads]):
                warnings.warn(
                    "Some custom dihedrals were on the backbone and will not be re-calculated."
                )
                self._custom_dihedral_quads = [cust_dih for cust_dih
                                               in self._custom_dihedral_quads
                                               if cust_dih not in self._dihedral_quads]
        else:
            self._dihedral_quads = []
        self._dihedral_quads.extend(self._custom_dihedral_quads)
        if len(self._dihedral_quads) > 0:
            self._get_dihedrals()

        self.feature_tuples = []
        self.master_description_tuples = []
        self._master_stat_array = [[] for _ in range(3)]

        for feature_type in self.order:
            if feature_type not in ['Dihedral_cosines', 'Dihedral_sines']:
                self.feature_tuples.extend(self.descriptions[feature_type])
                self.master_description_tuples.extend(
                    self.descriptions[feature_type])
                self._master_stat_array[0].extend(
                    self._stats_dict[feature_type]['mean'])
                self._master_stat_array[1].extend(
                    self._stats_dict[feature_type]['std'])
                self._master_stat_array[2].extend(
                    self._stats_dict[feature_type]['k'])

            else:
                self.master_description_tuples.extend(
                    [self._get_key(desc, feature_type)
                     for desc in self.descriptions[feature_type]])
                self._master_stat_array[0].extend(
                    self._stats_dict[feature_type]['mean'])
                self._master_stat_array[1].extend(
                    self._stats_dict[feature_type]['std'])
                self._master_stat_array[2].extend(
                    self._stats_dict[feature_type]['k'])
                if feature_type == 'Dihedral_cosines':
                    # because they have the same indices as dihedral sines,
                    # do only cosines
                    self.feature_tuples.extend(self.descriptions[feature_type])
        self._master_stat_array = np.array(self._master_stat_array)

    def _process_custom_feature_tuples(self):
        """Helper function to sort custom features into distances, angles,
        and dihedrals.
        """
        if len(self.custom_feature_tuples) > 0:
            if (np.min([len(feat) for feat in self.custom_feature_tuples]) < 2 or
                    np.max([len(feat) for feat in self.custom_feature_tuples]) > 4):
                raise ValueError(
                    "Custom features must be tuples of length 2, 3, or 4."
                )
            if np.max([np.max(bead)
                       for bead in self.custom_feature_tuples]) > self.n_beads - 1:
                raise ValueError(
                    "Bead index in at least one feature is out of range."
                )

            _temp_dict = dict(
                zip(self.custom_feature_tuples,
                    np.arange(len(self.custom_feature_tuples))))
            if len(_temp_dict) < len(self.custom_feature_tuples):
                self.custom_feature_tuples = list(_temp_dict.keys())
                warnings.warn(
                    "Some custom feature tuples are repeated and have been removed."
                )

            self._custom_distance_pairs = [
                feat for feat in self.custom_feature_tuples if len(feat) == 2]
            self._custom_angle_trips = [
                feat for feat in self.custom_feature_tuples if len(feat) == 3]
            self._custom_dihedral_quads = [
                feat for feat in self.custom_feature_tuples if len(feat) == 4]
        else:
            self._custom_distance_pairs = []
            self._custom_angle_trips = []
            self._custom_dihedral_quads = []

    def _get_backbone_map(self):
        """Helper function that maps bead indices to indices along the backbone
        only.

        Returns
        -------
        backbone_map : dict
            Dictionary with bead indices as keys and, as values, backbone
            indices for beads along the backbone or np.nan otherwise.
        """
        backbone_map = {mol_ind: bb_ind for bb_ind, mol_ind
                        in enumerate(self.backbone_inds)}
        pad_map = {mol_ind: np.nan for mol_ind
                   in range(self.n_beads) if mol_ind not in self.backbone_inds}
        return {**backbone_map, **pad_map}

    def _process_backbone(self, backbone_inds):
        """Helper function to obtain attributes needed for backbone atoms.
        """
        if type(backbone_inds) is str:
            if backbone_inds == 'all':
                self.backbone_inds = np.arange(self.n_beads)
                self._backbone_map = {ind: ind for ind in range(self.n_beads)}
            else:
                raise RuntimeError(
                    "backbone_inds must be list or np.ndarray of indices, 'all', or None"
                    )
        elif type(backbone_inds) in [list, np.ndarray]:
            if len(np.unique(backbone_inds)) != len(backbone_inds):
                raise ValueError(
                    'Backbone is not allowed to have repeat entries')
            self.backbone_inds = np.array(backbone_inds)

            if not np.all(np.sort(self.backbone_inds) == self.backbone_inds):
                warnings.warn(
                    "Your backbone indices aren't sorted. Make sure your backbone indices are in consecutive order."
                )

            self._backbone_map = self._get_backbone_map()
        elif backbone_inds is None:
            if len(self.custom_feature_tuples) == 0:
                raise RuntimeError(
                    "Must have either backbone or custom features. Did you forget "
                    "to specify backbone_inds='all'?")
            self.backbone_inds = np.array([])
            self._backbone_map = None
        else:
            raise RuntimeError(
                "backbone_inds must be list or np.ndarray of indices, 'all', or None"
            )
        self.n_backbone_beads = len(self.backbone_inds)

    def _get_distances(self):
        """Obtains all pairwise distances for the two-bead indices provided.
        """
        self.distances = g.get_distances(
            self._distance_pairs, self.data, norm=True)
        self.descriptions['Distances'].extend(self._distance_pairs)
        self._get_stats(self.distances, 'Distances')
        self.order += ['Distances']
        if self.get_redundant_distance_mapping:
            self.redundant_distance_mapping = g.get_redundant_distance_mapping(
                self._distance_pairs)

    def _get_angles(self):
        """Obtains all planar angles for the three-bead indices provided.
        """
        self.angles = g.get_angles(self._angle_trips, self.data)

        self.descriptions['Angles'].extend(self._angle_trips)
        self._get_stats(self.angles, 'Angles')
        self.order += ['Angles']

    def _get_dihedrals(self):
        """Obtains all dihedral angles for the four-bead indices provided.
        """
        (self.dihedral_cosines,
            self.dihedral_sines) = g.get_dihedrals(self._dihedral_quads, self.data)

        self.descriptions['Dihedral_cosines'].extend(self._dihedral_quads)
        self.descriptions['Dihedral_sines'].extend(self._dihedral_quads)

        self._get_stats(self.dihedral_cosines, 'Dihedral_cosines')
        self._get_stats(self.dihedral_sines, 'Dihedral_sines')
        self.order += ['Dihedral_cosines']
        self.order += ['Dihedral_sines']

    def _get_stats(self, X, key):
        """Populates stats dictionary with mean and std of feature.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        var = np.var(X, axis=0)
        k = 1/var/self.beta
        self._stats_dict[key] = {}
        self._stats_dict[key]['mean'] = mean
        self._stats_dict[key]['std'] = std
        self._stats_dict[key]['k'] = k

    def _get_key(self, key, name):
        """Returns keys for zscore and bond constant dictionaries based on
        description names.
        """
        if name == 'Dihedral_cosines':
            return tuple(list(key) + ['cos'])
        if name == 'Dihedral_sines':
            return tuple(list(key) + ['sin'])
        else:
            return key

    def _flip_dict(self, mydict):
        """Flips the dictionary; see documentation for get_zscores or
        get_bond_constants.
        """
        all_inds = list(mydict['mean'].keys())

        newdict = {}
        for i in all_inds:
            newdict[i] = {}
            for stat in mydict.keys():
                if i in mydict[stat].keys():
                    newdict[i][stat] = mydict[stat][i]
        return newdict

    def get_prior_statistics(self, features=None, tensor=True,
                             as_list=False, flip_dict=True):
        """Obtain prior statistics (mean, standard deviation, and
        bond/angle/dihedral constants) for features

        Parameters
        ----------
        features : str or list of tuples (default=None)
            specifies which feature to form the prior statistics for. If list
            of tuples is provided, only those corresponding features will be
            processed. If None, all features will be processed.
        tensor : Boolean (default=True)
            Returns (innermost data) of type torch.Tensor if True and np.array
             if False
        as_list : Boolean (default=True)
            if True, a list of individual dictionaries is returned instead of
            a nested dictionary. The ordering of the list is the same as the
            ordering of input feature tuples.
        flip_dict : Boolean (default=True)
            If as_list is False, returns a dictionary with outer keys as
            indices if True and outer keys as statistic string names if False

        Returns
        -------
        prior_statistics_dict : python dictionary (if as_dict=True)
            If flip_dict is True, the outer keys will be bead pairs, triples,
            or quadruples+phase, e.g. (1, 2) or (0, 1, 2, 3, 'cos'), and
            the inner keys will be 'mean', 'std', and 'k' statistics.
            If flip_dict is False, the outer keys will be the 'mean' and 'std'
            statistics and the inner keys will be bead pairs, triples, or
            quadruples+phase
        prior_statistics_list : list of python dictionaries (if as_list=True)
            Each element of the list is a dictionary containing the 'mean',
            'std', and 'k' statistics. The list elements share the same order
            as the input feature tuples
        prior_statistics_keys: dict_keys (tuples of beads)
            If as_list=True, the prior statistics dictionary keys are returned
            which correspond to the ordering of the prior_statistiscs_list

        Notes
        -----
        Dihedral features must specify 'cos' or 'sin', e.g. (1, 2, 3, 4, 'sin')
        """
        if features is not None:
            stats_inds = self.return_indices(features)
        else:
            stats_inds = np.arange(len(self.master_description_tuples))
        self._stats_inds = stats_inds

        prior_statistics_keys = [self.master_description_tuples[i]
                                 for i in stats_inds]
        prior_statistics_array = self._master_stat_array[:, stats_inds]

        if tensor:
            prior_statistics_array = torch.from_numpy(
                prior_statistics_array).float()
        self.prior_statistics_keys = prior_statistics_keys
        self._prior_statistics_array = prior_statistics_array
        prior_statistics_dict = {}
        for i, stat in enumerate(['mean', 'std', 'k']):
            prior_statistics_dict[stat] = dict(zip(prior_statistics_keys,
                                                   prior_statistics_array[i, :]))
        if as_list:
            prior_statistics_list = []
            for i in range(prior_statistics_array.shape[1]):
                prior_statistics_list.append(
                    {'mean': prior_statistics_array[0, i],
                     'std': prior_statistics_array[1, i],
                     'k': prior_statistics_array[2, i]}
                )
            prior_statistics_dict = self._flip_dict(prior_statistics_dict)
            return prior_statistics_list, prior_statistics_keys
        else:
            if flip_dict:
                prior_statistics_dict = self._flip_dict(prior_statistics_dict)
            return prior_statistics_dict

    def get_zscore_array(self, features=None, tensor=True):
        """Obtain a 2 x n array of means and standard deviations, respectively,
        for n features.

        Parameters
        ----------
        features : str or list of tuples (default=None)
            specifies which feature to form the prior statistics for. If list
            of tuples is provided, only those corresponding features will be
            processed. If None, all features will be processed.
        tensor : Boolean (default=True)
            Returns (innermost data) of type torch.Tensor if True and np.array
             if False

        Returns
        -------
        zscore_array : np.ndarray
            Array of shape [2, n_features] containing the means in the first
            row and the standard deviations in the second row. The elements of
            the array second dimension share the same order as the input
            feature tuples
        zscore_keys: dict_keys (tuples of beads)
            The features corresponding to the indices of the second dimension
            of the zscore_array

        Notes
        -----
        Dihedral features must specify 'cos' or 'sin', e.g. (1, 2, 3, 4, 'sin')
        """
        all_stat_values, zscore_keys = self.get_prior_statistics(
            features=features,
            tensor=False,
            as_list=True
        )

        zscore_array = np.vstack([[all_stat_values[i][stat]
                                   for i in range(len(all_stat_values))]
                                  for stat in ['mean', 'std']])

        if tensor:
            zscore_array = torch.tensor(zscore_array).float()

        return zscore_array, zscore_keys

    def return_indices(self, features):
        """Return all indices for specified feature type. Useful for
        constructing priors or other layers that make callbacks to
        a subset of features output from a GeometryFeature()
        layer

        Parameters
        ----------
        features : str in {'Distances', 'Bonds', 'Angles',
                               'Dihedral_sines', 'Dihedral_cosines'}
                   or list of tuples of integers
            specifies for which feature type the indices should be returned.
            If tuple input, the indices corresponding to those bead groups
            will be returned instead.

        Returns
        -------
        indices : list(int)
            list of integers corresponding the indices of specified features
            output from a GeometryFeature() layer.

        Notes
        -----
        Dihedral features must specify 'cos' or 'sin', e.g. (1, 2, 3, 4, 'sin')

        """
        if isinstance(features, str):
            if features not in self.descriptions.keys() and features != 'Bonds':
                raise RuntimeError(
                    "Error: \'{}\' is not a valid backbone feature.".format(
                        features)
                )
            if features in ["Distances", "Angles"]:
                return [ind for ind, feat in
                        enumerate(self.master_description_tuples)
                        if feat in self.descriptions[features]]
            elif features == "Dihedral_cosines":
                return [ind for ind, feat in
                        enumerate(self.master_description_tuples)
                        if feat[:-1] in self.descriptions[features]
                        and feat[-1] == 'cos']
            elif features == "Dihedral_sines":
                return [ind for ind, feat in
                        enumerate(self.master_description_tuples)
                        if feat[:-1] in self.descriptions[features]
                        and feat[-1] == 'sin']
            elif features == 'Bonds':
                return [ind for ind, feat in
                        enumerate(self.master_description_tuples)
                        if feat in self.bond_pairs]

        elif isinstance(features, list):
            if any(len(bead_tuple) == 4 for bead_tuple in features):
                raise ValueError("Bead tuples of 4 beads need to specify "
                                 "'cos' or 'sin' as 5th element")
            if (np.min([len(bead_tuple) for bead_tuple in features]) < 2 or
                    np.max([len(bead_tuple) for bead_tuple in features]) > 5):
                raise ValueError(
                    "Features must be tuples of length 2, 3, or 5."
                )

            tupl_to_ind_dict = {tupl: i for i, tupl in
                                enumerate(self.master_description_tuples)}
            return [tupl_to_ind_dict[tupl] for tupl in features]

        else:
            raise ValueError(
                "features must be description string or list of tuples.")


def kl_divergence(dist_1, dist_2):
    r"""Compute the Kullback-Leibler (KL) divergence between two discrete
    distributions according to:

    \sum_i P_i \log(P_i / Q_i)

    where P_i is the reference distribution and Q_i is the test distribution

    Parameters
    ----------
    dist_1 : numpy.array
        reference distribution of shape [n,] for n points
    dist_2 : numpy.array
        test distribution of shape [n,] for n points

    Returns
    -------
    divergence : float
        the Kullback-Leibler divergence of the two distributions

    Notes
    -----
    The KL divergence is not symmetric under distribution exchange;
    the expectation is taken over the reference distribution.

    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')

    dist_1m = np.ma.masked_where(dist_1 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_2 == 0, dist_2)
    summand = dist_1m * np.ma.log(dist_1m / dist_2m)
    divergence = np.ma.sum(summand)
    return divergence


def js_divergence(dist_1, dist_2):
    r"""Compute the Jenson-Shannon (JS) divergence between two discrete
    distributions according to:

    0.5 * \sum_i P_i \log(P_i / M_i) + 0.5 * \sum_i Q_i \log(Q_i / M_i),

    where M_i is the elementwise mean of P_i and Q_i. This is equivalent to,

    0.5 * kl_divergence(P, Q) + 0.5 * kl_divergence(Q, P).

    Parameters
    ----------
    dist_1 : numpy.array
        first distribution of shape [n,] for n points
    dist_2 : numpy.array
        second distribution of shape [n,] for n points

    Returns
    -------
    divergence : float
        the Jenson-Shannon divergence of the two distributions

    Notes
    -----
    The JS divergence is the symmetrized extension of the KL divergence.
    It is also referred to as the information radius.

    References
    ----------
    Lin, J. (1991). Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory.
        https://dx.doi.org/10.1109/18.61115

    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')

    dist_1m = np.ma.masked_where(dist_1 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_2 == 0, dist_2)
    elementwise_mean = 0.5 * (dist_1m + dist_2m)
    divergence = (0.5*kl_divergence(dist_1m, elementwise_mean) +
                  0.5*kl_divergence(dist_2m, elementwise_mean))
    return divergence


def discrete_distribution_intersection(dist_1, dist_2, bin_edges=None,
                                       tol=1e-6):
    """Compute the intersection between two discrete distributions

    Parameters
    ----------
    dist_1 : numpy.array
        first distribution of shape [n,] for n points
    dist_2 : numpy.array
        second distribution of shape [n,] for n points
    bin_edges : None or numpy.array (default=None)
        Edges for (consecutive bins) for both dist1 and dist2; must be
        identical for both distributions of shape [k + 1,] for k consecutive
        bins with k+1 edges. If None, consecutive bins of uniform size are
        assumed.
    tol : float (default=1e-6)
        Tolerance for ensuring distributions are normalized. You shouldn't
        need to change this.

    Returns
    -------
    intersect : float
        The intersection of the two distributions; i.e., the overlapping
        density. A full overlap returns 1 and zero overlap returns 0.
    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')
    if bin_edges is not None and len(dist_1) + 1 != len(bin_edges):
        raise ValueError(
            'bin_edges length must be 1 more than distribution length')

    if np.max([np.abs(np.sum(dist_1)-1.), np.abs(np.sum(dist_2)-1.)]) > tol:
        raise ValueError(
            'Distributions must be normalized.'
        )

    if bin_edges is None:
        # The bins should be separated by 1 for normalized distributions
        bin_edges = np.linspace(0, len(dist_1), len(dist_1) + 1)

    intervals = np.diff(bin_edges)

    dist_1m = np.ma.masked_where(dist_1*dist_2 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_1*dist_2 == 0, dist_2)

    intersection = np.ma.multiply(np.ma.min([dist_1m, dist_2m], axis=0),
                                  intervals).sum()
    return intersection
