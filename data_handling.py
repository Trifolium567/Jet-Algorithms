import numba as nb
import numpy as np
import four_vectors as fv
from numba.typed import List as nblist

listofints_type = nb.extending.as_numba_type(list[int])  # ty: ignore[possibly-missing-submodule]


@nb.experimental.jitclass
class Event:

    status_array: nb.int16[:]  # ty: ignore[invalid-type-form]
    PID_array: nb.int64[:]  # ty: ignore[invalid-type-form]
    four_vectors: fv.FourVectorArray
    four_vectors_reference: fv.FourVectorArray
    clustered_four_vectors: fv.FourVectorArray
    index_reference: list[list[int]]
    clustered_indices: list[list[int]]

    def __init__(
            self, status_array: np.typing.ArrayLike,  # np.typing.ArrayLike is any
            PID_array: np.typing.ArrayLike,           # object that can be coerced into
            E_array: np.typing.ArrayLike,             # an ndarray
            p_x_array: np.typing.ArrayLike,
            p_y_array: np.typing.ArrayLike,
            p_z_array: np.typing.ArrayLike,
        ) -> None:
        """Initialise Event Object.

        Parameters
        ----------
        status_array : array_like
            the status of the detected particles
        PID_array : array_like
            the PID of the detected particles
        E_array : array_like
            the Energies of the detected particles
        p_x_array : array_like
            The momentum in the x-direction of the particles
        p_y_array : array_like
            The momentum in the y-direction of the particles
        p_z_array : array_like
            The momentum in the z-direction of the particles

        Raises
        ------
        ValueError
            If arrays are not all the same shape and not all one dimension
        """
        status_array = np.asarray(status_array, dtype=np.int16)
        PID_array = np.asarray(PID_array, dtype=np.int_)
        E_array = np.asarray(E_array, dtype=np.float64)
        p_x_array = np.asarray(p_x_array, dtype=np.float64)
        p_y_array = np.asarray(p_y_array, dtype=np.float64)
        p_z_array = np.asarray(p_z_array, dtype=np.float64)

        if not (
            status_array.ndim == 1 and PID_array.ndim == 1 and
            E_array.ndim == 1 and p_x_array.ndim == 1 and p_y_array.ndim == 1
            and p_z_array.ndim == 1
        ) or not (
            status_array.shape == PID_array.shape ==
            E_array.shape == p_x_array.shape ==
            p_y_array.shape == p_z_array.shape
            ):
            raise ValueError(
             "All inputted arrays must be of the same shape and dimensions: "
            f"status_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}, \n"
            f"PID_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}, \n"
            f"E_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}, \n"
            f"p_x_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}, \n"
            f"p_y_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}, \n"
            f"p_z_array.ndim = {E_array.ndim}, E_array.shape = {E_array.shape}. "
            )

        self.status_array = status_array
        self.PID_array = PID_array
        self.four_vectors = fv.FourVectorArray(E_array, p_x_array,
                                               p_y_array, p_z_array)
        
        self.four_vectors_reference = fv.FourVectorArray(
            E_array.copy(), p_x_array.copy(), p_y_array.copy(), p_z_array.copy()
        )

        self.index_reference = nblist([nblist([i]) for i in
                                       range(len(self.status_array))])
        self.clustered_indices = nblist.empty_list(listofints_type)
        self.clustered_four_vectors = fv.FourVectorArray([0,], [0,], [0,], [0,])

    def combine_in_reference(self, key1: int, key2: int):
        """Combine two particles into one particle.

        This combines the vectors in a reference FourVectorArray object,
        `four_vectors_reference` and logs which particles have been combined in the 
        `index_reference` array.

        Parameters
        ----------
        key1 : int
            The index of one particle to be combined
        key2 : int
            The index of the other particle to be combined
        """
        self.four_vectors_reference.combine(key1, key2)
        # self.index_reference[key1] += self.index_reference[key2]
        self.index_reference[key1].extend(self.index_reference[key2])
        del self.index_reference[key2]

    def get_one_particle(self, index):
        return self.status_array[index], self.PID_array[index], \
            self.four_vectors.get_one_fourvector(index)
    
    def make_cuts(self, new_obj: bool = False,  # * makes the rest keyword-only arguments
                  min_E: float = 0,
                  max_E: float = np.inf, 
                  min_p_T: float = -np.inf,
                  max_p_T: float = np.inf,
                  min_azimuthal_angle: float = -np.inf,
                  max_azimuthal_angle: float = np.inf,
                  min_rapidity: float = -np.inf,
                  max_rapidity: float = np.inf,
        ) -> "None | Event":
        """Make cuts to the data.

        The removed objects are not completely deleted, they are just removed from the
        reference `FourVectorArray` and the `index_reference`

        If new_obj is True, return a new EventData object with the cuts made.

        Parameters
        ----------
        new_obj : bool, optional
            If new_obj is True, return a new EventData object with the cuts made.
            by default False
        min_E : float, optional
            minimum Energy, by default 0
        max_E : float, optional
            maximum Energy, by default np.inf
        min_p_T : float, optional
            minimum transverse momentum, by default -np.inf
        max_p_T : float, optional
            maximum transverse momentum, by default np.inf
        min_azimuthal_angle : float, optional
            minimum azimuthal angle, by default -np.inf
        max_azimuthal_angle : float, optional
            maximum azimuthal angle, by default np.inf
        min_rapidity : float, optional
            minimum rapidity, by default -np.inf
        max_rapidity : float, optional
            maximum rapidity, by default np.inf

        Returns
        -------
        `None`:
            if new_obj is set to `False`
        
        EventData:
            if new_obj is set to `True`
        """
        # make checks to make sure the important ones are finite
        # cartesian_vectors[0] is energy.
        check_array = np.isfinite(self.four_vectors_reference.cartesian_vectors[0]) &\
                      np.isfinite(self.four_vectors_reference.p_T_array) & \
                      np.isfinite(self.four_vectors_reference.rapidity_array) & \
                      np.isfinite(self.four_vectors_reference.azimuthal_angle_array)

        # specified checks
        check_array &= self.four_vectors_reference.cartesian_vectors[0] > min_E
        check_array &= self.four_vectors_reference.cartesian_vectors[0] < max_E
        check_array &= self.four_vectors_reference.p_T_array > min_p_T
        check_array &= self.four_vectors_reference.p_T_array < max_p_T
        check_array &= self.four_vectors_reference.rapidity_array > min_rapidity
        check_array &= self.four_vectors_reference.rapidity_array < max_rapidity
        check_array &= self.four_vectors_reference.azimuthal_angle_array \
                        > min_azimuthal_angle
        check_array &= self.four_vectors_reference.azimuthal_angle_array \
            < max_azimuthal_angle

        # cut the index_reference list
        new_indices = np.nonzero(check_array)[0]
        new_index_reference = nblist.empty_list(listofints_type, len(new_indices))
        for i in new_indices:
            new_index_reference.append(self.index_reference[i])
        
        checked_four_vectors = self.four_vectors_reference[check_array]
        
        if new_obj:
            return Event(
                self.status_array[check_array],
                self.PID_array[check_array],
                checked_four_vectors.cartesian_vectors[0],
                checked_four_vectors.cartesian_vectors[1],
                checked_four_vectors.cartesian_vectors[2],
                checked_four_vectors.cartesian_vectors[3]
            )
        # else:
        self.four_vectors_reference = checked_four_vectors
        self.index_reference = new_index_reference

    def copy(self):
        return Event(
            self.status_array.copy(), self.PID_array.copy(),
            self.four_vectors.cartesian_vectors[0].copy(),
            self.four_vectors.cartesian_vectors[1].copy(),
            self.four_vectors.cartesian_vectors[2].copy(),
            self.four_vectors.cartesian_vectors[3].copy(),
        )
    
    def _one_kt_pass(self, k, R):
        _parallelised_one_kt_pass(self, k, R)

    def kt_algorithm(self, k, R):
        if not len(self.index_reference):
            raise ValueError("This event is already clustered. Try making a copy using"
                             " .copy() and clustering that one.")
        # else:
        while len(self.index_reference) != 0:
            self._one_kt_pass(k, R)
        # TODO: is this the best exception to raise?
        # except Exception:
        #     print(self.index_reference)
        #     print(len(self.index_reference))
        #     print(self.four_vectors_reference.cartesian_vectors)
        #     print(self.clustered_four_vectors.cartesian_vectors)
        #     print(self.clustered_indices)
    
    # TODO: add reset method? with choice to return new object?


@nb.jit(parallel=True)
def _parallelised_one_kt_pass(instance, k, R):
    # unique combinations is equivalent to a single triangle (displaced from the
    # diagonal by 1) of the indices of an n by n array where n is the number of 
    # events (e.g. upper triangle or lower triangle)
    n = len(instance.index_reference)
    index0, index1 = np.triu_indices(n, k=1)
    # index0 = np.asarray(index0)
    # index1 = np.asarray(index1)
    n_combinations = len(index0)
    n_dparam = n_combinations + n
    i_values = np.empty(n_dparam, dtype=np.int_)
    i_values[:n_combinations] = index0
    j_values = np.empty(n_dparam, dtype=np.int_)
    j_values[:n_combinations] = index1
    d_array = np.empty(n_dparam, dtype=np.float64)
    for m in nb.prange(n_combinations):
        i = index0[m]
        j = index1[m]
        p_T_i = instance.four_vectors_reference.p_T_array[i] ** (2 * k)
        p_T_j = instance.four_vectors_reference.p_T_array[j] ** (2 * k)

        if p_T_i < p_T_j:
            p_T = p_T_i
        else:
            p_T = p_T_j

        Delta_R_ij_sq = (instance.four_vectors_reference.rapidity_array[i] 
                            - instance.four_vectors_reference.rapidity_array[j]) ** 2 \
                        + (instance.four_vectors_reference.azimuthal_angle_array[i]
                        - instance.four_vectors_reference.azimuthal_angle_array[j]) ** 2
        
        d_array[m] = p_T * Delta_R_ij_sq / R ** 2

    for m in nb.prange(n_combinations, n_combinations + n):
        i_values[m] = -1  # -1 codes for distance parameter with the beam
        j_values[m] = m - n_combinations
        d_array[m] = instance.four_vectors_reference.p_T_array[m - n_combinations] \
                                                                        ** (2 * k)

    min_index = np.argmin(d_array)

    if i_values[min_index] == -1:  # if distance parameter with the beam
        # remove detected particle from clustering
        j = j_values[min_index]
        instance.clustered_indices.append(instance.index_reference.pop(j))
        instance.clustered_four_vectors = instance.clustered_four_vectors\
                                          + instance.four_vectors_reference.pop(j)
        # the above appends the four-vector at index in the four-vector reference to
        # the jets FourVectorArray
    else:  # otherwise, it must be between two particles
        instance.combine_in_reference(i_values[min_index], j_values[min_index])

# event_type = nb.extending.as_numba_type(EventData)
# @nb.jit
def events_from_file(filename):
    # event_list = nblist.empty_list(event_type, 10000)
    event_list = []
    with open(filename, mode="r") as file:
        # particle_list = nblist.empty_list(nb.float64[:])
        particle_list = []
        next(file)
        next(file)  # skip first two lines
        # i = 1
        for line in file:
            line_string = line.lstrip().rstrip()
            if not line_string:
                particle_list = np.array(particle_list)
                event_list.append(Event(*particle_list.T))
                particle_list = []
                # i += 1
                # if i % 1000 == 0:
                #     print("Event", i)
                continue
            elif line_string[0] == "#":
                continue
            else:
                particle_list.append(np.fromstring(line, count=6, sep=" "))

        # reached end of file. If there is anything left in particle_list,
        # add it to the event list.
        if particle_list:
            particle_list = np.array(particle_list)
            event_list.append(Event(*particle_list.T))
        # print(f"Done! returned {i} Events")

        return event_list
