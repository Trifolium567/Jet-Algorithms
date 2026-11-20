import numba as nb
import numpy as np
import four_vectors as fv
from numba.typed import List as nblist

listofints_type = nb.extending.as_numba_type(list[int])  # ty: ignore[possibly-missing-submodule]


@nb.experimental.jitclass
class Event:
    """An object representing all the particles of a single event.

    Returns
    -------
    Event
        An instance of this class

    Raises
    ------
    ValueError
        If the input arrays are not all the same shape and not all one dimension
    """

    status_array: nb.int16[:]  # pyright: ignore[reportInvalidTypeForm] # ty: ignore[invalid-type-form]
    PID_array: nb.int64[:]  # pyright: ignore[reportInvalidTypeForm] # ty: ignore[invalid-type-form]
    four_vectors: fv.FourVectorArray

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
            If the input arrays are not all the same shape and not all one dimension.
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

    def get_four_vectors(self, subscript) -> fv.FourVectorArray:
        """Return the four-vectors of the event from a subscript.

        `subscript` can be a list of ints.

        Returns
        -------
        four_vectors.FourVectorArray
        """
        index_array = np.asarray(subscript)
        return self.four_vectors[index_array]

    # def get_one_particle(self, index):
    #     """"""
    #     return self.status_array[index], self.PID_array[index], \
    #         self.four_vectors.get_one_fourvector(index)
    
    def make_cuts(self, new_obj: bool = False,
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

        If new_obj is True, return a new EventData object with the cuts made.

        Parameters
        ----------
        new_obj : bool, optional
            If new_obj is True, return a new EventData object with the cuts made.
            Otherwise, perform the cuts in-place.
            By default `False`.
        min_E : float, optional
            minimum Energy, by default `0`
        max_E : float, optional
            maximum Energy, by default `np.inf`
        min_p_T : float, optional
            minimum transverse momentum, by default `-np.inf`
        max_p_T : float, optional
            maximum transverse momentum, by default `np.inf`
        min_azimuthal_angle : float, optional
            minimum azimuthal angle, by default `-np.inf`
        max_azimuthal_angle : float, optional
            maximum azimuthal angle, by default `np.inf`
        min_rapidity : float, optional
            minimum rapidity, by default `-np.inf`
        max_rapidity : float, optional
            maximum rapidity, by default `np.inf`

        Returns
        -------
        `None`:
            if new_obj is set to `False`
        
        EventData:
            if new_obj is set to `True`
        """
        # make checks to make sure the important ones are finite
        # cartesian_vectors[0] is energy.
        check_array = np.isfinite(self.four_vectors.cartesian_vectors[0]) &\
                      np.isfinite(self.four_vectors.p_T_array) & \
                      np.isfinite(self.four_vectors.rapidity_array) & \
                      np.isfinite(self.four_vectors.azimuthal_angle_array)

        # specified checks
        check_array &= self.four_vectors.cartesian_vectors[0] > min_E
        check_array &= self.four_vectors.cartesian_vectors[0] < max_E
        check_array &= self.four_vectors.p_T_array > min_p_T
        check_array &= self.four_vectors.p_T_array < max_p_T
        check_array &= self.four_vectors.rapidity_array > min_rapidity
        check_array &= self.four_vectors.rapidity_array < max_rapidity
        check_array &= self.four_vectors.azimuthal_angle_array > min_azimuthal_angle
        check_array &= self.four_vectors.azimuthal_angle_array < max_azimuthal_angle

        checked_status_array = self.status_array[check_array]
        checked_PID_array = self.PID_array[check_array]
        checked_four_vectors = self.four_vectors[check_array]
        
        if new_obj:
            return Event(
                checked_status_array,
                checked_PID_array,
                checked_four_vectors.cartesian_vectors[0],
                checked_four_vectors.cartesian_vectors[1],
                checked_four_vectors.cartesian_vectors[2],
                checked_four_vectors.cartesian_vectors[3]
            )
        # else:
        self.status_array = checked_status_array
        self.PID_array = checked_PID_array
        self.four_vectors = checked_four_vectors

    def copy(self):
        """Return a copy of the Event instance."""
        return Event(
            self.status_array.copy(), self.PID_array.copy(),
            self.four_vectors.cartesian_vectors[0].copy(),
            self.four_vectors.cartesian_vectors[1].copy(),
            self.four_vectors.cartesian_vectors[2].copy(),
            self.four_vectors.cartesian_vectors[3].copy(),
        )

    def kt_algorithm(
            self, k: int, R: float, status: int
        ) -> tuple[list[list[int]], fv.FourVectorArray]:
        """Perform the k-t algorithm with on particles of status `status`.

        Parameters
        ----------
        k : int
            The number which determines which algorithm to perform
        R : float
            The R-parameter used in the algorithm
        status : int
            The particles upon which the algorithm is to be performed

        Returns
        -------
        tuple[list[list[int]], fv.FourVectorArray]
            - a list of the clusters represented as indexes into the `status_array`,
                `PID_array` and `four_vectors` attributes of the `Event` instance that
                this method was called from.
            - a `FourVectorArray` object containing the four-vectors of the clusters. 
        """
        detectable_indexes = np.nonzero(self.status_array == status)[0]
        index_reference = nblist.empty_list(listofints_type)
        for i in detectable_indexes:
            index_reference.append(nblist([i]))

        four_vectors_reference = self.four_vectors[self.status_array == status]

        clustered_indices = nblist.empty_list(listofints_type)
        clustered_four_vectors = fv.FourVectorArray([0,], [0,], [0,], [0,], )
        # create four_vector array with zero four-vectors because can't really create
        # empty one (remove this before returning it)

        # else:
        while len(index_reference) != 0:
            _parallelised_one_kt_pass(k, R, index_reference, four_vectors_reference,
                                      clustered_indices, clustered_four_vectors)
        clustered_four_vectors.pop(0)  # remove the first vector
        return clustered_indices, clustered_four_vectors


@nb.jit(parallel=True)  # decorator parallelises code
def _parallelised_one_kt_pass(k, R, index_reference, four_vectors_reference,
                              clustered_indices, clustered_four_vectors):
    """One pass of the k-t algorithm, done outside the class to make use of numba
    parallelisation.
    """  # ruff: ignore[missing-blank-line-after-summary]
    # unique combinations is equivalent to a single triangle (displaced from the
    # diagonal by 1) of the indices of an n by n array where n is the number of 
    # events (e.g. upper triangle or lower triangle)
    n = len(index_reference)
    index0, index1 = np.triu_indices(n, 1)
    n_combinations = len(index0)
    n_dparam = n_combinations + n
    i_values = np.zeros(n_dparam, dtype=np.int_)
    i_values[:n_combinations] = index0
    i_values[n_combinations:] = -1  # identifier for distance parameter with beam
    j_values = np.zeros(n_dparam, dtype=np.int_)
    j_values[:n_combinations] = index1
    j_values[n_combinations:] = np.arange(n)
    d_array = np.zeros(n_dparam, dtype=np.float64)
    for m in nb.prange(n_combinations):
        i = i_values[m]
        j = j_values[m]
        p_T_i = four_vectors_reference.p_T_array[i] ** (2 * k)
        p_T_j = four_vectors_reference.p_T_array[j] ** (2 * k)

        if p_T_i < p_T_j:
            p_T = p_T_i
        else:
            p_T = p_T_j

        # delta_phi = four_vectors_reference.delta_phi(i, j)
        # delta_phi = np.mod(four_vectors_reference.azimuthal_angle_array[i] -
        #                     four_vectors_reference.azimuthal_angle_array[j], np.pi)
        # delta_phi = four_vectors_reference.azimuthal_angle_array[i] - \
        #                     four_vectors_reference.azimuthal_angle_array[j]
        delta_phi = abs(four_vectors_reference.azimuthal_angle_array[i]
                            - four_vectors_reference.azimuthal_angle_array[j])
        delta_phi = min(delta_phi, 2 * np.pi - delta_phi)

        Delta_R_ij_sq = (four_vectors_reference.rapidity_array[i] 
                            - four_vectors_reference.rapidity_array[j]) ** 2 \
                        + delta_phi ** 2

        d_array[m] = p_T * Delta_R_ij_sq / (R ** 2)

    for m in nb.prange(n_combinations, n_combinations + n):
        j = j_values[m]
        d_array[m] = four_vectors_reference.p_T_array[j] ** (2 * k)

    min_index = np.argmin(d_array)

    if i_values[min_index] == -1:  # if distance parameter is with the beam
        #                            remove particle from clustering
        j = j_values[min_index]
        clustered_indices.append(index_reference.pop(j))
        clustered_four_vectors = clustered_four_vectors + four_vectors_reference.pop(j)
        # the above appends the four-vector at index in the four-vector reference to
        # the jets FourVectorArray
    else:  # otherwise, it must be between two particles
        # combine the particles
        i = i_values[min_index]
        j = j_values[min_index]
        four_vectors_reference.combine(i, j)
        index_reference[i].extend(index_reference[j])
        del index_reference[j]


# event_type = nb.extending.as_numba_type(EventData)
# @nb.jit
def events_from_file(filename: str) -> list[Event]:
    """Return a list of Event objects created from inforation read from `filename`.

    Parameters
    ----------
    filename : str
        The name of the file to be read from

    Returns
    -------
    list[Event]
    """
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
