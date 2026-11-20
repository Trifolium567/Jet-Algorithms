import numba as nb
import numpy as np
from numba.typed import List as nblist


@nb.experimental.jitclass([
    ("cartesian_vector", nb.float64[:]),
])
class FourVector:
    """Four-Vector object."""

    cartesian_vector: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(self, E: float, p_x: float, p_y: float, p_z: float):
        """Initialise FourVector object.

        Parameters
        ----------
        E : float
            Total energy in GeV
        p_x : float
            The momentum in the x-direction, in units of  GeV / c
        p_y : float
            The momentum in the y-direction, in units of Gev / c
        p_z : float
            The momentum in the z-direction, in units of Gev / v
        """
        self.cartesian_vector = np.asarray([E, p_x, p_y, p_z])

    @property
    def p_T(self):
        """Transverse momentum of the four-vector."""
        return find_transverse_momentum(self.cartesian_vector[1],
                                        self.cartesian_vector[2])
    
    @property
    def azimuthal_angle(self):
        """Azimuthal angle of the four-vector."""
        return find_azimuthal_angle(self.cartesian_vector[1], self.cartesian_vector[2])

    @property
    def rapidity(self):
        """Rapidity of the four-vector."""
        return find_rapidity(self.cartesian_vector[0], self.cartesian_vector[3])
        
    def __abs__(self) -> float:
        """Return the magnitude of the four-vector, p^2.
        
        p^2 = E^2 - p_x^2 - p_y^2 - p_z^2
        builtin abs(FourVector) is faster than numpy.abs(FourVector) as numpy has its
        own overheads and will not give any improvements
        """
        return np.dot(self.cartesian_vector, self.cartesian_vector
                      * np.array([1, -1, -1, -1]))

    def __add__(
            self, other: "FourVector | FourVectorArray"
        ) -> "FourVector | FourVectorArray":
        """Add the FourVector Objects.

        Parameters
        ----------
        other : FourVector
            The FourVector to add to the FourVector object this is called from

        Returns
        -------
        FourVector
            The result of the addition
        """
        if isinstance(other, FourVector):
            result = self.cartesian_vector + other.cartesian_vector
            return FourVector(result[0], result[1], result[2], result[3])
        elif isinstance(other, FourVectorArray):
            append_array = np.array([[self.cartesian_vector[0]],
                                     [self.cartesian_vector[1]],
                                     [self.cartesian_vector[2]],
                                     [self.cartesian_vector[3]]])
            result_array = np.append(other.cartesian_vectors, append_array, axis=1)
            return FourVectorArray(result_array[0], result_array[1], result_array[2],
                                   result_array[3])
        else:
            raise TypeError(f"Operator '+' is only supported when {self} and {other}"
                            " are of type FourVector or FourVectorArray")
 
    # def __str__(self) -> str:  # useful for debugging
    #     """Represent the FourVector object as a string."""
    #     return f"FourVector({self.vector[0]}, {self.vector[1]}, " + \
    #            f"{self.vector[2]}, {self.vector[3]})"
    
    def __str__(self) -> str:  # for debugging - (done this way for numba compatibility)
        """Print the FourVector object."""
        print("FourVector(", self.cartesian_vector[0], "\b,", self.cartesian_vector[1],
              "\b,", self.cartesian_vector[2], "\b,", self.cartesian_vector[3], ")") 
        # numba currently does not have support for str(numba.float64)
        # return f"FourVector({self.vector[0]}, {self.vector[1]}, " + \
        #        f"{self.vector[2]}, {self.vector[3]})"
        return ""
    
    def boost(self, beta_x: float, beta_y: float,
              beta_z: float,
        ) -> "FourVector":
        """Boost the FourVector object to another frame.

        Parameters
        ----------
        beta_x : float
            The velocity in x of the frame, in units of c
        beta_y : float
            The veloctiy in y of the frame, in units of c
        beta_z : float
            The veloctiy in z of the frame, in units of c

        Returns
        -------
        FourVector
            The boosted four-vector
        """
        gamma = 1 / np.sqrt(1 - beta_x**2 - beta_y**2 - beta_z**2)
        alpha = gamma ** 2 / (1 + gamma)
        result = np.array(
            [[gamma, - gamma * beta_x, - gamma * beta_y, - gamma * beta_z],
             [-gamma * beta_x, 1 + alpha * beta_x**2, alpha * beta_x * beta_y,
              alpha * beta_x * beta_z],
             [-gamma * beta_y, alpha * beta_y * beta_z, 1 + alpha * beta_y**2,
              alpha * beta_y * beta_z],
             [-gamma * beta_z, alpha * beta_z * beta_x, alpha * beta_z * beta_y,
              1 + alpha * beta_z**2]
        ]) @ self.cartesian_vector
        # numba will give a performance warning saying that self.cartesian_vectors isn't
        # contigious. It has been set so above.

        return FourVector(result[0], result[1], result[2], result[3])


@nb.experimental.jitclass([
    ("cartesian_vectors", nb.float64[:, :]),
    ("p_T_array", nb.float64[:]),
    ("azimuthal_angle_array", nb.float64[:]),
    ("rapidity_array", nb.float64[:]),
])
class FourVectorArray:
    """Object to store and effectively operate on four-vectors."""

    cartesian_vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    p_T_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    azimuthal_angle_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    rapidity_array: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(self,
                 E_array: np.typing.ArrayLike,
                 p_x_array: np.typing.ArrayLike,
                 p_y_array: np.typing.ArrayLike,
                 p_z_array: np.typing.ArrayLike,
        ):
        """Initialise the FourVectorArray object.

        Parameters
        ----------
        E_array : array_like
            Total energy in GeV
        p_x_array : array_like
            The momentum in the x-direction, in units of  GeV / c
        p_y_array : array_like
            The momentum in the y-direction, in units of Gev / c
        p_z_array : array_like
            The momentum in the z-direction, in units of Gev / v

        Raises
        ------
        ValueError
            If the parameters are not of the same length or are not of only 1 dimension.
        """
        E = np.asarray(E_array, dtype=np.float64)
        p_x = np.asarray(p_x_array, dtype=np.float64)
        p_y = np.asarray(p_y_array, dtype=np.float64)
        p_z = np.asarray(p_z_array, dtype=np.float64)

        E_shape = E.shape
        x_shape = p_x.shape
        y_shape = p_x.shape
        z_shape = p_x.shape

        error_message = ""
        if not (E_shape == x_shape == y_shape == z_shape):
            error_message += f"E_array.shape = {E_shape}, p_x_array.shape = " + \
                f"{x_shape}, p_y_array.shape = {y_shape}, p_z_array.shape = {z_shape} "
        if not (E.ndim == 1 and p_x.ndim == 1 and p_y.ndim == 1 and p_z.ndim == 1):
            error_message += f"E_array.ndim = {E.ndim}, p_x_array.ndim = {p_x.ndim}," +\
                f" p_y_array.ndim = {p_y.ndim}, p_z_array.ndim = {p_z.ndim}. " + \
                f"All inputs should be 1d."
    
        if error_message:
            raise ValueError(error_message)
        # else:
        container_array = np.zeros((4, E_shape[0]), dtype=np.float64)

        container_array[0] = E
        container_array[1] = p_x
        container_array[2] = p_y
        container_array[3] = p_z

        self.cartesian_vectors = np.ascontiguousarray(container_array)
        self.p_T_array = find_transverse_momentum(self.cartesian_vectors[1],
                                                  self.cartesian_vectors[2])
        self.azimuthal_angle_array = find_azimuthal_angle(self.cartesian_vectors[1],
                                                            self.cartesian_vectors[2])
        self.rapidity_array = find_rapidity(self.cartesian_vectors[0],
                                            self.cartesian_vectors[3])

    def __getitem__(self, subscript) -> "FourVectorArray":
        """Define indexing FourVectorArray objects.
        
        Subscript works the same as numpy arrays. DOES NOT WORK FOR INDEXING ONE
        FOUR-VECTOR AT A TIME.

        Returns
        -------
        FourVectorArray
        """
        result = self.cartesian_vectors.T[subscript].T
        instance = FourVectorArray(result[0], result[1], result[2], result[3])
        return instance
    
    def __setitem__(self, key, value) -> None:
        """Set the item or items at `key` to value.
        
        Value should be of the signature:
        `[E_array, p_x_array, p_y_array, p_z_array]` or `[E, p_x, p_y, p_z]`
        """
        if isinstance(value, FourVector):
            self.cartesian_vectors.T[key] = value.cartesian_vector
            self.p_T_array[key] = value.p_T
            self.rapidity_array[key] = value.rapidity
            self.azimuthal_angle_array[key] = value.azimuthal_angle
        elif isinstance(value, FourVectorArray):
            self.cartesian_vectors.T[key] = value.cartesian_vectors.T
            self.azimuthal_angle_array[key] = value.azimuthal_angle_array
            self.rapidity_array[key] = value.rapidity_array
            self.p_T_array[key] = value.p_T_array
        else:
            self.cartesian_vectors.T[key] = value.T
            self.p_T_array[key] = find_transverse_momentum(value[1], value[2])
            self.azimuthal_angle_array[key] = find_azimuthal_angle(value[1], value[2])
            self.rapidity_array[key] = find_rapidity(value[0], value[3])

    def __abs__(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]: 
        """Return an array of the magnitudes of the four-vectors, p^2.

        Where p^2 = E^2 - p_x^2 - p_y^2 - p_z^2

        builtin abs(FourVector) is faster than numpy.abs(FourVector) as numpy has its
        own overheads and will not give any improvements.
        """
        return np.sum(
            self.cartesian_vectors ** 2 * np.array(((1,), (-1,), (-1,), (-1,))), axis=0,
        )

    def __add__(self,
                other: "FourVectorArray | FourVector"
        ) -> "FourVectorArray | None":
        """Append a FourVectorArray or FourVector to this one and return it."""
        if isinstance(other, FourVectorArray):
            result_array = np.append(self.cartesian_vectors, other.cartesian_vectors,
                                     axis=1)
            return FourVectorArray(result_array[0], result_array[1], result_array[2],
                                   result_array[3])
        elif isinstance(other, FourVector):
            append_array = np.array([[other.cartesian_vector[0]],
                                     [other.cartesian_vector[1]],
                                     [other.cartesian_vector[2]],
                                     [other.cartesian_vector[3]]])
            result_array = np.append(self.cartesian_vectors, append_array,
                                     axis=1)
            return FourVectorArray(result_array[0], result_array[1], result_array[2],
                                   result_array[3])
        else:
            raise TypeError(
                "The operation '+' on this object is only supported with FourVectors "
                "and FourVectorArrays."
            )

    # do not make a __delitem__, numba does not support it.
    
    def get_one_fourvector(self, index: int) -> FourVector:
        """Return a `FourVector` object from the vector stored at `index`."""
        vector = self.cartesian_vectors.T[index]
        return FourVector(vector[0], vector[1], vector[2], vector[3])

    def pop(self, index: int) -> FourVector:
        """Return the four-vector at `index` as a `FourVector` object and remove it."""
        value = self.get_one_fourvector(index)
        index_mask = np.arange(len(self.azimuthal_angle_array)) != index
        self.cartesian_vectors = self.cartesian_vectors.T[index_mask].T
        self.azimuthal_angle_array = self.azimuthal_angle_array[index_mask]
        self.rapidity_array = self.rapidity_array[index_mask]
        self.p_T_array = self.p_T_array[index_mask]
        return value

    def combine(self, index1: int, index2: int) -> None:
        """Combine two vectors at indices `index1` and `index2`."""
        # get add the four-vectors and set index1 to the result
        result = self.cartesian_vectors.T[index1] + self.cartesian_vectors.T[index2]
        self.cartesian_vectors.T[index1] = result  # result signature [E, p_x, p_y, p_z]
        # recalculate the other values and set them
        self.azimuthal_angle_array[index1] = find_azimuthal_angle(result[1], result[2])
        self.rapidity_array[index1] = find_rapidity(result[0], result[3])
        self.p_T_array[index1] = find_transverse_momentum(result[1], result[2])
        # delete the four-vector at index2
        index_mask = np.arange(len(self.azimuthal_angle_array)) != index2
        self.cartesian_vectors = self.cartesian_vectors.T[index_mask].T
        self.azimuthal_angle_array = self.azimuthal_angle_array[index_mask]
        self.rapidity_array = self.rapidity_array[index_mask]
        self.p_T_array = self.p_T_array[index_mask]

    def delta_phi(self, index1: int, index2: int) -> None:
        """Find the difference in azimuthal angle between the four-vectors at `index1` and `index2`."""  # ruff: ignore[line-too-long]
        p_1 = np.asarray(self.cartesian_vectors.T[index1][1:3])
        p_2 = np.asarray(self.cartesian_vectors.T[index2][1:3])
        cos_phi = np.clip(
            np.dot(p_1, p_2) / np.sqrt(np.sum(p_1 ** 2) * np.sum(p_2 ** 2)), -1.0, 1.0
        )
        return np.acos(cos_phi)

    def _resultofboost(
            self, beta_x: float, beta_y: float, beta_z: float
        ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Boost the whole array of Four-Vectors.

        Parameters
        ----------
        beta_x : float
            The velocity in x of the frame, in units of c
        beta_y : float
            The veloctiy in y of the frame, in units of c
        beta_z : float
            The veloctiy in z of the frame, in units of c
        
        Returns
        -------
        numpy.ndarray
            The result of the boost on all of the four-vectors
        """
        gamma = 1 / np.sqrt(1 - beta_x**2 - beta_y**2 - beta_z**2)
        alpha = gamma ** 2 / (1 + gamma)
        result = np.array(
            [[gamma, - gamma * beta_x, - gamma * beta_y, - gamma * beta_z],
             [-gamma * beta_x, 1 + alpha * beta_x**2, alpha * beta_x * beta_y,
              alpha * beta_x * beta_z],
             [-gamma * beta_y, alpha * beta_y * beta_z, 1 + alpha * beta_y**2,
              alpha * beta_y * beta_z],
             [-gamma * beta_z, alpha * beta_z * beta_x, alpha * beta_z * beta_y,
              1 + alpha * beta_z**2]
        ]) @ self.cartesian_vectors
        # numba will give a performance warning saying that self.cartesian_vectors isn't
        # contigious. It has been set so above.

        return result

    def boost(self, beta_x: float, beta_y: float, beta_z: float) -> None:
        """Boost the four-vectors in place.

        Parameters
        ----------
        beta_x : float
            The velocity in x of the frame, in units of c
        beta_y : float
            The veloctiy in y of the frame, in units of c
        beta_z : float
            The veloctiy in z of the frame, in units of c
        """
        self.cartesian_vectors = self._resultofboost(beta_x, beta_y, beta_z)
        self.rapidity_array = find_rapidity(self.cartesian_vectors[0],
                                            self.cartesian_vectors[3])
        self.p_T_array = find_transverse_momentum(self.cartesian_vectors[1],
                                                  self.cartesian_vectors[2])
        self.azimuthal_angle_array = find_azimuthal_angle(self.cartesian_vectors[1],
                                                          self.cartesian_vectors[2])

    def boosted(self, beta_x: float, beta_y: float, beta_z: float) -> "FourVectorArray":
        """Return a boosted FourVectorArray object.

        Parameters
        ----------
        beta_x : float
            The velocity in x of the frame to be boosted into, in units of c
        beta_y : float
            The veloctiy in y of the frame to be boosted into, in units of c
        beta_z : float
            The veloctiy in z of the frame to be boosted into, in units of c

        Returns
        -------
        FourVectorArray
        """
        result_array = self._resultofboost(beta_x, beta_y, beta_z)
        return FourVectorArray(result_array[0], result_array[1], result_array[2],
                               result_array[3])


@nb.jit
def find_rapidity(E, p_z):
    """Return the rapidity from energy `E` and momentum `p_z`."""
    return np.log((E + p_z) / (E - p_z)) / 2


@nb.jit
def find_azimuthal_angle(p_x, p_y):
    """Return the azimuthal angle between 0 and 2 pi given by `p_x` and `p_y`."""
    return np.atan2(p_y, p_x)


@nb.jit
def find_transverse_momentum(p_x, p_y):
    """Return the transverse momentum given by `p_x` and `p_y`."""
    return np.sqrt(p_x ** 2 + p_y ** 2)


def from_FourVector_list(fv_list: list[FourVector]) -> "FourVectorArray":
    """Return a FourVectorArray object from a list of FourVectorArrays."""
    n_fv = len(fv_list)
    E = nblist.empty_list(nb.float64, n_fv)
    p_x = nblist.empty_list(nb.float64, n_fv)
    p_y = nblist.empty_list(nb.float64, n_fv)
    p_z = nblist.empty_list(nb.float64, n_fv)
    for fv in fv_list:
        E.append(fv.cartesian_vector[0])
        p_x.append(fv.cartesian_vector[1])
        p_y.append(fv.cartesian_vector[2])
        p_z.append(fv.cartesian_vector[3])

    return FourVectorArray(E, p_x, p_y, p_z)
