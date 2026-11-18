import numba as nb
import numpy as np


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
        return np.sqrt(self.cartesian_vector[1]**2 + self.cartesian_vector[2]**2)
    
    @property
    def azimuthal_angle(self):
        """Azimuthal angle of the four-vector."""
        return np.atan(self.cartesian_vector[2] / self.cartesian_vector[1])

    @property
    def rapidity(self):
        """Rapidity of the four-vector."""
        return np.log((self.cartesian_vector[0] + self.cartesian_vector[3]) 
                      / (self.cartesian_vector[0] - self.cartesian_vector[3]))
        
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

    def __init__(self, E_array, p_x_array, p_y_array, p_z_array):
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
        self.p_T_array = self.get_p_T()
        self.azimuthal_angle_array = self.get_azimuthal_angle()
        self.rapidity_array = self.get_rapidity()

    def get_p_T(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of transverse momentum of the four-vectors."""
        return np.sqrt(self.cartesian_vectors[1]**2 + self.cartesian_vectors[2]**2)
    
    def get_azimuthal_angle(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of azimuthal angles of four-vectors."""
        return np.atan(self.cartesian_vectors[2] / self.cartesian_vectors[1])

    def get_rapidity(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of rapidities of the four-vectors."""
        return np.log((self.cartesian_vectors[0] + self.cartesian_vectors[3]) 
                      / (self.cartesian_vectors[0] - self.cartesian_vectors[3]))

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
        `[E_array, p_x_array, p_y_array, p_z_array]`
        """
        if isinstance(value, FourVector):
            self.cartesian_vectors.T[key] = value.cartesian_vector
        elif isinstance(value, FourVectorArray):
            self.cartesian_vectors.T[key] = value.cartesian_vectors.T
        else:
            self.cartesian_vectors.T[key] = value.T

    def __abs__(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]: 
        """Return an array of the magnitudes of the four-vectors, p^2.

        Where p^2 = E^2 - p_x^2 - p_y^2 - p_z^2

        builtin abs(FourVector) is faster than numpy.abs(FourVector) as numpy has its
        own overheads and will not give any improvements.
        """
        return np.sum(
            self.cartesian_vectors ** 2 * np.array(((1,), (-1,), (-1,), (-1,))), axis=0,
        )

    def __add__(self, other) -> "FourVectorArray | None":
        """Append a FourVectorArray or FourVector this one and return it."""
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
                "The operation '+' is only supported with"
                "FourVectors and FourVectorArrays."
            )

    def get_one_fourvector(self, index: int) -> FourVector:
        """Return a `FourVector` object from the vector stored at `index`."""
        vector = self.cartesian_vectors.T[index]
        return FourVector(vector[0], vector[1], vector[2], vector[3])

    def pop(self, index: int) -> FourVector:
        """Return the four-vector at `index` as a `FourVector` object and remove it."""
        value = self.get_one_fourvector(index)
        self.cartesian_vectors = np.append(self.cartesian_vectors.T[:index],
                                           self.cartesian_vectors.T[index + 1:],
                                           axis=0).T
        return value

    def combine(self, key1: int, key2: int) -> None:
        """Combine two vectors at indices `key1` and `key2`."""
        result = self.cartesian_vectors.T[key1] + self.cartesian_vectors.T[key2]
        self.cartesian_vectors.T[key1] = result
        self.cartesian_vectors = np.append(self.cartesian_vectors.T[:key2],
                                           self.cartesian_vectors.T[key2 + 1:],
                                           axis=0).T

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
        self.rapidity_array = self.get_rapidity()
        self.p_T_array = self.get_p_T()
        self.azimuthal_angle_array = self.get_azimuthal_angle()

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

    # IN ORDER:
    # TODO: add chunking? maybe later

# TODO: add type validation?
# TODO: add all docstrings
# TODO: configure ty to help with typing?
