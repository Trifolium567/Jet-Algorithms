import numba as nb
import numpy as np


@nb.experimental.jitclass([
    ("cartesian_vector", nb.float64[:]),
])
class FourVector:
    """Four-Vector object."""

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
        # order="C" make sure memory layout is contingious
        self.cartesian_vector = np.array([E, p_x, p_y, p_z])

    @property
    def p_T(self):
        """Transverse momentum of the four-vector."""
        return np.sqrt(self.cartesian_vector[1]**2 + self.cartesian_vector[2]**2)
    
    @property
    def azimuthal_angle(self):
        """Azimuthal angle of the four-vector."""
        return np.atan(self.p_T / self.cartesian_vector[2])

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

    def __add__(self, other: "FourVector") -> "FourVector":
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
        result = self.cartesian_vector + other.cartesian_vector
        return FourVector(result[0], result[1], result[2], result[3])
        
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
        ]) @ np.ascontiguousarray(self.cartesian_vector)
        return FourVector(result[0], result[1], result[2], result[3])


@nb.experimental.jitclass([
    ("cartesian_vectors", nb.float64[:, :])
])
class FourVectorArray:
    """Object to store and effectively operate on four-vectors."""

    cartesian_vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]]

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
        # TODO: add validation to make sure that all the parameters are arrays
        # TODO: make sure they are arrays of the same shape as well
        # length is done. add shape
        E = np.array(E_array, dtype=np.float64)
        p_x = np.array(p_x_array, dtype=np.float64)
        p_y = np.array(p_y_array, dtype=np.float64)
        p_z = np.array(p_z_array, dtype=np.float64)

        E_length = len(E)
        x_length = len(p_x)
        y_length = len(p_y)
        z_length = len(p_z)

        error_message = ""
        if not (E_length == x_length == y_length == z_length):
            error_message = f"len(E_array) = {E_length}, len(p_x_array) = {x_length},"+\
                f" len(p_y_array) = {y_length}, len(p_z_array) = {z_length}"  # ruff: ignore[missing-whitespace-around-arithmetic-operator]
        elif not (E.ndim == 1 and p_x.ndim == 1 and p_y.ndim == 1 and p_z.ndim == 1):
            error_message = f"E_array.ndim = {E.ndim}, p_x_array.ndim = {p_x.ndim}," + \
                f" p_y_array.ndim = {p_y.ndim}, p_z_array.ndim = {p_z.ndim}. " + \
                f"All inputs should be 1d."
        
        if error_message:
            raise ValueError(error_message)
        # else:
        container_array = np.zeros((4, E_length), dtype=np.float64)
        # for i in range(E_length):
        #     container_array[0, i] = E[i]
        #     container_array[1, i] = p_x[i]
        #     container_array[2, i] = p_y[i]
        #     container_array[3, i] = p_z[i]

        container_array[0] = E
        container_array[1] = p_x
        container_array[2] = p_y
        container_array[3] = p_z

        self.cartesian_vectors = np.ascontiguousarray(container_array)
    
    @property
    def p_T(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of transverse momentum of the four-vectors."""
        return np.sqrt(self.cartesian_vectors[1]**2 + self.cartesian_vectors[2]**2)
    
    @property
    def azimuthal_angle(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of azimuthal angles of four-vectors."""
        return np.atan(self.p_T / self.cartesian_vectors[2])

    @property
    def rapidity(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Array of rapidities of the four-vectors."""
        return np.log((self.cartesian_vectors[0] + self.cartesian_vectors[3]) 
                      / (self.cartesian_vectors[0] - self.cartesian_vectors[3]))

    def __abs__(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]: 
        """Return an array of the magnitudes of the four-vectors, p^2.

        Where p^2 = E^2 - p_x^2 - p_y^2 - p_z^2

        builtin abs(FourVector) is faster than numpy.abs(FourVector) as numpy has its
        own overheads and will not give any improvements.
        """
        return np.sum(
            self.cartesian_vectors ** 2 * np.array(((1,), (-1,), (-1,), (-1,))), axis=0,
        )

    def __add__(self, other) -> "FourVectorArray":
        # TODO: possibly add typing here?
        return FourVectorArray(*np.append(self.cartesian_vectors.T,
                                          other.cartesian_vectors.T).T)

    def _boost_result(
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
        self.cartesian_vectors = self._boost_result(beta_x, beta_y, beta_z)

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
        return FourVectorArray(*self._boost_result(beta_x, beta_y, beta_z))

    # IN ORDER:
    # TODO: define del
    # TODO: add getitem / indexing - if only 1 then give as FourVector object
    # TODO: add indexing with an array of bools - gives slice
    # np.selection might be useful above
    # TODO: add combine rows - change implementation of p_T, rapdidity, and azimuthal angle
    # a.T only makes views - it's okay to use them
    # TODO: add chunking? maybe later
    # TODO: add from list of four-vectors (only if needed)

# TODO: add type validation?
# TODO: add all docstrings
# TODO: configure ty to help with typing?
