import numba as nb
import numpy as np


@nb.experimental.jitclass([
    ("vector", nb.float64[:]),
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
        self.vector: np.typing.NDArray[np.float64] = np.array([E, p_x, p_y, p_z])
    
    def __abs__(self) -> float:
        """Return the magnitude of the four-vector, p^2.
        
        p^2 = E^2 - p_x^2 - p_y^2 - p_z^2
        """
        # builtin abs(FourVector) is faster than numpy.abs(FourVector)
        # as numpy has its own overheads and will not give any improvements
        return np.dot(self.vector, self.vector * np.array([1, -1, -1, -1]))

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
        return FourVector(*(self.vector + other.vector))
        
    # def __str__(self) -> str:  # useful for debugging
    #     """Represent the FourVector object as a string."""
    #     return f"FourVector({self.vector[0]}, {self.vector[1]}, " + \
    #            f"{self.vector[2]}, {self.vector[3]})"
    
    def __str__(self) -> str:  # for debugging - (done this way for numba compatibility)
        """Print the FourVector object."""
        print("FourVector(", self.vector[0], "\b,", self.vector[1], "\b,",
              self.vector[2], "\b,", self.vector[3], ")") 
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
        return FourVector(*np.matmul(
            np.array([
                [gamma, - gamma * beta_x, - gamma * beta_y, - gamma * beta_z],
                [-gamma * beta_x, 1 + alpha * beta_x**2, alpha * beta_x * beta_y,
                 alpha * beta_x * beta_z],
                [-gamma * beta_y, alpha * beta_y * beta_z, 1 + alpha * beta_y**2,
                 alpha * beta_y * beta_z],
                [-gamma * beta_z, alpha * beta_z * beta_x, alpha * beta_z * beta_y,
                 1 + alpha * beta_z**2]
            ]), self.vector
        ))
