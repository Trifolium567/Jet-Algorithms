# %%
import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import golden as PHI
# %%


@nb.jit
def find_maximum(lower_bound: float, upper_bound: float,
                 f, abs_error: float = 1e-15
    ) -> float:
    """Find the maximum of a unimodal function, `f`, in a provided range.

    This function uses the Golden Section method to find the maximum of a unimodal
    function, `f` within the interval [lower, upper].

    It successively decreases the interval

    from numerical methods in c (book)
    Include: process, termination condition

    Parameters
    ----------
    lower_bound : float
        The lower bound of the range to search over
    upper_bound : float
        The upper bound of the range to search over
    f : function
        A function of signature f(x: float) -> float
        This is the function whose maximum should be found by the function.
        If numba is used, this should be of type / signature:
        FunctionType[array(float64, 1d, A)(array(float64, 1d, A))]
        or f(x: float64[:]) -> float64[:]
    abs_error:
        The absolute error within which the x-value of the
        maximum should be.
    
    Returns
    -------
    f_max : float
        The maximum function

    References
    ----------
    This method is a modified the golden section minimum finding algorithm
    from "Numerical Methods in C, Third Edition" page 495.
    """
    if not abs_error:
        tolerance = 1e-15
    else:
        tolerance = np.sqrt(abs_error)

    _r = PHI - 1
    x_0 = lower_bound
    x_3 = upper_bound
    distance_from_bound = _r * (x_3 - x_0)
    x_1 = x_0 + _r * distance_from_bound
    x_2 = x_3 - _r * distance_from_bound

    x_array = np.array([x_0, x_1, x_2, x_3])
    f_array = f(x_array)
    index_of_maximum = np.argmax(f_array)

    outside_tolerance = True
    while outside_tolerance:
        if index_of_maximum < 2:
            x_3 = x_2
            x_2 = x_1
            distance_from_bound = _r * (x_3 - x_0)
            x_1 = x_0 + _r * distance_from_bound
        else:
            x_0 = x_1
            x_1 = x_2
            distance_from_bound = _r * (x_3 - x_0)
            x_2 = x_3 - _r * distance_from_bound

        x_array = np.array([x_0, x_1, x_2, x_3])
        f_array = f(x_array)
        index_of_maximum = np.argmax(f_array)
        
        outside_tolerance = np.abs(x_3 - x_0) > tolerance * (np.abs(x_2) + np.abs(x_1))
    return f_array[index_of_maximum]


vectorised_function_signature = nb.float64[:](nb.float64[:])
vectorised_function_type = vectorised_function_signature.as_type()

scalar_function_signature = nb.float64(nb.float64)
scalar_function_type = scalar_function_signature.as_type()

@nb.experimental.jitclass(  # ruff: ignore[blank-lines-top-level]
    spec=[("vectorised_distribution", vectorised_function_type),
          ("scalar_distribution", scalar_function_type)],
)
class Distribution1D:
    """Random Number Generator object from a given distribution.
    
    distribution must be in one variable.
    use rejection sampling
    take a numpy.random.Generator object as a parameter
    include signature of dist
    TODO: complete docstrings
    default tolerance is sqrt floating point precision of numpy
    """

    def __init__(self, dist):
        self.vectorised_distribution = dist
        self.scalar_distribution = dist

    def _get_maximum_P(
            self, lower: float, upper: float,
            absolute_error: float = 1e-15
        ) -> float:
        """Get the maximum of the probability density distribution provided.

        This method uses `find_maximum` to find the maximum of the probability density
        function that defines this class. The provided density function does not need to
        be normalised.
        
        Parameters
        ----------
        lower : float
            The lower bound of the range over which to find the maximum.
        upper : float
            The upper bound of the range over which to find the maximum probability
            density.
        absolute_error : float
            The absolute error within which x_hat (the parameter which is varied by this
            function) is from the true x-value which maximises the probability density
            function. By default, this is set to `1e-15`, which is the approximate
            resolution of the np.float64 dtype.
        
        Returns
        -------
        P_max : float
            The maximum value of the probability density function
        
        See Also
        --------
        find_maximum
        """
        return find_maximum(lower, upper, self.vectorised_distribution, absolute_error)

    def _rejection_sample_once(
            self, rng: np.random.Generator, lower: float,
            upper: float, P_max: float) -> float:
        """Compute one rejection sample from the probability distribution.
        
        Parameters
        ----------
        rng : numpy.random.Generator
            numpy.random.Generator object that is used to randomly generate the samples
        lower : float
            The lower bound of the range over which to randomly generate the samples
        upper : float
            The upper bound of the range over which to randomly generate the samples
        P_max : float
            The maximum probability over the range over which to generate the samples
        """
        while True:
            x = rng.uniform(lower, upper)
            P = self.scalar_distribution(x)
            _ = rng.uniform(0, P_max)
            if _ < P:
                return x
    
    def rejection_sample(
            self, rng: np.random.Generator, lower: float,
            upper: float, n: int = 1,
            absolute_error=1e-15
        ) -> np.typing.NDArray[np.float64]:
        """Generate `n` samples from the distribution.
        
        Parameters
        ----------
        rng : numpy.random.Generator
            numpy.random.Generator object that is used to randomly generate the samples
        lower : float
            The lower bound of the range over which to randomly generate the samples
        upper : float
            The upper bound of the range over which to randomly generate the samples
        n : int
            The number of samples to be generated. By default, this is set to `1`.
        absolute_error: float
            The absolute error within which x_hat (the parameter which is varied by this
            function) is from the true x-value which maximises the probability density
            function. By default, this is set to `1e-15`, which is the approximate
            resolution of the np.float64 dtype.
        """
        maximum_P = self._get_maximum_P(lower, upper, absolute_error)
        if n == 1:
            return np.array([self._rejection_sample_once(rng, lower, upper, maximum_P)])
        elif n > 1:
            value_array = np.empty(n)
            for i in nb.prange(n):
                value_array[i] = self._rejection_sample_once(
                    rng, lower, upper, maximum_P
                )
            return value_array
        else:
            raise ValueError(
                f"You have passed {n} to `n`. `n` must be an integer"
                " larger than or equal to 1"
            )


@nb.jit
def normaldist(x: float | np.typing.NDArray[np.float64],
               mu: float, variance: float
    ) -> float:
    """Return the probability density for a normal distribtion at `x`."""
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-((x - mu) ** 2) / (2 * variance))

@nb.jit  # ruff: ignore[blank-lines-top-level]
def sincsquared(x: float) -> float:
    """Return the value of the function sin^2(`x`) / `x`^2."""
    return np.sinc(x) ** 2

@nb.jit  # ruff: ignore[blank-lines-top-level]
def standard_normal(x: float) -> float:
    """Return the probability density of the standard normal distribution at `x`."""
    return normaldist(x, 0, 1)


# %%
rng = np.random.default_rng()
NormalDist = Distribution1D(standard_normal)
SincSqDist = Distribution1D(sincsquared)

# run once to force numba to compile
NormalDist.rejection_sample(rng, -2, 2)
SincSqDist.rejection_sample(rng, -3, 3)

# %%
# using for loops
number_of_samples = 1000000
standard_normal_samples = []
sincsquared_samples = []
for i in range(number_of_samples):
    standard_normal_samples.append(NormalDist.rejection_sample(rng, -2, 2)[0])
    sincsquared_samples.append(SincSqDist.rejection_sample(rng, -3, 3)[0])

standard_normal_samples = np.array(standard_normal_samples)
sincsquared_samples = np.array(sincsquared_samples)
# %%
# using purely the compiled class
std_normal_samples2 = NormalDist.rejection_sample(rng, -2, 2, number_of_samples)
snc_sq_samples2 = SincSqDist.rejection_sample(rng, -3, 3, number_of_samples)
# %%
plt.hist(standard_normal_samples, 50)
plt.show()
plt.hist(std_normal_samples2, 50)
plt.show()
plt.hist(sincsquared_samples, 50)
plt.show()
plt.hist(snc_sq_samples2, 50)
plt.show()
# %%
