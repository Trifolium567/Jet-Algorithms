# %%
import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import golden as PHI

rng = np.random.default_rng()
# %%


@nb.jit
def find_maximum(lower_bound: float, upper_bound: float,
                 f, abs_error: float = 1e-15
    ) -> float:
    r"""Find the maximum of a unimodal function, `f`, in a provided range.

    This function uses the Golden Section method to find the maximum of a unimodal
    function, `f` within the interval [lower, upper]. It searches by probing an interval
    at two points, each `(1 - \phi) * length_of_interval` away from each bound to get
    the maximum values of each integral, then it narrows the interval depending on which
    of the probed values and the interval bounds was the maximum. The termination
    condition is when the distance between the two probed values reaches
    `sqrt(absolute_error) * length_of_interval`.

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

@nb.experimental.jitclass([
    ("_function_vectorised", nb.float64[:](nb.float64[:], nb.float64[:]).as_type()),
    ("_function_scalar", nb.float64(nb.float64, nb.float64).as_type()),
])
class MCIntegrator2D:
    """Monte Carlo Integrator Object over a provided function."""

    def __init__(self, f):
        """Create MCIntegrator2D object instance.
        
        Parameters
        ----------
        f : function
            This should be in the signature f(x: float, y: float) -> float. It should be
            compatible with numpy arrays.
        """
        self._function_vectorised = f
        self._function_scalar = f

    def _generate_n_values(
            self, lower_x: float, upper_x: float,
            lower_y: float, upper_y: float, rng, n: int = 1000
        ) -> np.typing.NDArray[np.float64]:
        x_values = rng.uniform(lower_x, upper_x, size=n)
        y_values = rng.uniform(lower_y, upper_y, size=n)
        f_values = self._function_vectorised(x_values, y_values)
        return f_values

    def integrate(self, lower_x: float, upper_x: float,
                  lower_y: float, upper_y: float,
                  rng: np.random.Generator, n: int = 1000
        ) -> float:
        """Monte Carlo integrate a provided function `f(x, y)` over a given range.

        Parameters
        ----------
        lower_x : float
            lower bound of x variable.
        upper_x : float
            upper bound of x variable.
        lower_y : float
            lower bound of y variable.
        upper_y : float
            upper bound of y variable.
        rng : numpy.random.Generator
            The random number generator object that is used for Monte-Carlo Integration
        f : float
            function of signature `f(x: float, y: float) -> float`. Should be compatible
            with arrays.
        n : int
            The number of values to use in the Monte-Carlo integration, by default 1000.

        Returns
        -------
        float
            The result of the monte carlo integrator

        Raises
        ------
        ValueError
            if lower_x > upper_x or lower_y > upper_y
        """
        if lower_x >= upper_x and lower_y >= upper_y:
            raise ValueError(f"lower_y = {lower_y}, upper_y = {upper_y}. upper_y must"
                             f" be > lower_y. lower_x = {lower_x}, upper_x = {upper_x}."
                             f" upper_x must be > lower_x.")
        elif lower_y >= upper_y:
            raise ValueError(f"lower_y = {lower_y}, upper_y = {upper_y}. upper_y must"
                              " be > lower_y")
        elif lower_x >= upper_x:
            raise ValueError(f"lower_x = {lower_x}, upper_x = {upper_x}. upper_x must"
                              " be > lower_x")

        x_range = upper_x - lower_x
        y_range = upper_y - lower_y
        f_values = self._generate_n_values(lower_x, upper_x, lower_y, upper_y, rng, n)
        return np.mean(f_values) * x_range * y_range

    def integrate_within_error(
            self, lower_x: float, upper_x: float,
            lower_y: float, upper_y: float,
            rng: np.random.Generator, abs_error: float = 0.01
        ) -> float:
        """Monte Carlo integrate such that the result's error is less than `abs_error`.

        Parameters
        ----------
        lower_x : float
            lower bound of x variable.
        upper_x : float
            upper bound of x variable.
        lower_y : float
            lower bound of y variable.
        upper_y : float
            upper bound of y variable.
        rng : numpy.random.Generator
            The random number generator used to do Monte-Carlo Integration
        abs_error : float
            the absolute error within which the result's standard deviation must be.

        Returns
        -------
        float
            The result of the integral.

        Raises
        ------
        ValueError
            if lower_y > upper_y or lower_x > upper_y
        """
        if lower_x >= upper_x and lower_y >= upper_y:
            raise ValueError(f"lower_y = {lower_y}, upper_y = {upper_y}. upper_y must"
                                f" be > lower_y. lower_x = {lower_x}, upper_x = "
                                f"{upper_x}. upper_x must be > lower_x.")
        elif lower_y >= upper_y: 
            raise ValueError(f"lower_y = {lower_y}, upper_y = {upper_y}. upper_y must"
                                " be > lower_y")
        elif lower_x >= upper_x:
            raise ValueError(f"lower_x = {lower_x}, upper_x = {upper_x}. upper_x must"
                              " be > lower_x")
        x_range = upper_x - lower_x
        y_range = upper_y - lower_y
        # compute a minimum of 1000 times
        N = 1000
        f_values = self._generate_n_values(lower_x, upper_x, lower_y,
                                           upper_y, rng, N)

        sample_variance = np.sum((f_values - np.mean(f_values))**2) / (N - 1)
        sum_f = np.sum(f_values)
        sum_f_squared = np.sum(f_values ** 2)

        # sigma(integral) ~ x_range * y_range * sigma(f) / sqrt(N)
        while x_range * y_range * np.sqrt(sample_variance / N) > abs_error:
            x_value = rng.uniform(lower_x, upper_x)
            y_value = rng.uniform(lower_y, upper_y)
            # size=1 is put in these arrays to make them compatible with numba's strong
            # typing.
            f_value = self._function_scalar(x_value, y_value)
            sum_f += f_value
            sum_f_squared += f_value ** 2
            N += 1
            sample_variance = sum_f_squared / (N - 1) - sum_f**2 / N / (N - 1)
        return (sum_f / N) * x_range * y_range

# %%
@nb.jit
def ellipse_indicator(x: float, y: float, a: float, b: float) -> float:
    """Indicate wether or not an x-y coordinate is in an ellipse."""
    # done this way to work with numba
    return 0.5 * (np.sign(1 - (x / a)**2 - (y / b)**2) + 1)

@nb.jit
def ellipse_indicator_specified(x: float, y: float) -> float:
    """Indicate wether or not a coordinate is in an ellipses of specified semi-minor and major axes."""
    return ellipse_indicator(x, y, 2, 1)


# %%
specified_ellipse = MCIntegrator2D(ellipse_indicator_specified)
print(specified_ellipse.integrate(-2, 2, -1, 1, rng))
print(specified_ellipse.integrate(-3, 3, -2, 2, rng))
print(specified_ellipse.integrate_within_error(-3, 3, -2, 2, rng, 0.01))

# %%

@nb.jit
def ellipse_indicator_specified2(x: float, y: float):
    return ellipse_indicator(x, y, 5, 3)

specified_ellipse_2 = MCIntegrator2D(ellipse_indicator_specified2)  # ruff: ignore[blank-lines-after-function-or-class]
print(specified_ellipse_2.integrate_within_error(-6, 6, -4, 4, rng, 0.001))
