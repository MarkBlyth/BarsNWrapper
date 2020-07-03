import numpy as np
import datetime
import subprocess
import os
from scipy.interpolate import BSpline

BARS_EXECUTABLE = "./BarsNWrapper/barsN.out"

PARAM_FILE_STRING = """
SET burn-in_iterations = {0}
SET sample_iterations = {1}
SET initial_number_of_knots = {2}
SET beta_iterations = 3
SET beta_threshhold = -10.0
SET proposal_parameter_tau = {3}
SET reversible_jump_constant_c = {4}
SET confidence_level = 0.95
SET number_of_grid_points = 150
SET sampled_knots_file = {8}
SET sampled_mu_file = none
SET sampled_mu-grid_file = none
SET sampled_params_file = none
SET summary_mu_file = {5}
SET summary_mu-grid_file = {6}
SET summary_params_file = none
SET Use_Logspline = false
SET verbose = true
{7}

END
"""


class ModelConfig:
    """Class for storing model configurations (number and location of
    knots, and data upper/lower bounds). Has a method for generating a
    callable model object from the config, when provided with training
    data.
    """
    def __init__(self, knot_locations, a, b):
        self.knotpoints = np.zeros(len(knot_locations) + 2)
        self.knotpoints[0] = a
        self.knotpoints[-1] = b
        self.knotpoints[1:-1] = knot_locations

    def fit(self, data_x, data_y):
        """Take x, y datapoints, and return a callable model object.

            data_x: 1d np array
                x data to fit the model to

            data_y: 1d np array
                y data to fit the model to

        Performs no checking! Returns a Model object, that can be
        called to evaluate the posterior data distribition, given this
        current knot configuration.
        """
        n_splines = len(self.knotpoints) - 4
        splines, cols = [], []
        # Build design matrix
        for i in range(n_splines):
            ith_spline = BSpline.basis_element(self.knotpoints[i:i+4])
            splines.append(ith_spline)  # Also store each spline for later
            cols.append(ith_spline(data_x))
        design_matrix = np.array(cols).T
        betas = (
            len(data_x)
            / (len(data_x) + 1)
            * np.linalg.lstsq(design_matrix, data_y, rcond=None)[0]
        )
        return Model(betas, splines)


class Model:
    """Simple callable B-Spline model"""

    def __init__(self, betas, spline_funcs):
        """To be called by ModelConfig.fit. betas are the BSpline
        coefficients, and spline_funcs are the BSPline functions."""
        self.betas = betas
        self.splines = spline_funcs

    def __call__(self, xs):
        """Evaluate the posterior distribution p(y | knots, x), at
        some x datapoints.

            xs: np array
                1d array of datapoints at which to evaluate the
                posterior distribution.

        Returns a np array of p(y | knots, x). """
        return np.dot(self.betas, np.array([b(xs) for b in self.splines]))


class ModelSet:
    """
    Class for handling collections of fitted spline models. Allows for
    the easy estimation of p(y | x), by averaging (mean, median) over
    the sets of samples p(y | knots, x) for each sampled knot set.
    """
    def __init__(self, model_list):
        self.models = model_list

    def posterior_samples(self, xs):
        """Find posterior p(y | knots, x) for each sampled knot set,
        for each x value. Each row is an evaluation p(y|x) for a
        different knot set"""
        return np.array([m(xs) for m in self.models])

    def posterior_median(self, xs):
        """Find the median value p(y | x)"""
        samples = self.posterior_samples(xs)
        return np.median(samples, axis=0)

    def posterior_mean(self, xs):
        """Find the mean value p(y | x)"""
        samples = self.posterior_samples(xs)
        return np.mean(samples, axis=0)

    def __call__(self, xs):
        """Find the mean value p(y|x). Takes the mean of p(y|x, knots)
        over the set of sampled knots.
            xs: np array
                1d array of points at which to evaluate the posterior

        Returns posterior mean p(y|x) at each x point.
        """
        return self.posterior_mean(xs)


def _random_filename(basename):
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = "_".join([basename, suffix])
    return filename


def _parse_knot_file(filename):
    with open(filename, "r") as infile:
        lines = infile.readlines()
    knotlist = []
    for line in lines:
        knotlist.append(np.array([float(x) for x in line.split()][2:]))
    return knotlist


def _build_model(knotlist, data_x, data_y):
    a = np.min(data_x)
    b = np.max(data_x)
    model_list = [ModelConfig(k, a, b).fit(data_x, data_y) for k in knotlist]
    return ModelSet(model_list)


def barsN(
    xs, ys, prior_param=(0, 60), iknots=25, burnin=2000, sims=2000, tau=50, c=0.4
):
    """
    Run a barsN executable on the provided data. This function
    performs no checking, so it is up to the user to ensure everything
    is entered correctly.

    xs : iterable of scalar float-likes
        The independent variable

    y : iterable of scalar float-likes
        A vector of the dependent variable

    iknots : int > 0
        Initial number of knots

    priorparam: float>0, tuple, or list of tuples
        The parameter for the prior belief on number of knots. The
        prior is one of

            uniform : uniform distribution over the number of knots;
            poisson : poisson distribution over the number of knots;
            user : custom distribution over the number of knots.

        The format of the priorparam is used to automatically
        determine which of these priors to use:

            a) if using Poisson, prior_param is a float lambda
               (poisson mean);
            b) if using Uniform, prior_param is a tuple of length 2,
                which gives the minimum number of knots, followed by
                the maximum number of knots. Lower and upper limits
                work the same way as in range(lower,upper).
            c) if using user-defined prior, prior_param is a list of
                tuples. The first entry should be the number of knots,
                and the second column should be the probability of
                obtaining this number of knots. Note the following
                example:

                   [
                    (2, 0.05),
                    (3, 0.15),
                    (4, 0.30),
                    (5, 0.30),
                    (6, 0.10),
                    (7, 0.10),
                   ]

        Default behaviour is a uniform prior with (0,60) knots.


burnin : int > 0
        The desired length of the burn-in for the MCMC chain (default
            = 200)

    sims : int > 0
        The number of simulations desired for the MCMC chain (default
            = 2000)

    tau : float > 0
        Parameter that controls the spread for the knot proposal
        distribution (default = 50.0)

    c : float > 0
        Parameter that controls the probability of birth and death
        candidates (default = 0.4)

    Returns a ModelSet object. Posterior p(y|x) can then be estimated
    by calling ModelSet(xs).
    """
    # Assume xs,ys are already lists of floats, of equal length
    data_tuples = list(zip(xs, ys))
    data_tuples.sort(key=lambda x: x[0])

    # Define the prior type based on the form of prior_param
    if isinstance(prior_param, (float, int)):
        # POISSON
        prior_pars = """SET prior_form = Poisson
SET Poisson_parameter_lambda = {0}""".format(
            float(prior_param)
        )

    elif isinstance(prior_param, (list, tuple, np.ndarray)) and len(prior_param) == 2:
        # UNIFORM
        prior_pars = """SET prior_form = Uniform
SET Uniform_parameter_L = {0}
SET Uniform_parameter_U = {1}""".format(
            prior_param[0] + 1, prior_param[1] - 1
        )

    elif (
        isinstance(prior_param, (list, tuple, np.ndarray)) and len(prior_param[0]) == 2
    ):
        # USER
        prior_pars = "SET prior_form = User"
        prior_string = "\n".join(["{0} {1}".format(n, p) for n, p in prior_param])

    else:
        raise ValueError("Could not infer prior type, check prior_param is correct")

    # Set up a prior file, if necessary
    prior_filename = None
    if "User" in prior_pars:
        prior_filename = _random_filename("prior")
        user_prior_file = open(prior_filename, "w")
        user_prior_file.write(prior_string)

    # Set up a datafile
    datafile_string = "\n".join(["{0} {1}".format(x, y) for x, y in data_tuples])
    datafile_filename = _random_filename("datafile")
    datafile_file = open(datafile_filename, "w")
    datafile_file.write(datafile_string)

    # Set up a parameter file
    sampled_mu_filename = _random_filename("sampled_mu")
    sampled_knots_filename = _random_filename("sampled_knots")
    gridded_mu_filename = _random_filename("gridded_mu")
    param_file_string = PARAM_FILE_STRING.format(
        burnin,
        sims,
        iknots,
        tau,
        c,
        sampled_mu_filename,
        gridded_mu_filename,
        prior_pars,
        sampled_knots_filename,
    )
    param_filename = _random_filename("param_file")
    param_file = open(param_filename, "w")
    param_file.write(param_file_string)

    # Close tempfiles
    if prior_filename is not None:
        user_prior_file.close()
    param_file.close()
    datafile_file.close()

    # Invoke BARS executable
    subproc_args = " ".join([BARS_EXECUTABLE, datafile_file.name, param_file.name])
    if prior_filename is not None:
        subproc_args.append(prior_filename)
    subprocess.run(subproc_args, shell=True)

    knotlist = _parse_knot_file(sampled_knots_filename)
    model = _build_model(knotlist, xs, ys)

    # Delete the relevant files
    os.remove(gridded_mu_filename)
    os.remove(sampled_knots_filename)
    os.remove(sampled_mu_filename)
    os.remove(datafile_filename)
    os.remove(param_filename)
    if prior_filename is not None:
        os.remove(prior_filename)

    return model
