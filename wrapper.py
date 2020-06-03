import numpy as np
import datetime
import subprocess
import os

BARS_EXECUTABLE = "./barsN.out"

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
SET sampled_knots_file = none
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


def _random_filename(basename):
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = "_".join([basename, suffix])
    return filename


def _parse_summary_file(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()
    ts = [float(x) for x in lines[0].split()]
    xs = [float(x) for x in lines[1].split()]
    return ts, xs


def barsN(xs, ys, prior_param=(0, 60), iknots=25, burnin=2000, sims=2000, tau=50, c=0.4):
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

    Returns four lists:
        1. mu_t_samp : training t points
        2. mu_x_samp : latent process mean, evaluated at the training
           t points
        3. mu_t_grid : t points along a uniform grid
        4. mu_x_grid : latent process mean, evaluated at that uniform
           grid
    """
    # Assume xs,ys are already lists of floats, of equal length
    data_tuples = list(zip(xs, ys))
    data_tuples.sort(key=lambda x: x[0])

    # Define the prior type based on the form of prior_param
    if isinstance(prior_param, (float, int)):
        # POISSON
        prior_pars = """SET prior_form = Poisson
SET Poisson_parameter_lambda = {0}""".format(float(prior_param))

    elif isinstance(prior_param, (list, tuple, np.ndarray)) and len(prior_param) == 2:
        # UNIFORM
        prior_pars = """SET prior_form = Uniform
SET Uniform_parameter_L = {0}
SET Uniform_parameter_U = {1}""".format(prior_param[0]+1, prior_param[1]-1)

    elif isinstance(prior_param, (list, tuple, np.ndarray)) and len(prior_param[0]) == 2:
        # USER
        prior_pars = "SET prior_form = User"
        prior_string = "\n".join(["{0} {1}".format(n, p)
                                  for n, p in prior_param])

    else:
        raise ValueError(
            "Could not infer prior type, check prior_param is correct")

    # Set up a prior file, if necessary
    prior_filename = None
    if "User" in prior_pars:
        prior_filename = _random_filename("prior")
        user_prior_file = open(prior_filename, "w")
        user_prior_file.write(prior_string)

    # Set up a datafile
    datafile_string = "\n".join(["{0} {1}".format(x, y)
                                 for x, y in data_tuples])
    datafile_filename = _random_filename("datafile")
    datafile_file = open(datafile_filename, "w")
    datafile_file.write(datafile_string)

    # Set up a parameter file
    sampled_mu_filename = _random_filename("sampled_mu")
    gridded_mu_filename = _random_filename("gridded_mu")
    param_file_string = PARAM_FILE_STRING.format(
        burnin, sims, iknots, tau, c, sampled_mu_filename, gridded_mu_filename, prior_pars)
    param_filename = _random_filename("param_file")
    param_file = open(param_filename, "w")
    param_file.write(param_file_string)

    # Close tempfiles
    if prior_filename is not None:
        user_prior_file.close()
    param_file.close()
    datafile_file.close()

    # Invoke BARS executable
    subproc_args = ' '.join(
        [BARS_EXECUTABLE, datafile_file.name, param_file.name])
    if prior_filename is not None:
        subproc_args.append(prior_filename)
    subprocess.run(subproc_args, shell=True)

    # Parse the output
    mu_t_grid, mu_x_grid = _parse_summary_file(gridded_mu_filename)
    mu_t_samp, mu_x_samp = _parse_summary_file(sampled_mu_filename)

    # Delete the relevant files
    os.remove(gridded_mu_filename)
    os.remove(sampled_mu_filename)
    os.remove(datafile_filename)
    os.remove(param_filename)
    if prior_filename is not None:
        os.remove(prior_filename)

    return mu_t_samp, mu_x_samp, mu_t_grid, mu_x_grid
