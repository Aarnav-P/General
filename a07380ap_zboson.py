# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:09:29 2023

________________TITLE________________
PHYS20161 - Assignment 2 - Z boson
-------------------------------------
This python script reads in data from .csv files. It then
validates the imported data by:

1) checking for non-numerical values
2) checking for data points with zero uncertainty

and removing these points.
The script then uses global variables to optimise a curve of best fit with 
fmin for the given data on the Z_0 boson, and produces a plot. It applies an 
equation relating the decay products of the boson to the interaction 
cross-section.
Afterwards, the script calculates the uncertainties on the parameters used
for the curve of best fit, and returns:

1) the width of the Z_0 boson
2) the mass of the Z_0 boson
3) the lifetime of the Z_0 boson
4) the reduced chi_squared value of the optimised fit
and the uncertainty in the first 3, as well as two png's of the graphs.

Last Updated: 01/12/23
username: a07380ap
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc
import seaborn as sns


FILENAMES = ["z_boson_data_1.csv", "z_boson_data_2.csv"]
GAMMA_EE = 0.08391  # GeV
GEV_TO_NANOBARNS = 389400  # conversion factor
SHOW_ALL_ITERATION_VALUES = True
SHOW_ALL_PLOTS = True
GRAIN_M_Z = 400  # this defines how dense the linspace is for m_Z
# same for Gamma_Z, NOTE that > 500 will take several secs.
GRAIN_GAMMA_Z = 400
STEP_M_Z = 0.04
STEP_GAMMA_Z = 0.04
DESIRED_SIGMA = 4 # defines how many sigma away an outlier is


# note that the accuracy of the dots relies on not zooming out too far with
# too small a grain. As a rule of thumb, 100 points per 0.01 in step works ok.
# going too low risks that the intervals skip over the extrema and the
# program will miss it, instead plotting a point slightly further in. Use
# common sense by looking at the graph to see when the grain is appropriate.


def read_data(file):
    """
    Reads in data in csv format using genfromtxt. Delimiter can be changed for
    other file formats.
    Parameters
    ----------
    file : string
        name of file, given in FILENAME list
    Returns
    -------
    data : array
    Raises
    -------
        ValueError: If there are more columns in one row than the others
        FileNotFoundError: file not in local directory.

    """
    try:
        data = np.genfromtxt(file, comments='%', delimiter=',')
    except ValueError as error:
        print('ValueError:', error)
        return 1
    except FileNotFoundError as error:
        print('FileNotFoundError:', error)
        return 1
    except:
        print(
            f"An unknown error occured in the read_data() function of {file}")
        return 1

    return data, file


def validate_data(data_and_filename):
    """
    Checks data for nan values from genfromtxt and for 0 uncertainty values
    Parameters
    ----------
    data : Array

    Returns
    -------
    filtered_data : array
        data with nan and 0-uncertainty rows removed.
    Raises
    -------
    Exception: Unknown error. (usually wrong delimiter)
    """
    data = data_and_filename[0]
    filename = data_and_filename[1]
    # np.genfromtxt will find all non numeric values, so we check for this first.

    filtered_data = np.empty((0, 3))

    try:
        for row in data:
            erroneous_data = np.isnan(row)
            if any(erroneous_data):
                continue
            if row[2] == 0: #uncertainty of zero would cause divbyzero error
                continue
            filtered_data = np.vstack((filtered_data, row))
    except:
        print(f"Validation of data failed. Data passed was {str(filename)}")
        print("Please ensure the delimiter is correct for the filetype given.")
        return 1

    return filtered_data


def combine_and_sort_data(data_1, data_2):
    """
    Takes in two data files, combines and sorts them.
    
    Parameters
    ----------
    data_1 : Array
    data_2 : Array

    Returns
    -------
    combined_data: Array

    """
    combined_data = np.vstack((data_1, data_2))
    combined_data = combined_data[combined_data[:, 0].argsort()]
    return combined_data


def sigma_equation(variables, centre_of_mass_energy, mode):
    """
    Computes cross section using values for width and mass of the Z_0 boson.
    Converts from natural units into nanobarns
    Parameters
    ----------
    variables: array
        contains floats for desired mass and width in calculation
    mode: str
        changes process depending on whether mass/width is being passed as a 
        float or an array
    centre_of_mass_energy: array
        contains all measured centre of mass energies
    Returns
    -------
    sigma_nanobarns: array
        calculated (predicted) value for sigma for each combination.
    """
    m_z = variables[0]
    gamma_z = variables[1]
    dimensions = np.shape(m_z)
    if mode == "fitting":
        sigma_natural_units = ((12*np.pi*(centre_of_mass_energy**2)
                                *GAMMA_EE**2)/(m_z**2 * (
            (centre_of_mass_energy**2 - m_z**2)**2 + (m_z**2)*(gamma_z**2))))  # note this is in GeV
        sigma_nanobarns = sigma_natural_units * GEV_TO_NANOBARNS

    if mode == "contour":
        sigma_mesh_calc = np.zeros(
            (dimensions[0], dimensions[1], len(centre_of_mass_energy)))

        for i in range(dimensions[0]):
            # second element is the grain of G_Z
            for j in range(dimensions[1]):
                m_z_element = m_z[i][j]
                gamma_z_element = gamma_z[i][j]
                sigma_natural_units = ((12*np.pi*(centre_of_mass_energy**2)
                                        *GAMMA_EE**2)/(m_z_element**2 * (
                            (centre_of_mass_energy**2 - m_z_element**2)**2 +
                            (m_z_element**2)*(gamma_z_element**2))))
                # note this is in GeV
                sigma_mesh_calc[i][j] = sigma_natural_units
        sigma_nanobarns = sigma_mesh_calc * GEV_TO_NANOBARNS

    return sigma_nanobarns


def chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the chi squared value.

    Parameters
    ----------
    observation : array 
    observation_uncertainty : array 
    prediction : array 
    Returns
    -------
    float (chi_squared value)
    """
    return np.sum((observation - prediction)**2 / observation_uncertainty**2)


def plot(data, result):
    """
    Plots COM energy against measured cross-section with uncertainties,
    overlaying a line of best fit that uses the predicted sigma.

    Parameters
    ----------
    data : array
        measured data from experiment.
    result : tuple
        returned from fmin, contains optimised fit parameters and other info.
    Returns
    -------
    0
    """
    sns.set_style("darkgrid")
    fig = plt.figure()

    com_energies = data[:, 0]
    x_values = np.linspace(com_energies[0], com_energies[-1], 1000)
    # because the data is sorted, this makes a linspace from
    # the smallest value of observed sigma to the largest.
    axes = fig.add_subplot(111)
    axes.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o',
                markersize=3, label='Measured data',
                alpha=0.85, color='grey')
    axes.plot(x_values, sigma_equation(result[0], x_values, "fitting"),
            color='orangered', label='fitted curve')
    axes.set_xlabel('Centre-of-mass Energy (GeV)', fontsize=14)
    axes.set_ylabel(r'$\sigma$(nb)', fontsize=14)
    # axes.scatter(result[0][0], result[1], label='Maxima', color='b')
    plt.legend(loc='upper left', shadow=True, edgecolor='slategray')
    axes.set_title(r'$m_z$ '+ f'= {result[0][0]:4.3e}'
                 + r"$\, \mathrm{GeV/c^2}, \,$"
                 + r'$\Gamma_z$' + f' = {result[0][1]:4.3e}'
                 + r"$\, \mathrm{GeV}$")
    axes.grid('on')
    axes.spines['right'].set_color((.6, .6, .6))
    axes.spines['top'].set_color((.6, .6, .6))
    axes.spines['left'].set_color((0, 0, 0))
    axes.spines['bottom'].set_color((0, 0, 0))
    plt.savefig('z_boson.png', dpi=300)
    plt.show()
    plt.close()
    return 0


def remove_outliers(data, fit_parameters):
    """
    Finds outliers in data and removes them to return new data
    Parameters
    ----------
    data : array
    fit_parameters : array
        values for mass and width determined from fmin
    Returns
    -------
    new_data : array
        data without the outliers

    """
    difference = np.abs(data[:, 1] - sigma_equation(fit_parameters, data[:, 0],
                                                    "fitting"))
    criterion = DESIRED_SIGMA * data[:, 2]  # this is the uncertainty on measured sigma
    indices = np.where(difference/criterion <= 1)
    new_data = data[indices]
    return new_data


def optimise(data, start_parameters):
    """
    Minimises the chi-squared value for a line of best fit using mass and width
    as parameters. Returns a graph and full details of iteration if wanted.
    Parameters
    ----------
    data : array
    start_parameters: tuple
        contains the starting guesses for the fit parameters
    Returns
    -------
    result : tuple
        output from fmin containing relevant parameters and info.
    """
    chi_squared_initialised = lambda variables: chi_square(
        data[:, 1], data[:, 2],  sigma_equation(variables, data[:, 0],
                                                "fitting"))
    # this function restates the chi_square function so that the values of
    # m_z and gamma_z are the inputs that can be varies for minimisation.
    # They are packaged into a single tuple called variables in order to
    # work with the fmin function.
    m_z_start = start_parameters[0]
    gamma_z_start = start_parameters[1]
    result = fmin(chi_squared_initialised, (m_z_start,
                  gamma_z_start), full_output=True, disp=False)

    if SHOW_ALL_ITERATION_VALUES is True:
        print(f"\nMinimum value at m_z = {result[0][0]: .3f}, gamma_z = {result[0][1]: .3f}\n\
        Function value at minima = {result[1]: .3f}\n\
        Number of iterations = {result[2]:d}\n\
        Number of function calls = {result[3]:d}\n\
        Error Code = {result[4]:d}")
    if SHOW_ALL_PLOTS is True:
        plot(data, result)
    return result


def check_if_fully_optimised(new_result, counter, condition, results):
    """
    Collates parameters from fmin and compares to previous fit to see whether
    they are the same to 4s.f. Once they are, passes a variable as false to 
    signal this.
    Parameters
    ----------
    new_result : tuple
        output from fmin.
    counter : int
        keeps track of number of iterations of fitting process
    condition : boolean
        states whether fitting process should continue for another loop.
    results : array
        vertically stacked fit-parameters from fmin, first row is initial values.

    Returns
    -------
    condition : boolean
    counter :  int
    results : array
        this is done so that the array has "memory"
    """
    results = np.vstack((results, new_result[0]))

    if abs(results[counter][0] - results[counter+1][0]) < 0.01:
        if abs(results[counter][1] - results[counter+1][1]) < 0.001:
            condition = False
        else:
            counter += 1
    else:
        counter += 1

    return condition, counter, results


def optimisation_loop_until_fit(data, start_parameters):
    """
    Continues to iterate the optimisation process until suitable parameters
    have been found, that are identical to 4 significant figures.

    Parameters
    ----------
    data : array
    start_parameters: tuple
        contains the starting guesses for the fit parameters
    Returns
    -------
    new_result: tuple
        final fmin output, but with abs value of fit parameters.
    filtered_data: array
        measured data with outliers (according to final fit only) removed.
    """
    counter = 0
    condition = True
    results = np.array([start_parameters[0], start_parameters[1]])
    new_result = optimise(data, start_parameters)
    while condition is True:
        filtered_data = remove_outliers(data, new_result[0])
        new_result = optimise(filtered_data, start_parameters)
        condition, counter, results = check_if_fully_optimised(new_result,
                                                               counter,
                                                               condition,
                                                               results)
    print(f"Data fully optimised after {counter} iterations")
    print("Determined values to 4s.f. are:")
    print(f"m_Z = {abs(results[counter+1][0]):.3e} GeV/c^2")
    print(f"gamma_z = {abs(results[counter+1][1]):.3e} GeV^-2")
    print(f"Minimised chi-squared value is {new_result[1]:.3f}")

    reduced_chi_squared = new_result[1]/(len(filtered_data[:, 0]) - 2)
    print(f"Minimised reduced chi-squared value is {reduced_chi_squared:.3f}")
    # convert GeV to eV
    tau_z = pc.hbar / (new_result[0][1] * (pc.eV*10**(9)))
    print(f"Lifetime of the Z boson is {abs(tau_z):.3e}s")

    new_result = (np.array([abs(new_result[0][0]),abs(new_result[0][1])]),
                  new_result[1], new_result[2],new_result[3],new_result[4])

    if SHOW_ALL_PLOTS is False:
        plot(filtered_data, new_result)
    # lines are added in to ensure that is a negative value is converged on
    # the positive value will be returned instead, as the values are >0 and
    # the cross section equation has symmetrical minima.

    return new_result, filtered_data


def contourplot(parameters, data):
    """
    Creates a range of values over which to plot a grid of points (determined
    by global variables). Makes a contour plot using each point's chi-squared
    value, and uses it to plot the uncertainties in the fit parameters.

    Parameters
    ----------
    parameters : tuple
        (final) output from fmin
    data : array
        (should be filtered data from last fit for useful results)
    Returns
    -------
    0
    """
    m_z_index = np.linspace(parameters[0][0] + STEP_M_Z,
                             parameters[0][0] - STEP_M_Z,
                             num=GRAIN_M_Z)
    gamma_z_values = np.linspace(parameters[0][1] + STEP_GAMMA_Z,
                                 parameters[0][1] - STEP_GAMMA_Z,
                                 num=GRAIN_GAMMA_Z)

    x_mesh, y_mesh = np.meshgrid(m_z_index, gamma_z_values)
    sigma_mesh_values = sigma_equation((x_mesh, y_mesh), data[:, 0], "contour")
    # recall that the grain effectively defines the dimensions of the array
    chi_squared_mesh_calc = np.zeros((GRAIN_GAMMA_Z, GRAIN_M_Z))
    # note dims are swapped due to meshgrid
    for i in range(GRAIN_GAMMA_Z):
        for j in range(GRAIN_M_Z):
            chi_squared_mesh_calc[i][j] = chi_square(data[:, 1],
                                                     data[:, 2],
                                                     sigma_mesh_values[i][j])

    fig = plt.figure(figsize=(6, 5), facecolor='white',
                     edgecolor='white', linewidth=4)
    axes = fig.add_subplot(111)

    plotted_lines = [parameters[1]+1, parameters[1]+2.30, parameters[1]+5.99,
                     parameters[1]+9.21]

    line_labels = {
        parameters[1]+1: r"$\chi_{min}^2 + 1$",
        parameters[1]+2.30: r"$\chi_{min}^2$ + 2.30",
        parameters[1]+5.99: r"$\chi_{min}^2$ + 5.99",
        parameters[1]+9.21: r"$\chi_{min}^2$ + 9.21"}

    contour_plot = axes.contour(x_mesh, y_mesh,
                              chi_squared_mesh_calc,
                              plotted_lines, linestyles='dashdot',
                              cmap = 'copper')
    # 20, cmap = 'RdGy'
    axes.scatter(parameters[0][0], parameters[0][1],
               label=r"$\chi_{min}^2 = $" + f"{parameters[1]:.3f}",
               color='r')
    axes.grid(False)
    axes.clabel(contour_plot, fontsize=11, colors='black', fmt=line_labels,
              inline_spacing=5)
    bounds = [parameters[0][0] - STEP_M_Z, parameters[0][0] + STEP_M_Z,
              parameters[0][1] - STEP_GAMMA_Z, parameters[0][1] + STEP_GAMMA_Z]
    colourscale = axes.imshow(chi_squared_mesh_calc, extent=bounds,
                            origin='lower', cmap='Spectral_r', alpha=0.9)
    # vlag # Spectral_r other good options for colouring
    fig.colorbar(colourscale)
    axes.set_xlabel(r"$m_Z \, \mathrm{(GeV/c^2)}$", fontsize=14)
    axes.set_ylabel(r"$\Gamma_Z \, \mathrm{(GeV )}$", fontsize=14)
    axes.set_title(r"Width against mass of the $Z^0$ boson", fontsize=16)

    gamma_z_uncertainty, gamma_z_index = find_parameter_uncertainties(chi_squared_mesh_calc,
                                                                      parameters[1]+1,
                                                                      parameters[0][0],
                                                                      'x', x_mesh)
    m_z_uncertainty, m_z_index = find_parameter_uncertainties(chi_squared_mesh_calc,
                                                              parameters[1]+1,
                                                              parameters[0][1],
                                                              'y', y_mesh)

    print(f"The uncertainty in Gamma_Z is +- {gamma_z_uncertainty:.3f} GeV^-2")
    print(f"The uncertainty in m_Z is +- {m_z_uncertainty:.2f} GeV/c^2")

    tau_z_uncertainty = ((gamma_z_uncertainty * (pc.hbar/ (parameters[0][1] * pc.eV*10**9))/
                         parameters[0][1]))
    print("The uncertainty in the lifetime of the Z boson is "
          + f"{tau_z_uncertainty:.1e}s")

    # note the indices are stored in the standard format of row, column.
    # Therefore to access the x-value, which are the same along each column,
    # I take a given row, then use the SECOND index to pick the column.
    axes.scatter(x_mesh[0, gamma_z_index[0][1]], y_mesh[gamma_z_index[0][0], 0],
               label=r'$\Gamma_Z$ extrema', color='b')
    axes.scatter(x_mesh[0, gamma_z_index[1][1]], y_mesh[gamma_z_index[1][0], 0],
               color='b')
    axes.scatter(x_mesh[0, m_z_index[0][1]], y_mesh[m_z_index[0][0], 0],
               label=r'$m_Z$ extrema', color='w')
    axes.scatter(x_mesh[0, m_z_index[1][1]], y_mesh[m_z_index[1][0], 0],
               color='w')
    axes.legend(loc='upper left', shadow=True, edgecolor='saddlebrown')
    plt.savefig('chi_squared_contours.png', dpi=300)
    plt.show()
    plt.close()

    return 0


def find_parameter_uncertainties(plotted_points, value_of_contour,
                                 uncertainty_starting_value, axis, mesh):
    """
    
    Finds the highest and lowest values along the specified axis
    for plotted points that have the value of the contour specified by 
    desired_value. Precision is automatically 4s.f., but accuracy depends on 
    global variables being appropriately chosen.
    Parameters
    ----------
    plotted_points : array
        array of all points' chi_squared value on the contour plot's grid
    value_of_contour : float
        value of the contour on which to find the extrema
    uncertainty_starting_value : float
        The value at the minimum on that axis. Used to set the tolerance/precision.
    axis : str
        determines whether the function scans horizontally or vertically.
    mesh : array
        array of the x/y (as specified) value for each point on the grid.

    Returns
    -------
    uncertainty : float
        difference between the two extrema halved, uncertainty on that parameter.
    extrema : array
        contains the indexes of the extrema points, so they can be plotted.

    """
    # now we attempt to find the extrema using the plotted data
    extreme_value_found = False
    index = 0
    tolerance = 0.0005 * 10**math.floor(
        math.log(abs(uncertainty_starting_value), 10))  # finds uncertainties to 4sf

    if axis == 'x':

        while extreme_value_found is False:
            indices = np.where(abs(plotted_points[:, index] - value_of_contour)
                               <= tolerance)
            if len(indices[0]) > 0:
                extreme_value_found = True
                break
            index += 1

        max_value = mesh[0][index]
        index_max = [math.floor(np.average(indices[0])), index]
        index = 0
        extreme_value_found = False

        while extreme_value_found is False:
            indices = np.where(abs(plotted_points[:, -index] - value_of_contour)
                               <= tolerance)
            if len(indices[0]) > 0:
                extreme_value_found = True
                break
            index += 1

        min_value = mesh[0][-index]
        uncertainty = abs(max_value-min_value)/2
        index_min = [math.floor(np.average(indices[0])), -index]
        extrema = np.vstack((index_max, index_min))

    if axis == 'y':
        while extreme_value_found is False:
            indices = np.where(abs(plotted_points[index] - value_of_contour)
                               <= tolerance)

            if len(indices[0]) > 0:
                extreme_value_found = True
                break
            index += 1
        max_value = mesh[index][0]
        index_max = [index, math.floor(np.average(indices[0]))]
        index = 0
        extreme_value_found = False

        while extreme_value_found is False:
            indices = np.where(abs(plotted_points[-index] - value_of_contour)
                               <= tolerance)
            if len(indices[0]) > 0:
                extreme_value_found = True
                break
            index += 1
        min_value = mesh[-index][0]
        uncertainty = abs(max_value-min_value)/2
        index_min = [-index, math.floor(np.average(indices[0]))]
        extrema = np.vstack((index_max, index_min))
    return uncertainty, extrema


def main():
    """
    The main function here is split into a few main sections that subdivide
    the code into its main tasks: 
    
    1) data input&validation
    2) data fitting and fit optimisation
    3) Finding parameter uncertainties
    
    Functions are bundled in this way so that if something goes wrong it is 
    clear which section to check, each of which functions as its own smaller
    main(). E.g. if the first image or the parameter values throw errors, 
    check section 2; if the uncertainties are nonsensical, check section 3.
    
    It is also useful as it avoids the need to return a lot of variables and 
    to pass them as arguments to lots and lots of smaller functions. By 
    keeping these local environments open, a lot of clutter is saved.
    Returns
    -------
    0
    """
    filtered_data_1 = validate_data(read_data(FILENAMES[0]))
    filtered_data_2 = validate_data(read_data(FILENAMES[1]))

    if isinstance(filtered_data_1, int):
        return 1
    if isinstance(filtered_data_2, int):
        return 1
    
    combined_data = combine_and_sort_data(filtered_data_1, filtered_data_2)
    # the data has now been read in, validated, combined and sorted.
    # note: combined_data[0] is Centre-of-Mass energy in GeV
    # combined_data[1] is cross section in nanobarns, uncertainty (nb)
    # combined_data[2] is uncertainty on the cross section in nanobarns

    start_parameters = (np.average(combined_data[:,0]),np.average(combined_data[:,1]))
    # this section optimises and fits the code, then plots the final result
    # according to preferences stated by global variables
    fit_parameters, new_data = optimisation_loop_until_fit(combined_data, start_parameters)

    # this section makes a contour plot to examine the chi-squared variation
    # with a mesh of parameters

    # It also finds the uncertainties of those parameters and returns them,
    # which is included here to prevent the need to returning and passing
    # lots of variables unnecessarily when they are already
    # in the local environment.
    contourplot(fit_parameters, new_data)

    # to find the uncertainties, simply create a meshgrid of parameters +-
    # a bit, and then ax.contour them with an array of desired chi^2 values
    # Then extract from the plot the correct data points
    # EASIER SAID THAN DONE LMAO
    # I'm a genius.
    # Geniuses would've finished this assignment sooner.
    # And with fewer lines of code, jesus.

    return 0


if __name__ == "__main__":
    main()
