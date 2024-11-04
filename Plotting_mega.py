# Import Files
FILES = ['lambert_law.csv'] 
SKIPHEADER = 1 # 1 for yes, 0 for no
# Set Data Limits in xdata
TRUNCATE_DATA_X = False
DATA_X_RANGE_LOW = 2*(10**9)
DATA_X_RANGE_HIGH = 1*(10**10)

TRUNCATE_DATA_Y = False
DATA_Y_RANGE_LOW = -0.05
DATA_Y_RANGE_HIGH = 2

# Perform Analysis
# TODO: make it so this program can be imported into another script where the analysis is
# done separately, and the relevant parameters are exported and written into a file
ANALYSE_FIT = True

# Plot specification
PLOT_NAME = 'lambert_law'
NAME_OF_FIGURE_DATA = 'lambert_law' # Saves as
MARKERSIZE = 5

PLOT_X_LIMITS = False
PLOT_Y_LIMITS = False
X_LIM_LOWER = 9
X_LIM_HIGHER = 10
Y_LIM_LOWER = 0.001
Y_LIM_HIGHER = 0.002

X_LABEL = "Measured Voltage"
Y_LABEL = "$V_0 \cos θ$"

# Nature of the data
LINEAR = True # will automatically attempt fit; will error unless true linear

# Non linear fits
ATTEMPT_FIT = False # Fitting can be attempted with two parameters; Write function() first
PARAMETER_ESTIMATE = [80, 5]

# Find peaks/troughs
FIND_PEAKS = False
FIND_TROUGHS = False
PROMINENCE = 0.07

# Note about data formatting:
# For files of dimension 2 - x, y
# For files of dimension 3 - x, y, y_uncertainties
# For files of dimension 4 - x, y, x_uncertainties, y_uncertainties

### IMPORTS

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.optimize as sc
from scipy import odr
from scipy.signal import find_peaks
from scipy.odr import *
from scipy.optimize import curve_fit


### FUNCTIONS ###

def clean_data(data, file_dimension):
    """
    Clean up data by removing NaN values and duplicates.

    Args:
        data (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Cleaned data array.
    """
    #Remove text and nans
    data = data[~np.isnan(data).any(axis=1)]
    
    # Remove duplicates
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    data = data[unique_indices]

    # Sort data from lowest energy to highest energy
    data = data[np.lexsort((data[:, 0],))]
    
    if file_dimension == 3:
        zero_index = np.where((data[:,1] == 0) | (data[:,2] <= 0))[0]
        data = np.delete(data, zero_index, 0)
        
    elif file_dimension == 4:
        zero_index = np.where((data[:, 2] <= 0) | (data[:, 3] <= 0))[0]
        data = np.delete(data, zero_index, 0)
    
    
    
    return data


def read_and_check_files(validated_files):
    
    try:
        
        input_file = np.genfromtxt(validated_files[0], delimiter = ',', skip_header = SKIPHEADER)
    except FileNotFoundError:
        print(f"Error: the file {validated_files[0]} could not be found. Check spelling")
        sys.exit()
    
    
    if input_file.shape[1] == 2:
        # Handle (x, y) format
        file_dimension = 2
        data = np.zeros([0,2])
        

    elif input_file.shape[1] == 3:
        # Handle (x, y, uncertainty in y) format
        file_dimension = 3
        data = np.zeros([0,3])
        
        zero_index = np.where(data[:,1] <= 0)[0]
        data = np.delete(data, zero_index, 0)

    
    elif input_file.shape[1] == 4:
        # Handle (x, y, uncertainty in x, uncertainty in y) format
        file_dimension = 4
        data = np.zeros([0,4])
        

    else:
        print("File shape is not recognised.")

    for file_name in validated_files:
        try:
            # Read data from file
            input_file = np.genfromtxt(file_name, delimiter = ',',
                                       skip_header = 0)
            data = np.vstack((data, input_file))

        except FileNotFoundError:
            print(f"Error: The file {file_name} could not be found. Check spelling\n",
                  "Exiting...")
            sys.exit()
        except IndexError:
            print(f"Error: The file {file_name} could not be read. Exiting...")
            sys.exit()
        
            print(len(data))

    
    return data, file_dimension


def truncate(data):
    """
    Truncates data based on a specified range of x values.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D numpy array containing data, where the first column represents x values.

    Returns:
    --------
    numpy.ndarray
        A truncated version of the input data containing only rows where the x values 
        fall within the range [-1.5, 0].

    """

    if TRUNCATE_DATA_X:
        
        # Determine indices of rows where x values fall within the specified range
        indices_to_keep = np.logical_and(data[:, 0] >= DATA_X_RANGE_LOW, data[:, 0] <= DATA_X_RANGE_HIGH)

        # Truncate the data based on the determined indices
        data = data[indices_to_keep]
        
    if TRUNCATE_DATA_Y:
        
        indices_to_keep = np.logical_and(data[:, 1] >= DATA_Y_RANGE_LOW, data[:, 1] <= DATA_Y_RANGE_HIGH)
        
        data = data[indices_to_keep]

    return data



def plot_data(data, fit_values):
    
    # Plot data points
    if LINEAR:
        fig, (ax, ax_residuals) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax.errorbar(data[:, 0], data[:, 1], fmt='x', xerr = X_ERR, yerr = Y_ERR, markersize = MARKERSIZE, label="Data ", color='#6c5b7b')
    else:
        fig, ax = plt.subplots(1,1)
        ax.plot(data[:,0], data[:,1], markersize = MARKERSIZE, marker = 'x', label = "Data", color = 'blue')

    
    if np.any(fit_values) and not LINEAR:
        ax.plot(data[:,0], fit_values, label = "Fit function", color = '#355c7d')
    else:
        None

    # Set axis labels
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # Add legend
    ax.legend(loc='lower right', shadow=True)

    # Set title
    ax.set_title(PLOT_NAME, fontweight='bold')

    # Enable grid
    ax.grid(True, linestyle=(0, (3, 5, 1, 5)), linewidth=0.5, color='grey')

    # Set axis limit
    if PLOT_X_LIMITS:
        ax.set_xlim(X_LIM_LOWER, X_LIM_HIGHER)
    if PLOT_Y_LIMITS:
        ax.set_ylim(Y_LIM_LOWER, Y_LIM_HIGHER)

    if LINEAR and (file_dimension == 3 or file_dimension == 4):
        fig, (ax, ax_residuals) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

        if PLOT_X_LIMITS:
            ax.set_xlim(X_LIM_LOWER, X_LIM_HIGHER)
        if PLOT_Y_LIMITS:
            ax.set_ylim(Y_LIM_LOWER, Y_LIM_HIGHER)



        ax.errorbar(data[:, 0], data[:, 1], fmt='x', xerr = X_ERR, yerr = Y_ERR, markersize = MARKERSIZE, label="Data values", color='#6c5b7b')
        ax.plot(data[:,0], fit_values, label = "Fit function", color = '#355c7d')
        # Set axis labels
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)

        # Add legend
        ax.legend(loc='lower right', shadow=True)

        # Set title
        ax.set_title(PLOT_NAME, fontweight='bold')

        # Enable grid
        ax.grid(True, linestyle=(0, (3, 5, 1, 5)), linewidth=0.5, color='grey')
        
        
        residuals = data[:,1] - fit_values  # Compute residuals
        ax_residuals.errorbar(data[:,0], residuals, yerr=data[:,3], fmt='x', color='black')
        ax_residuals.plot(data[:,0], 0 * data[:,0], color='red')
        ax_residuals.grid(True)
        ax_residuals.set_title('Residuals', fontsize=14)
    
        # Adjust spacing between subplots
        fig.subplots_adjust(hspace=0.3)
        fig.savefig(NAME_OF_FIGURE_DATA, dpi = 300)
        print("print")
        plt.show()
        print("display")
    



    # Save and show plot
    fig.savefig(NAME_OF_FIGURE_DATA, dpi=300)
    plt.show()


def chi_squared_function(x_data, y_data, y_uncertainties, parameters):
    """Calculates the chi squared for the data given, assuming a linear
    relationship.
    Args:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
        parameters: numpy array of floats, [slope, offset]
    Returns:
        chi_squared: float
    """
    return np.sum((linear_function([parameters[0], parameters[1]], x_data)
                   - y_data)**2 / y_uncertainties**2)

def plot_linear_data(data, parameter_uncertainties):
    figure = plt.figure(figsize=(8, 6))

    axes_main_plot = figure.add_subplot(211)

    axes_main_plot.errorbar(data[:,0], data[:,1], yerr=data[:,2],
                            fmt='x', color='black')
    axes_main_plot.plot(data[:,0], linear_function(parameters, data[:,0]),
                        color='red')
    axes_main_plot.grid(True)
    axes_main_plot.set_title(PLOT_NAME, fontsize=14)
    axes_main_plot.set_xlabel(X_LABEL)
    axes_main_plot.set_ylabel(Y_LABEL)
    # Fitting details
    chi_squared = chi_squared_function(data[:,0], data[:,1], data[:,2],
                                       parameters)
    degrees_of_freedom = len(data[:,0]) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom

    axes_main_plot.annotate((r'$\chi^2$ = {0:4.2f}'.
                             format(chi_squared)), (1, 0), (-60, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                             format(degrees_of_freedom)), (1, 0), (-147, -55),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.2f}'.
                             format(reduced_chi_squared)), (1, 0), (-104, -70),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate('Fit: $y=mx+c$', (0, 0), (0, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points')
    axes_main_plot.annotate(('m = {0:6.4e}'.format(parameters[0])), (0, 0),
                            (0, -55), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[0])),
                            (0, 0), (100, -55), xycoords='axes fraction',
                            va='top', fontsize='10',
                            textcoords='offset points')
    axes_main_plot.annotate(('c = {0:6.4e}'.format(parameters[1])), (0, 0),
                            (0, -70), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[1])),
                            (0, 0), (100, -70), xycoords='axes fraction',
                            textcoords='offset points', va='top',
                            fontsize='10')
    # Residuals plot
    residuals = data[:,1] - linear_function(parameters, data[:,0])
    axes_residuals = figure.add_subplot(414)
    axes_residuals.errorbar(data[:,0], residuals, yerr=data[:,2],
                            fmt='x', color='black')
    axes_residuals.plot(data[:,0], 0 * data[:,0], color='red')
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14)


    plt.savefig(NAME_OF_FIGURE_DATA)
    plt.show()


def define_error_plotting(file_dimension):
    global Y_ERR
    global X_ERR
    global Y_ERR
    global X_ERR
    
    if file_dimension == 2:
        X_ERR = False
        Y_ERR = False
    if file_dimension == 3:
        Y_ERR = data[:,2]
        X_ERR = False
    elif file_dimension == 4:
        Y_ERR = data[:,3]
        X_ERR = data[:,2]
    return None

def linear_fitting_procedure(data):
    """Implements an analytic approach according to source in header.
    Args:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
    Returns:
        parameters: numpy array of floats, [slope, offset]
        parameter_uncertainties: numpy array of floats, [slope_uncertainty,
                                 offset_uncertainty]
        """
    weights = 1. / data[:,2]**2
    repeated_term = (np.sum(weights) * np.sum(data[:,0]**2 * weights)
                     - np.sum(data[:,0] * weights)**2)
    slope = ((np.sum(weights) * np.sum(data[:,0] * data[:,1] * weights)
              - np.sum(data[:,0] * weights) * np.sum(data[:,1] * weights))
             / repeated_term)
    slope_uncertainty = np.sqrt(np.sum(weights) / repeated_term)
    offset = ((np.sum(data[:,1] * weights) * np.sum(data[:,0]**2 * weights)
               - np.sum(data[:,0] * weights) * np.sum(data[:,0] * data[:,1] * weights))
              / repeated_term)
    offset_uncertainty = np.sqrt(np.sum(data[:,0]**2 * weights) / repeated_term)
    
    print(f"Gradient: {slope:.5g} +- {slope_uncertainty:.5g}")
    print(f"Intercept: {offset:.5g} +- {offset_uncertainty:.5g}")
    
    return (np.array([slope, offset]), np.array([slope_uncertainty,
                                                 offset_uncertainty]))

def linear_function(parameters, x_variable):

    return parameters[0] * x_variable + parameters[1]

def linear_function_2(x, m, c):

    return m*x + c

def function(x_data, parameter):
    
    return 


def fit_function(data, file_dimension):
    if ATTEMPT_FIT and file_dimension == 3:
        
        fit_parameters, cov = sc.curve_fit(function, data[:,0],
                                                     data[:,1],
                                                     p0=PARAMETER_ESTIMATE,
                                                     sigma=data[:,2],
                                                     absolute_sigma=True)
        fit_values = function(data[:,0], fit_parameters[0], fit_parameters[1])
    
    elif ATTEMPT_FIT and file_dimension == 2:
        try:
            
            fit_parameters, cov = sc.curve_fit(function, data[:,0],
                                                     data[:,1],
                                                     p0=PARAMETER_ESTIMATE)
            fit_values = function(data[:,0], fit_parameters[0])
        except: RuntimeError
        print("Cannot find a fit")
        fit_values = False
        fit_parameters = False
        cov = False

    return fit_values, fit_parameters, cov


def calculate_chi_squared(data, fit_values):
    return np.sum((abs(data[:,1] - fit_values) /
                   data[:,2]))


def calculate_reduced_chi_squared(chi_squared, data, parameters):

    return chi_squared / (len(data) - len(parameters))


def find_maxima(data):
    """
    Finds local maxima in the data.

    Parameters:
    -----------
    data : numpy.ndarray
       A 2D numpy array containing data, where the first column represents x values
       and the second column represents y values.

    prominence : float
       The prominence of peaks to consider. Peaks with a prominence lower than this
       value will be ignored.

    Returns:
    --------
    numpy.ndarray, numpy.ndarray
       Two numpy arrays containing the x and y coordinates of the identified local maxima.

    """

    # Find peaks in the y values of the data
    peaks, _ = find_peaks(data[:,1], prominence = PROMINENCE) 
    
    # Extract the y values of the local maxima
    y_max = data[:,1][peaks]
    
    # Extract the x values corresponding to the identified local maxima
    x_max = data[:,0][peaks]
    
    print("\n X-Maxima: ", x_max)
    print("\n Y_Maxima: ", y_max)

    return x_max, y_max

def find_minima(data):
    """
    Finds local minima in the data.

    Parameters:
        -----------
    data : numpy.ndarray
        A 2D numpy array containing data, where the first column represents x values
        and the second column represents y values.

    prominence : float
        The prominence of troughs to consider. Troughs with a prominence lower than this
        value will be ignored.

    Returns:
    --------
    numpy.ndarray, numpy.ndarray
        Two numpy arrays containing the x and y coordinates of the identified local minima.

    """
    
    # Find peaks in the negative of the y values of the data
    # This efficiently finds troughs or local minima
    peaks, _ = find_peaks(- data[:,1], prominence = PROMINENCE)
    
    # Extract the y values of the local minima
    y_min = data[:,1][peaks]
    
    # Extract the x values corresponding to the identified local minima
    x_min = data[:,0][peaks]
    
    print("\n X-Minima", x_min)
    print("\n Y-Minima", y_min)

    return x_min, y_min

def print_format_output(parameters_value, cov): 
    # Format boson mass
    parameter_string = f"{parameters_value:.4g}"
    number = count_decimal_places(parameter_string)
    print(f"Parameter: {parameters_value:.4g} +/- {np.sqrt(cov[0,0]):.{number}g}" )

def count_decimal_places(parameters):
    if '.' in parameters:
        return len(parameters.split('.')[1])
    return 0

def linear_odr_fitting(data, file_dimension):

    linear_model = Model(linear_function)
    odr_data = RealData(data[:,0], data[:,1], sx = data[:,2], sy = data[:,3])
    #odr_data = RealData(data[:,0], data[:,1])
    
    odr = ODR(odr_data, linear_model, beta0=[0.1, 0.1])
    out = odr.run()
    #out.pprint()
    
    define_error_plotting(file_dimension)
    plot_data(data, linear_function(out.beta, data[:,0]))
    #breakpoint()
    print(f"Gradient: {out.beta[0]:.5g} +- {out.sd_beta[0]/np.sqrt(out.res_var):.5g}")
    print(f"Intercept: {out.beta[1]:.5g} +- {out.sd_beta[1]:.5g}")
    #TODO later
    fit_pars = (out.beta)
    print(f"Reduced chi squared: {out.res_var:.4g}")
    params, covariance = curve_fit(linear_function_2, data[:, 0], data[:, 1], sigma=data[:, 3])
    print("Gradient:", params[0])
    print("Intercept:", params[1])
    return None

def non_linear_odr_function(parameter, x_data):

    a , b, c = parameter
    return a*(x_data - b)**2 + c

def odr_fitting(data, file_dimension):
    quad_model = Model(non_linear_odr_function)
    
    
    odr_data = RealData(data[:,0], data[:,1], sx=data[:,2], sy=data[:,3])
    odr = ODR(odr_data, quad_model, beta0=[-300, 1.4, 15000])
    out = odr.run()
    #out.pprint()
    
    
    define_error_plotting(file_dimension)
    plot_data(data, non_linear_odr_function(out.beta, data[:,0]))
    
    
    
    #parameters, _ = sc.curve_fit(non_linear_odr_function, data[:,0], data[:,1], p0 = [-3, 1.4, 20000])
    
    
    
    return None

def analyse_fit(pars, par_unc):
    current_analysis = "Stefan Boltzmann constant"
    if ANALYSE_FIT == True:
        print("The current analysis routine is for:", current_analysis)
        r_CN = 0.01 
        r_D = 0.0125
        R = 0.20
        length_unc = 0.002
        sb = pars[0]/(np.pi*(r_CN**2) * np.pi*(r_D**2)/(np.pi*(R**2)))      
        sb_unc = sb*np.sqrt(((par_unc[0]/pars[0])**2)+4*(length_unc**2)*(1/(r_CN**2) + 1/(r_D**2) + 1/(R**2)))  
        print(f"sb = {sb:.5g} +- {sb_unc:.5g}")
        print("True value: 5.670367 × 10−8 W m−2 K−4")
    return 0 

data, file_dimension = read_and_check_files(FILES)
data = clean_data(data, file_dimension)
#data[:,3] = 0.001*data[:,3]
#print(data)

data = truncate(data)

if LINEAR and file_dimension == 2:
    define_error_plotting(file_dimension)
    fit_values = False
    plt.plot(data, fit_values)

elif not LINEAR and file_dimension == 2:
    if not ATTEMPT_FIT:
        print("!")
        define_error_plotting(file_dimension)
        fit_values = False
        plot_data(data, fit_values)
    elif ATTEMPT_FIT:
        fit_values, parameters, cov = fit_function(data, file_dimension)
        define_error_plotting(file_dimension)
        plot_data(data, fit_values)


elif LINEAR and file_dimension == 3:
    parameters, linear_parameter_uncertainty = linear_fitting_procedure(data)
    plot_linear_data(data, linear_parameter_uncertainty)
    analyse_fit(parameters, linear_parameter_uncertainty)
elif not LINEAR and file_dimension == 3:
    if ATTEMPT_FIT:
        
        fit_values, parameters, cov = fit_function(data, file_dimension)
        define_error_plotting(file_dimension)
        
        plot_data(data, fit_values)
        
        chi_squared = calculate_chi_squared(data, fit_values)
        reduced_chi_squared = calculate_reduced_chi_squared(chi_squared, data, parameters)
        
        print_format_output(parameters[0], cov)
        print_format_output(parameters[1], cov)
        print(f"fReduced Chi squared: {reduced_chi_squared:.4g}")
    else:
        fit_values = False
        define_error_plotting(file_dimension)
        plot_data(data, fit_values)

elif LINEAR and file_dimension == 4:
    linear_odr_fitting(data, file_dimension)
elif not LINEAR and file_dimension == 4:
    if ATTEMPT_FIT:
        define_error_plotting(file_dimension)
        odr_fitting(data, file_dimension)
    elif not ATTEMPT_FIT:
        fit_values = False
        define_error_plotting(file_dimension)
        plot_data(data, fit_values)
else:
    print("This code works with data and uncertainties in up to 4 columns.")
    sys.exit()

if FIND_PEAKS:
    x_max, y_max = find_maxima(data)
    
if FIND_TROUGHS:
    x_min, y_min = find_minima(data)






