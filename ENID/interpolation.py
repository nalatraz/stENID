from george import kernels
import george
from scipy.optimize import minimize

def GP_simple(x, y, yerr, plot=True):
    
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)
    
    kernel = np.var(y) * kernels.ExpSquaredKernel(20)
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    
    x_pred = np.linspace(min(x), max(x), 1000)
    
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    p0 = gp.get_parameter_vector()
    
    gp.set_parameter_vector(result.x)
    
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    pred_error = np.sqrt(pred_var)
    
    if plot == True:
        plt.figure()
        plt.fill_between(x_pred, pred - pred_error, pred + pred_error,
                        color="k", alpha=0.2)
        plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.xlabel("Days")
        plt.ylabel("Normalised Flux")
        #plt.gca().invert_yaxis()
        plt.show()
    
    return x_pred, pred, pred_error
    
def mag2flux_normalised(mag, mag_error):
    
    # Generate arrays for storage
    lc_flux = np.array([])
    lc_error = np.array([])
    
    # Convert magnitudes to flux. We have assumed zero points equal to 0.
    for i in range(len(mag)):
        flux = 10 ** (- mag[i] / 2.5)
        flux_error = flux * mag_error[i] / 1.0857
        
        lc_flux = np.append(lc_flux, flux)
        lc_error = np.append(lc_error, flux_error)
            
    # Normalise the flux so that the peak value is 1
    scale_factor = np.nanmax(lc_flux)
    lc_flux = lc_flux / scale_factor
    lc_error = lc_error / scale_factor
    
    return lc_flux, lc_error, scale_factor

def flux2mag(flux, flux_error, scale_factor):
    
    # Scale flux back to true value
    flux_scaled = flux * scale_factor
    error_scaled = flux_error * scale_factor
    
    # Generate arrays for storage
    lc_mag = np.array([])
    lc_mag_error = np.array([])
    
    # Convert flux to magnitude. We have assumed zero points equal to 0.
    for i in range(len(flux)):
        mag = -2.5*np.log(flux_scaled[i])
        mag_error = 1.0857 * error_scaled[i] / flux_scaled
        
        lc_mag = np.append(lc_mag, mag)
        lc_mag_error = np.append(lc_mag_error, mag_error)
            
    return lc_mag, lc_mag_error

def interpolate(mag, mag_error, mjd):
    
    # Compute the normalised flux
    flux, flux_error, scale = mag2flux_normalised(mag, mag_error)
    
    # Interpolate the lightcurve with a simple Gaussian Process
    xpred, pred, pred_error = GP_simple(mjd, flux, flux_error)
    
    # Convert back to magnitudes
    mag, mag_error = flux2mag(pred, pred_error, scale)
    
    return mag, mag_error
