import logging
import numpy as np
from numba import njit


from ngmix.em import EM_MAXITER, EM_RANGE_ERROR
from ngmix.gmix import GMix
from ngmix.gexceptions import GMixRangeError
from ngmix.gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gauss2d_set_norm,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)
from ngmix.fastexp import expd

logger = logging.getLogger(__name__)


class GMixEMFixCen(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    Parameters
    ----------
    obs: Observation
        An ngmix.Observation object

        The image should not have zero or negative pixels. You can
        use the ngmix.em.prep_image() function to ensure this.
    minimum: number, optional
        The minimum number of iterations, default 10
    maxiter: number, optional
        The maximum number of iterations, default 1000
    tol: number, optional
        The tolerance in the moments that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level
    """
    def __init__(self,
                 obs,
                 miniter=10,
                 maxiter=1000,
                 tol=0.001,
                 vary_sky=False):

        self._obs = obs

        self.miniter = miniter
        self.maxiter = maxiter
        self.tol = tol
        self.vary_sky = vary_sky

        self._gm = None
        self._sums = None
        self._result = None

        self._set_runner()

    def has_gmix(self):
        """
        returns True if a gmix is set
        """
        if hasattr(self, '_gm'):
            return True
        else:
            return False

    def get_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm.copy()

    def get_convolved_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        gm = self.get_gmix().copy()
        em_convolve_1gauss(
            gm.get_data(),
            self._obs.psf.gmix.get_data(),
        )
        return gm

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self):
        """
        Get an image of the best fit mixture
        """
        return self._gm.make_image(
            self._obs.image.shape,
            jacobian=self._obs.jacobian,
        )

    def go(self, gmix_guess, sky):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing a starting
            guess for the algorithm.  This should be *before* psf convolution.
        sky: number
            The sky value added to the image
        """

        if hasattr(self, '_gm'):
            del self._gm

        obs = self._obs
        gmix_psf = obs.psf.gmix

        conf = self._make_conf()
        conf['sky'] = sky

        gm = gmix_guess.copy()

        sums = self._make_sums(len(gm))

        pixels = obs.pixels.copy()

        if np.any(pixels['ierr'] <= 0.0):
            fill_zero_weight = True
        else:
            fill_zero_weight = False

        flags = 0
        try:
            numiter, fdiff, sky = self._runner(
                conf,
                pixels,
                sums,
                gm.get_data(),
                gmix_psf.get_data(),
                fill_zero_weight=fill_zero_weight,
            )

            pars = gm.get_full_pars()
            self._gm = GMix(pars=pars)

            if numiter >= self.maxiter:
                flags = EM_MAXITER
                message = 'maxit'
            else:
                message = 'OK'

            result = {
                'flags': flags,
                'numiter': numiter,
                'fdiff': fdiff,
                'sky': sky,
                'message': message,
            }

        except GMixRangeError as err:
            message = str(err)
            logger.info(message)
            result = {
                'flags': EM_RANGE_ERROR,
                'message': message,
            }

        self._result = result

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_fixcen)

    def _make_conf(self):
        """
        make the sum structure
        """
        conf = np.zeros(1, dtype=_em_conf_dtype)
        conf = conf[0]

        conf['tol'] = self.tol
        conf['miniter'] = self.miniter
        conf['maxiter'] = self.maxiter
        conf['pixel_scale'] = self._obs.jacobian.scale
        conf['vary_sky'] = self.vary_sky

        return conf

    def _set_runner(self):
        self._runner = em_run_fixcen


class GMixEMPOnly(GMixEMFixCen):
    """
    Fit an image with a gaussian mixture using the EM algorithm,
    allowing only the fluxes to vary

    Parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    minimum: number, optional
        The minimum number of iterations, default 10
    maxiter: number, optional
        The maximum number of iterations, default 1000
    tol: number, optional
        The tolerance in the fluxes that implies convergence,
        default 0.001
    """

    def _set_runner(self):
        self._runner = em_run_ponly

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_ponly)


@njit
def em_run_fixcen(conf, pixels, sums, gmix, gmix_psf, fill_zero_weight=False):
    """
    run the EM algorithm, with fixed positions and a psf mixture to provide a
    shapred minimim resolution

    Parameters
    ----------
    conf: array
        Should have fields

            tol: tolerance for stopping
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            pixel_scale: pixel scale
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    sums: array with fields
        The sums array, a type _sums_dtype_fixcen
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    """

    em_convolve_1gauss(gmix, gmix_psf)
    gmix_set_norms(gmix)

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcen(sums)
        set_logtau_logdet(gmix, sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums_fixcen(pixel, gmix, sums)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums_fixcen(sums, pixel, gtot)

        gmix_set_from_sums_fixcen(gmix, gmix_psf, sums, pix_area)

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        numiter = i+1
        if numiter >= conf['miniter']:
            frac_diff = (elogL - elogL_last)/elogL
            if frac_diff < tol:
                break

        elogL_last = elogL

    em_deconvolve_1gauss(gmix, gmix_psf)
    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def fill_zero_weight_pixels(gmix, pixels, sky):
    """
    fill zero weight pixels with the model
    """

    for pixel in pixels:
        if pixel['ierr'] <= 0.0:
            val = gmix_eval_pixel_fast(gmix, pixel)
            pixel['val'] = sky + val


@njit
def do_scratch_sums_fixcen(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using scratch space in the sums struct
    """

    gsum = 0.0
    logL = 0.0

    n_gauss = gmix.size
    for i in range(n_gauss):
        gauss = gmix[i]
        tsums = sums[i]

        v = pixel['v']
        u = pixel['u']

        vdiff = v-gauss['row']
        udiff = u-gauss['col']

        u2 = udiff*udiff
        v2 = vdiff*vdiff
        uv = udiff*vdiff

        chi2 = \
            gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

        if chi2 < 25.0 and chi2 >= 0.0:
            tsums['gi'] = gauss['pnorm']*expd(-0.5*chi2)
        else:
            tsums['gi'] = 0.0

        gsum += tsums['gi']

        tsums['tv2sum'] = v2*tsums['gi']
        tsums['tuvsum'] = uv*tsums['gi']
        tsums['tu2sum'] = u2*tsums['gi']

        logL += tsums['gi']*(tsums['logtau'] - 0.5*tsums['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums_fixcen(sums, pixel, gtot):
    """
    do the sums based on the scratch values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['u2sum'] += tsums['tu2sum']*factor
        tsums['uvsum'] += tsums['tuvsum']*factor
        tsums['v2sum'] += tsums['tv2sum']*factor


@njit
def gmix_set_from_sums_fixcen(gmix, gmix_psf, sums, pix_area):
    """
    fill the gaussian mixture from the em sums, requiring
    that the covariance matrix after psf deconvolution
    is not singular
    """

    # assuming single gaussian for psf
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        irr = tsums['v2sum']*pinv - psf_irr
        irc = tsums['uvsum']*pinv - psf_irc
        icc = tsums['u2sum']*pinv - psf_icc

        if irr < 0.0 or icc < 0.0:
            irr, irc, icc = 0.0001, 0.0, 0.0001

        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        # ngmix works in surface brightness, so multiply
        # p by area since it has the actual image value in there
        gauss2d_set(
            gauss,
            p*pix_area,
            gauss['row'],
            gauss['col'],
            irr + psf_irr,
            irc + psf_irc,
            icc + psf_icc,
        )

        gauss2d_set_norm(gauss)


@njit
def set_logtau_logdet(gmix, sums):
    """
    set log(tau) and log(det) for every gaussian
    """

    for i in range(gmix.size):
        gauss = gmix[i]
        tsums = sums[i]
        tsums['logtau'] = np.log(gauss['p'])
        tsums['logdet'] = np.log(gauss['det'])


@njit
def clear_sums_fixcen(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    sums['tu2sum'][:] = 0.0
    sums['tuvsum'][:] = 0.0
    sums['tv2sum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0
    sums['u2sum'][:] = 0.0
    sums['uvsum'][:] = 0.0
    sums['v2sum'][:] = 0.0


@njit
def em_run_ponly(conf, pixels, sums, gmix, gmix_psf, fill_zero_weight=False):
    """
    run the EM algorithm, allowing only fluxes to vary

    Parameters
    ----------
    conf: array
        Should have fields
            tol: tolerance for stopping
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            pixel_scale: pixel scale
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_ponly
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    """

    em_convolve_1gauss(gmix, gmix_psf)
    gmix_set_norms(gmix)

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    p_last = gmix['p'].sum()

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_ponly(sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix, pixels, sky)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gsum = do_scratch_sums_ponly(pixel, gmix, sums)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            skysum += sky*pixel['val']/gtot

            do_sums_ponly(sums, pixel, gtot)

        gmix_set_from_sums_ponly(gmix, sums, pix_area)

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        psum = gmix['p'].sum()

        numiter = i+1
        if numiter >= conf['miniter']:
            frac_diff = abs(psum/p_last-1)
            if frac_diff < tol:
                break

        p_last = psum

    em_deconvolve_1gauss(gmix, gmix_psf)
    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_ponly(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    gsum = 0.0

    n_gauss = gmix.size
    for i in range(n_gauss):
        gauss = gmix[i]
        tsums = sums[i]

        v = pixel['v']
        u = pixel['u']

        vdiff = v-gauss['row']
        udiff = u-gauss['col']

        u2 = udiff*udiff
        v2 = vdiff*vdiff
        uv = udiff*vdiff

        chi2 = \
            gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

        if chi2 < 25.0 and chi2 >= 0.0:
            tsums['gi'] = gauss['pnorm']*expd(-0.5*chi2)
        else:
            tsums['gi'] = 0.0

        gsum += tsums['gi']

    return gsum


@njit
def do_sums_ponly(sums, pixel, gtot):
    """
    do the sums based on the scratch values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # gi*image_val/sum(gi)
        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau


@njit
def gmix_set_from_sums_ponly(gmix, sums, pix_area):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        # ngmix works in surface brightness
        p = tsums['pnew']*pix_area

        gauss2d_set(
            gauss,
            p,
            gauss['row'],
            gauss['col'],
            gauss['irr'],
            gauss['irc'],
            gauss['icc'],
        )

        gauss2d_set_norm(gauss)


@njit
def clear_sums_ponly(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0


@njit
def em_convolve_1gauss(gmix, gmix_psf, fix=False):
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]

    for i in range(gmix.size):
        gauss = gmix[i]

        irr = gauss['irr'] + psf_irr
        irc = gauss['irc'] + psf_irc
        icc = gauss['icc'] + psf_icc

        if fix:
            det = irr*icc - irc**2
            if det < GMIX_LOW_DETVAL:
                T = irr + icc
                irr = icc = T/2
                irc = 0.0

        gauss2d_set(
            gauss,
            gauss['p'],
            gauss['row'],
            gauss['col'],
            irr,
            irc,
            icc,
        )


@njit
def em_deconvolve_1gauss(gmix, gmix_psf):
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]

    for i in range(gmix.size):
        gauss = gmix[i]

        gauss2d_set(
            gauss,
            gauss['p'],
            gauss['row'],
            gauss['col'],
            gauss['irr'] - psf_irr,
            gauss['irc'] - psf_irc,
            gauss['icc'] - psf_icc,
        )


@njit
def get_flux(gsum, g2sum, gIsum):
    """
    calculate the flux from the cross-correlation sums
    """
    return gsum*gIsum/g2sum


_em_conf_dtype = [
    ('tol', 'f8'),
    ('maxiter', 'i4'),
    ('miniter', 'i4'),
    ('sky', 'f8'),
    ('vary_sky', 'bool'),
    ('pixel_scale', 'f8'),
]
_em_conf_dtype = np.dtype(_em_conf_dtype, align=True)

_sums_dtype_fixcen = [
    ('gi', 'f8'),

    # used for convergence tests only
    ('logtau', 'f8'),
    ('logdet', 'f8'),

    # scratch on a given pixel
    ('tu2sum', 'f8'),
    ('tuvsum', 'f8'),
    ('tv2sum', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
    ('u2sum', 'f8'),
    ('uvsum', 'f8'),
    ('v2sum', 'f8'),
]
_sums_dtype_fixcen = np.dtype(_sums_dtype_fixcen, align=True)

_sums_dtype_ponly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_ponly = np.dtype(_sums_dtype_ponly, align=True)

"""
_flux_sums_dtype = [
    ('gsum', 'f8'),
    ('g2sum', 'f8'),
    ('gIsum', 'f8'),
]
_flux_sums_dtype = np.dtype(_flux_sums_dtype, align=True)
"""
