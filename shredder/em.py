"""
TODO
    - do we need to force the pre-psf gaussian to be posdev etc?
"""
import logging
import numpy as np
from numba import njit

import ngmix
from ngmix.em import EM_MAXITER, EM_RANGE_ERROR
from ngmix.gmix import GMix
from ngmix.gexceptions import GMixRangeError
from ngmix.gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gmix_get_cen,
    gmix_convolve_fill,
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
        Get a copy of the gaussian mixture from the final iteration
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm.copy()

    def get_convolved_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm_conv.copy()

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
            del self._gm_conv

        obs = self._obs

        # makes a copy
        if not obs.has_psf() or not obs.psf.has_gmix():
            logger.debug('NO PSF SET')
            gmix_psf = ngmix.GMixModel([0., 0., 0., 0., 0., 1.0], 'gauss')
        else:
            gmix_psf = obs.psf.gmix
            gmix_psf.set_flux(1.0)

        conf = self._make_conf()
        conf['sky'] = sky

        gm = gmix_guess.copy()
        gm_conv = gm.convolve(gmix_psf)

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
                gm_conv.get_data(),
                fill_zero_weight=fill_zero_weight,
            )

            pars = gm.get_full_pars()
            pars_conv = gm_conv.get_full_pars()
            self._gm = GMix(pars=pars)
            self._gm_conv = GMix(pars=pars_conv)

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

        except (GMixRangeError, ZeroDivisionError) as err:
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
def em_run_fixcen(conf,
                  pixels,
                  sums,
                  gmix,
                  gmix_psf,
                  gmix_conv,
                  fill_zero_weight=False):
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
    sums: array with fields
        The sums array, a type _sums_dtype_fixcen
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcen(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums_fixcen(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums_fixcen(sums, pixel, gtot)

        gmix_set_from_sums_fixcen(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
            pix_area,
        )

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

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
def do_scratch_sums_fixcen(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tv2sum'] = 0.0
        tsums['tuvsum'] = 0.0
        tsums['tu2sum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*expd(-0.5*chi2)
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tv2sum'] += v2*val
            tsums['tuvsum'] += uv*val
            tsums['tu2sum'] += u2*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

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
def gmix_set_from_sums_fixcen(gmix,
                              gmix_psf,
                              gmix_conv,
                              sums,
                              pix_area):
    """
    fill the gaussian mixture from the em sums, requiring that the covariance
    matrix before psf deconvolution is not singular

    We may want to relax that
    """

    minval = 1.0e-4

    _, _, psf_irr, psf_irc, psf_icc, _ = gmix_get_moms(gmix_psf)

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        irr = tsums['v2sum']*pinv
        irc = tsums['uvsum']*pinv
        icc = tsums['u2sum']*pinv

        # get pre-psf moments, only works if the psf gaussians
        # are all centered
        irr = irr - psf_irr
        irc = irc - psf_irc
        icc = icc - psf_icc

        # currently are forcing the sizes of pre-psf gaussians to
        # be positive.  We may be able to relax this

        if irr < 0.0 or icc < 0.0:
            irr, irc, icc = minval, 0.0, minval

        # this causes oscillations in likelihood
        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        # ngmix works in surface brightness, so multiply p by area since it
        # has the actual image value in there

        gauss2d_set(
            gauss,
            p*pix_area,
            gauss['row'],
            gauss['col'],
            irr,
            irc,
            icc,
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


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
def em_run_ponly(conf,
                 pixels,
                 sums,
                 gmix,
                 gmix_psf,
                 gmix_conv,
                 fill_zero_weight=False):
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
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    p_last = gmix['p'].sum()

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_ponly(sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gsum = do_scratch_sums_ponly(pixel, gmix_conv, sums, ngauss_psf)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            skysum += sky*pixel['val']/gtot

            do_sums_ponly(sums, pixel, gtot)

        gmix_set_from_sums_ponly(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
            pix_area,
        )

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

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_ponly(pixel, gmix_conv, sums, ngauss_psf):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v-gauss['row']
            udiff = u-gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*expd(-0.5*chi2)
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

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
def gmix_set_from_sums_ponly(gmix,
                             gmix_psf,
                             gmix_conv,
                             sums,
                             pix_area):
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

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


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

        print('p:', gauss['p'])
        print('T before:', gauss['irr'] + gauss['icc'])
        gauss2d_set(
            gauss,
            gauss['p'],
            gauss['row'],
            gauss['col'],
            gauss['irr'] - psf_irr,
            gauss['irc'] - psf_irc,
            gauss['icc'] - psf_icc,
        )
        print('T after:', gauss['irr'] + gauss['icc'])


@njit
def get_flux(gsum, g2sum, gIsum):
    """
    calculate the flux from the cross-correlation sums
    """
    return gsum*gIsum/g2sum


@njit
def gmix_get_moms(gmix):
    """
    get row, col, irr, irc, icc, psum
    """

    row, col, psum = gmix_get_cen(gmix)

    irr = irc = icc = 0.0

    for i in range(gmix.size):
        gauss = gmix[i]

        rowdiff = gauss['row'] - row
        coldiff = gauss['col'] - col

        p = gauss['p']
        irr += p*(gauss['irr'] + rowdiff**2)
        irc += p*(gauss['irc'] + rowdiff*coldiff)
        icc += p*(gauss['icc'] + coldiff**2)

    irr /= psum
    irc /= psum
    icc /= psum

    return row, col, irr, irc, icc, psum


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

_tau_dtype = [
    ('logtau', 'f8'),
    ('logdet', 'f8'),
]
_tau_dtype = np.dtype(_tau_dtype)

_sums_dtype_ponly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_ponly = np.dtype(_sums_dtype_ponly, align=True)
