"""
TODO
    - make ponly also use a psf gmix
"""

import os
import numpy as np

import ngmix
from ngmix.gmix import GMix


from ngmix.gexceptions import GMixRangeError
from ngmix import UnitJacobian
from ngmix import Observation

import fitsio
from numba import njit
import esutil as eu
import tempfile

# argh, forgot to import
from esutil.pbar import prange

from ngmix.gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gauss2d_set_norm,
    GMIX_LOW_DETVAL,
)
from ngmix.fastexp import expd
import biggles
import plotting
import images
import copy


EM_RANGE_ERROR = 2**0
EM_MAXITER = 2**1
GMIX_LOW_DETVAL_THIS = 0.001


def show_image(fname):
    os.system('feh --force-aliasing -B black %s &' % fname)


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
def em_run_fixcen(conf, pixels, sums, gmix, gmix_psf):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    em_convolve_1gauss(gmix, gmix_psf)
    gmix_set_norms(gmix)

    tol = conf['tol']
    counts = conf['counts']

    area = pixels.size*conf['pixel_scale']*conf['pixel_scale']

    nsky = conf['sky_guess']/counts
    psky = conf['sky_guess']/(counts/area)

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcen(sums)
        set_logtau_logdet(gmix, sums)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gtot, tlogL = do_scratch_sums_fixcen(pixel, gmix, sums)

            elogL += tlogL

            gtot += nsky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel['val']/counts
            skysum += nsky*imnorm/gtot
            igrat = imnorm/gtot

            # multiply sums by igrat, among other things
            do_sums_fixcen(sums, igrat)

        gmix_set_from_sums_fixcen(gmix, gmix_psf, sums)

        psky = skysum
        nsky = psky/area

        # print(elogL_last, elogL)
        if i > conf['miniter']:
            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    numiter = i+1

    sky = psky*(counts/area)

    em_deconvolve_1gauss(gmix, gmix_psf)
    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_fixcen(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    gtot = 0.0
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

        gtot += tsums['gi']

        tsums['tv2sum'] = v2*tsums['gi']
        tsums['tuvsum'] = uv*tsums['gi']
        tsums['tu2sum'] = u2*tsums['gi']

        logL += tsums['gi']*(tsums['logtau'] - 0.5*tsums['logdet'] - 0.5*chi2)

    if gtot == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gtot

    return gtot, logL


@njit
def do_sums_fixcen(sums, igrat):
    """
    do the sums based on the scratch values
    """

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # wtau is gi[pix]/gtot[pix]*imnorm[pix]
        # which is Dave's tau*imnorm = wtau
        wtau = tsums['gi']*igrat

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['u2sum'] += tsums['tu2sum']*igrat
        tsums['uvsum'] += tsums['tuvsum']*igrat
        tsums['v2sum'] += tsums['tv2sum']*igrat


@njit
def gmix_set_from_sums_fixcen(gmix, gmix_psf, sums):
    """
    fill the gaussian mixture from the em sums
    """

    # assuming single gaussian for psf
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]
    # psf_irr, psf_irc, psf_icc = 0.0, 0.0, 0.0

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

        gauss2d_set(
            gauss,
            p,
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


class GMixEMFixCen(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    """
    def __init__(self, obs):

        self._obs = obs

        self._counts = obs.image.sum()

        self._gm = None
        self._sums = None
        self._result = None
        self._sky_guess = None

    def get_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        return self._gm

    def get_convolved_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        gm = self.get_gmix().copy()
        em_convolve_1gauss(
            gm.get_data(),
            self.obs.psf.gmix.get_data(),
        )
        return gm

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self, counts=None):
        """
        Get an image of the best fit mixture
        """
        im = self._gm.make_image(self._obs.image.shape,
                                 jacobian=self._obs.jacobian)
        if counts is not None:
            im *= (counts/im.sum())
        return im

    def go(self, gmix_guess, sky_guess, miniter=10, maxiter=100, tol=1.e-6):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing
            a starting guess for the algorithm
        sky_guess: number
            A guess at the sky value
        maxiter: number, optional
            The maximum number of iterations, default 100
        tol: number, optional
            The tolerance in the moments that implies convergence,
            default 1.e-6
        """

        if hasattr(self, '_gm'):
            del self._gm

        obs = self._obs
        gmix_psf = obs.psf.gmix

        conf = self._make_conf()
        conf['tol'] = tol
        conf['maxiter'] = maxiter
        conf['miniter'] = miniter
        conf['sky_guess'] = sky_guess
        conf['counts'] = self._counts
        conf['pixel_scale'] = obs.jacobian.get_scale()

        gm = gmix_guess.copy()

        sums = self._make_sums(len(gm))

        flags = 0
        try:
            numiter, fdiff, sky = em_run_fixcen(
                conf,
                obs.pixels,
                sums,
                gm.get_data(),
                gmix_psf.get_data(),
            )

            # we have mutated the _data elements, we want to make
            # sure the pars are propagated.  Make a new full gm
            pars = gm.get_full_pars()
            self._gm = GMix(pars=pars)

            if numiter >= maxiter:
                flags = EM_MAXITER

            result = {
                'flags': flags,
                'numiter': numiter,
                'fdiff': fdiff,
                'sky': sky,
                'message': 'OK',
            }

        # except (GMixRangeError, ZeroDivisionError) as err:
        except GMixRangeError as err:
            # most likely the algorithm reached an invalid gaussian
            message = str(err)
            print(message)
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
        conf_arr = np.zeros(1, dtype=_em_conf_dtype)
        return conf_arr[0]


_sums_dtype_fixcen = [
    ('gi', 'f8'),
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


@njit
def em_run_fixsize_logL(conf, pixels, sums, gmix):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    # ngauss = gmix.size
    # rows_last = np.zeros(ngauss) - 9999.0
    # cols_last = np.zeros(ngauss) - 9999.0

    gmix_set_norms(gmix)
    tol = conf['tol']
    counts = conf['counts']

    area = pixels.size*conf['pixel_scale']*conf['pixel_scale']

    nsky = conf['sky_guess']/counts
    psky = conf['sky_guess']/(counts/area)

    elogL_last = 9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixsize(sums)
        set_logtau_logdet(gmix, sums)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gtot, tlogL = do_scratch_sums_fixsize(pixel, gmix, sums)

            elogL += tlogL

            gtot += nsky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel['val']/counts
            skysum += nsky*imnorm/gtot
            igrat = imnorm/gtot

            # multiply sums by igrat, among other things
            do_sums_fixsize(sums, igrat)

        gmix_set_from_sums_fixsize(gmix, sums)

        psky = skysum
        nsky = psky/area

        frac_diff = abs((elogL - elogL_last)/elogL)
        # frac_diff = abs(elogL_last - elogL)
        # print(elogL_last, elogL, frac_diff)
        if frac_diff < tol:
            break

        elogL_last = elogL

        """
        converged = True
        maxdiff = 0.0
        for j in range(ngauss):
            row_diff = abs(gmix['row'][j] - rows_last[j])
            col_diff = abs(gmix['col'][j] - cols_last[j])

            maxdiff = max(maxdiff, row_diff, col_diff)
            if row_diff > tol or col_diff > tol:
                converged = False

            rows_last[j] = gmix['row'][j]
            cols_last[j] = gmix['col'][j]

        # print(maxdiff)
        if converged:
            break
        """

    numiter = i+1
    # return numiter, maxdiff, psky*(counts/area)
    return numiter, frac_diff, psky*(counts/area)


@njit
def em_run_fixsize_pos(conf, pixels, sums, gmix):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    ngauss = gmix.size
    rows_last = np.zeros(ngauss) - 9999.0
    cols_last = np.zeros(ngauss) - 9999.0

    gmix_set_norms(gmix)
    tol = conf['tol']
    counts = conf['counts']

    area = pixels.size*conf['pixel_scale']*conf['pixel_scale']

    nsky = conf['sky_guess']/counts
    psky = conf['sky_guess']/(counts/area)

    # elogL_last = 9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixsize(sums)
        set_logtau_logdet(gmix, sums)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gtot, tlogL = do_scratch_sums_fixsize(pixel, gmix, sums)

            elogL += tlogL

            gtot += nsky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel['val']/counts
            skysum += nsky*imnorm/gtot
            igrat = imnorm/gtot

            # multiply sums by igrat, among other things
            do_sums_fixsize(sums, igrat)

        gmix_set_from_sums_fixsize(gmix, sums)

        psky = skysum
        nsky = psky/area

        """
        frac_diff = abs((elogL - elogL_last)/elogL)
        # frac_diff = abs(elogL_last - elogL)
        # print(elogL_last, elogL, frac_diff)
        if frac_diff < tol:
            break

        elogL_last = elogL
        """

        converged = True
        maxdiff = 0.0
        for j in range(ngauss):
            row_diff = abs(gmix['row'][j] - rows_last[j])
            col_diff = abs(gmix['col'][j] - cols_last[j])

            maxdiff = max(maxdiff, row_diff, col_diff)
            if row_diff > tol or col_diff > tol:
                converged = False

            rows_last[j] = gmix['row'][j]
            cols_last[j] = gmix['col'][j]

        # print(maxdiff)
        if converged:
            break

    numiter = i+1
    return numiter, maxdiff, psky*(counts/area)
    # return numiter, frac_diff, psky*(counts/area)


@njit
def do_scratch_sums_fixsize(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    gtot = 0.0
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

        gtot += tsums['gi']

        tsums['trowsum'] = v*tsums['gi']
        tsums['tcolsum'] = u*tsums['gi']

        # logL += tsums['gi']*(tsums['logtau']
        # - 0.5*tsums['logdet'] - 0.5*chi2)
        logL += tsums['gi']*(tsums['logtau'] - 0.5*chi2)

    if gtot == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gtot

    return gtot, logL


@njit
def do_sums_fixsize(sums, igrat):
    """
    do the sums based on the scratch values
    """

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # wtau is gi[pix]/gtot[pix]*imnorm[pix]
        # which is Dave's tau*imnorm = wtau
        wtau = tsums['gi']*igrat

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['rowsum'] += tsums['trowsum']*igrat
        tsums['colsum'] += tsums['tcolsum']*igrat


@njit
def gmix_set_from_sums_fixsize(gmix, sums):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        if p < 1.0e-20:
            p = 1.0e-20
        pinv = 1.0/p

        gauss2d_set(
            gauss,
            p,
            tsums['rowsum']*pinv,
            tsums['colsum']*pinv,
            gauss['irr'],
            gauss['irc'],
            gauss['icc'],
        )

        gauss2d_set_norm(gauss)


@njit
def clear_sums_fixsize(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    sums['trowsum'][:] = 0.0
    sums['tcolsum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0
    sums['rowsum'][:] = 0.0
    sums['colsum'][:] = 0.0


class GMixEMFixSize(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    """
    def __init__(self, obs):

        self._obs = obs

        self._counts = obs.image.sum()

        self._gm = None
        self._sums = None
        self._result = None
        self._sky_guess = None

    def get_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        return self._gm

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self, counts=None):
        """
        Get an image of the best fit mixture
        """
        im = self._gm.make_image(
            self._obs.image.shape,
            jacobian=self._obs.jacobian,
        )

        if counts is not None:
            im *= (counts/im.sum())
        return im

    def go(self, gmix_guess, sky_guess, maxiter=100, tol=1.e-5):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing
            a starting guess for the algorithm
        sky_guess: number
            A guess at the sky value
        maxiter: number, optional
            The maximum number of iterations, default 100
        tol: number, optional
            The tolerance in the moments that implies convergence,
            default 1.e-5, tighter than for fixcen
        """

        if hasattr(self, '_gm'):
            del self._gm

        conf = self._make_conf()
        conf['tol'] = tol
        conf['maxiter'] = maxiter
        conf['sky_guess'] = sky_guess
        conf['counts'] = self._counts
        conf['pixel_scale'] = self._obs.jacobian.get_scale()

        gm = gmix_guess.copy()
        sums = self._make_sums(len(gm))

        flags = 0
        try:
            numiter, fdiff, sky = em_run_fixsize_logL(
                conf,
                self._obs.pixels,
                sums,
                gm.get_data(),
            )

            # we have mutated the _data elements, we want to make
            # sure the pars are propagated.  Make a new full gm
            pars = gm.get_full_pars()
            self._gm = GMix(pars=pars)

            if numiter >= maxiter:
                flags = EM_MAXITER

            result = {
                'flags': flags,
                'numiter': numiter,
                'fdiff': fdiff,
                'sky': sky,
                'message': 'OK',
            }

        # except (GMixRangeError, ZeroDivisionError) as err:
        except GMixRangeError as err:
            # most likely the algorithm reached an invalid gaussian
            message = str(err)
            print('error in em_run_fixsize:', message)
            result = {
                'flags': EM_RANGE_ERROR,
                'message': message,
            }

        self._result = result

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_fixsize)

    def _make_conf(self):
        """
        make the sum structure
        """
        conf_arr = np.zeros(1, dtype=_em_conf_dtype)
        return conf_arr[0]


_sums_dtype_fixsize = [
    ('gi', 'f8'),
    ('logtau', 'f8'),
    ('logdet', 'f8'),

    # scratch on a given pixel
    ('trowsum', 'f8'),
    ('tcolsum', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
    ('rowsum', 'f8'),
    ('colsum', 'f8'),
]
_sums_dtype_fixsize = np.dtype(_sums_dtype_fixsize, align=True)

_em_conf_dtype = [
    ('tol', 'f8'),
    ('maxiter', 'i4'),
    ('miniter', 'i4'),
    ('sky_guess', 'f8'),
    ('counts', 'f8'),
    ('pixel_scale', 'f8'),
]
_em_conf_dtype = np.dtype(_em_conf_dtype, align=True)


@njit
def em_run_ponly(conf, pixels, sums, gmix):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    p_last = gmix['p'].sum()

    gmix_set_norms(gmix)
    tol = conf['tol']
    counts = conf['counts']

    area = pixels.size*conf['pixel_scale']*conf['pixel_scale']

    nsky = conf['sky_guess']/counts
    psky = conf['sky_guess']/(counts/area)

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_ponly(sums)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gtot = do_scratch_sums_ponly(pixel, gmix, sums)

            gtot += nsky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel['val']/counts
            skysum += nsky*imnorm/gtot
            igrat = imnorm/gtot

            # multiply sums by igrat, among other things
            do_sums_ponly(sums, igrat)

        gmix_set_from_sums_ponly(gmix, sums)

        psky = skysum
        nsky = psky/area

        psum = gmix['p'].sum()
        frac_diff = abs(psum/p_last-1)
        if frac_diff < tol:
            break

        p_last = psum

    numiter = i+1
    return numiter, frac_diff, psky*(counts/area)


@njit
def do_scratch_sums_ponly(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    gtot = 0.0

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

        gtot += tsums['gi']

    return gtot


@njit
def do_sums_ponly(sums, igrat):
    """
    do the sums based on the scratch values
    """

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # wtau is gi[pix]/gtot[pix]*imnorm[pix]
        # which is Dave's tau*imnorm = wtau
        wtau = tsums['gi']*igrat

        tsums['pnew'] += wtau


@njit
def gmix_set_from_sums_ponly(gmix, sums):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']

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


class GMixEMPOnly(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    """
    def __init__(self, obs):

        self._obs = obs

        self._counts = obs.image.sum()

        self._gm = None
        self._sums = None
        self._result = None
        self._sky_guess = None

    def get_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        return self._gm

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self, counts=None):
        """
        Get an image of the best fit mixture
        """
        im = self._gm.make_image(
            self._obs.image.shape,
            jacobian=self._obs.jacobian,
        )

        if counts is not None:
            im *= (counts/im.sum())
        return im

    def go(self, gmix_guess, sky_guess, maxiter=100, tol=1.e-6):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing
            a starting guess for the algorithm
        sky_guess: number
            A guess at the sky value
        maxiter: number, optional
            The maximum number of iterations, default 100
        tol: number, optional
            The tolerance in the moments that implies convergence,
            default 1.e-6
        """

        if hasattr(self, '_gm'):
            del self._gm

        conf = self._make_conf()
        conf['tol'] = tol
        conf['maxiter'] = maxiter
        conf['sky_guess'] = sky_guess
        conf['counts'] = self._counts
        conf['pixel_scale'] = self._obs.jacobian.get_scale()

        gm = gmix_guess.copy()
        sums = self._make_sums(len(gm))

        flags = 0
        try:
            numiter, fdiff, sky = em_run_ponly(
                conf,
                self._obs.pixels,
                sums,
                gm.get_data(),
            )

            # we have mutated the _data elements, we want to make
            # sure the pars are propagated.  Make a new full gm
            pars = gm.get_full_pars()
            self._gm = GMix(pars=pars)

            if numiter >= maxiter:
                flags = EM_MAXITER

            result = {
                'flags': flags,
                'numiter': numiter,
                'fdiff': fdiff,
                'sky': sky,
                'message': 'OK',
            }

        except (GMixRangeError, ZeroDivisionError) as err:
            # most likely the algorithm reached an invalid gaussian
            message = str(err)
            print('error in em_run_ponly:', message)
            result = {
                'flags': EM_RANGE_ERROR,
                'message': message,
            }

        self._result = result

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_ponly)

    def _make_conf(self):
        """
        make the sum structure
        """
        conf_arr = np.zeros(1, dtype=_em_conf_dtype)
        return conf_arr[0]


_sums_dtype_ponly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_ponly = np.dtype(_sums_dtype_ponly, align=True)


KERNEL = np.array([
    [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
    [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
    [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
    [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],
    [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
    [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
    [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
])


def get_flux(im, wt, model_in):
    flux = -9999.e9
    flux_err = 1.e9

    model = model_in.copy()
    for ipass in [1, 2]:

        if ipass == 1:
            model *= 1.0/model.sum()
            xcorr_sum = (model*im*wt).sum()
            msq_sum = (model*model*wt).sum()
        else:
            model *= flux
            chi2 = ((model-im)**2 * wt).sum()

        if ipass == 1:
            if msq_sum == 0:
                break
            flux = xcorr_sum/msq_sum

    # final flux calculation with error checking
    if msq_sum == 0:
        print('cannot calculate err')
    else:

        arg = chi2/msq_sum/(im.size-1)
        if arg >= 0.0:
            flux_err = np.sqrt(arg)
        else:
            print('cannot calculate err')

    return flux, flux_err


def scale_image(im, noise):
    minval = -noise*0.1
    # minval = noise*1.0
    im = im.clip(min=minval)
    logim = np.log(im)
    logim -= logim.min()

    return logim


def asinh_scale(im_input, noise, nonlinear=0.012, scale=0.0005):
    """
    Scale the image using and asinh stretch

        I = image*f
        f=asinh(image/nonlinear)/(image/nonlinear)

    Values greater than 1.0 after this scaling will be shown as white, so you
    are responsible for pre-scaling your image.  Values < 0 are clipped.

    parameters
    ----------
    image:
        The image.
    nonlinear: keyword
        The non-linear scale.
    """
    from numpy import arcsinh

    noise_fac = 0.1
    minval = noise_fac*noise

    im = im_input.copy()
    im.clip(min=minval, out=im)
    im *= scale

    fac = 1.0/nonlinear
    Im = fac*im

    f = arcsinh(Im)/Im

    im *= f

    im.clip(max=1.0, out=im)
    return im


def compare_images(image,
                   weight,
                   model,
                   seg,
                   scale=0.0005,
                   label='image'):
    import biggles
    import images

    diff = image - model
    chi_image = (image - model)*np.sqrt(weight)

    chi2 = (chi_image**2).sum()
    chi2per = chi2/(image.size-1)

    noise = np.sqrt(1.0/weight[0, 0])

    tim = asinh_scale(image, noise, scale=scale)
    tmodel = asinh_scale(model, noise, scale=scale)
    tdiff = asinh_scale(diff, noise, scale=scale)

    tab = biggles.Table(2, 2, aspect_ratio=1)
    tab[0, 0] = images.view(
        tim,
        show=False,
        title=label,
    )
    tab[0, 1] = images.view(
        tmodel,
        show=False,
        title='model',
    )
    tab[1, 0] = images.view(
        tdiff,
        show=False,
        title='chi2/dof: %.2f' % chi2per,
    )
    tab[1, 1] = images.view(
        seg,
        show=False,
        autoscale=True,
    )

    return tab


def view_mbobs(mbobs, scale, show=False):

    imlist = [olist[0].image for olist in mbobs]
    wtlist = [olist[0].weight for olist in mbobs]

    rgb = make_rgb(imlist, wtlist, scale=scale)
    plt = images.view(rgb, show=show)
    return plt


def view_rgb(imlist, wtlist, scale):
    rgb = make_rgb(imlist, wtlist, scale=scale)
    images.view(rgb)


def compare_rgb_images(image, model, diffim,
                       seg, width, chi2,
                       rng=None,
                       title=None,
                       model_noisy=None):
    import biggles
    import images

    dof = image.size-3
    chi2per = chi2/dof

    if model_noisy is not None:
        arat = 2.0/3.0
        tab = biggles.Table(2, 3, aspect_ratio=arat)
    else:
        arat = image.shape[1]/image.shape[0]
        tab = biggles.Table(2, 2, aspect_ratio=arat)

    tab[0, 0] = images.view(
        image,  # /maxval,
        show=False,
        title='image',
    )
    if model_noisy is not None:
        tab[0, 1] = images.view(
            model_noisy,  # /maxval,
            show=False,
            title='model+noise',
        )
        tab[0, 2] = images.view(
            model,  # /maxval,
            show=False,
            title='model',
        )
    else:
        tab[0, 1] = images.view(
            model,  # /maxval,
            show=False,
            title='model',
        )

    tab[1, 0] = images.view(
        diffim,
        show=False,
        title='chi2/dof: %.2f' % chi2per,
    )

    """
    tab[1, 1] = images.view(
        seg,
        show=False,
        title='seg',
    )
    """
    tab[1, 1] = plot_seg(seg, rng=rng, width=width, title='seg')

    if title is not None:
        tab.title = title

    fname = '/tmp/tmp-comp.png'
    tab.write_img(width, width*arat, fname)
    show_image(fname)


def get_shape_guess(g1, g2, width, rng, gmax=0.9):
    """
    Get guess, making sure in range
    """

    g = np.sqrt(g1**2 + g2**2)
    if g > gmax:
        fac = gmax/g

        g1 = g1 * fac
        g2 = g2 * fac

    shape = ngmix.Shape(g1, g2)

    while True:
        try:
            g1_offset = rng.uniform(low=width, high=width)
            g2_offset = rng.uniform(low=width, high=width)
            shape_new = shape.get_sheared(g1_offset, g2_offset)
            break
        except GMixRangeError:
            pass

    return shape_new.g1, shape_new.g2


def make_dbsim(rng, sim_config='dbsim.yaml', **kw):
    import dbsim

    sim_conf = eu.io.read(sim_config)

    pos_sampler = dbsim.descwl_sim.PositionSampler(
        sim_conf['positions'],
        rng,
    )
    cat_sampler = dbsim.descwl_sim.CatalogSampler(
        sim_conf,
        rng,
    )

    sim = dbsim.descwl_sim.DESWLSim(
        sim_conf,
        cat_sampler,
        pos_sampler,
        rng,
    )
    sim.update(kw)

    return sim


def get_uberseg(seg, weight_in, object_number, fast=True):
    """
    Get the cweight map and zero out pixels not nearest to central
    object.

    Adapted from Niall Maccrann and Joe Zuntz.

    Parameters
    ----------
    iobj : int
        Index of the object.
    icutout : int
        Index of cutout.
    fast : bool, optional
        Use the fast C code.

    Returns
    -------
    weight : np.array
        The weight map as a numpy array.
    """
    import meds

    weight = weight_in.copy()

    # if only have sky and object, then just return
    if len(np.unique(seg)) == 2:
        return weight

    # First get all indices of all seg map pixels which contain an object
    # i.e. are not equal to zero

    obj_inds = np.where(seg != 0)

    if fast:
        # call fast c code with tree
        Nx, Ny = seg.shape
        Ninds = len(obj_inds[0])
        seg = seg.astype(np.int32)
        weight = weight.astype(np.float32, copy=False)

        obj_inds_x = obj_inds[0].astype(np.int32, copy=False)
        obj_inds_y = obj_inds[1].astype(np.int32, copy=False)

        meds.meds._uberseg.uberseg_tree(
            seg, weight, Nx, Ny,
            object_number, obj_inds_x, obj_inds_y, Ninds)

    else:
        # Then loop through pixels in seg map, check which obj ind it is
        # closest to.  If the closest obj ind does not correspond to the
        # target, set this pixel in the weight map to zero.

        for i, row in enumerate(seg):
            for j, element in enumerate(row):
                obj_dists = (i-obj_inds[0])**2 + (j-obj_inds[1])**2
                ind_min = np.argmin(obj_dists)

                segval = seg[obj_inds[0][ind_min], obj_inds[1][ind_min]]
                if segval != object_number:
                    weight[i, j] = 0.

    return weight


def do_psf_fit(psf_obs, rng):
    runner = ngmix.bootstrap.PSFRunner(
        psf_obs,
        'gauss',
        4.0*psf_obs.jacobian.scale,
        {},
        rng=rng,
    )
    runner.go()
    gmix = runner.fitter.get_gmix()
    print(gmix)
    psf_obs.set_gmix(gmix)


def get_model_fits(model,
                   image,
                   weight,
                   psf_obs,
                   jacobian,
                   objs,
                   seg,
                   rng,
                   show=False,
                   width=1000):

    cen_prior = ngmix.priors.CenPrior(0, 0, 0.01, 0.01, rng=rng)
    g_prior = ngmix.priors.GPriorBA(0.2, rng=rng)
    T_prior = ngmix.priors.FlatPrior(-0.5, 1.e9, rng=rng)
    # prior = ngmix.priors.LogNormal(0.5, 2.3, rng=rng)
    flux_prior = ngmix.priors.FlatPrior(-1000.0, 1.e9, rng=rng)

    if 'bd' in model:
        fracdev_prior = ngmix.priors.Normal(
            0.5, 0.1, rng=rng, bounds=[0.0, 1.0],
        )

    if model == 'bdf':
        prior = ngmix.joint_prior.PriorBDFSep(
            cen_prior,
            g_prior,
            T_prior,
            fracdev_prior,
            flux_prior,
        )
    elif model == 'bd':
        prior = ngmix.joint_prior.PriorBDSep(
            cen_prior,
            g_prior,
            T_prior,
            # on log10(Tratio) gives std of 0.1
            ngmix.priors.Normal(0.0, 0.05, rng=rng),
            fracdev_prior,
            flux_prior,
        )

    else:
        prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            flux_prior,
        )
    max_pars = {
        'method': 'lm',
        'lm_pars': {
            # 'maxfev': 4000,
            'xtol': 1.0e-5,
            'ftol': 1.0e-5,
        }
    }

    if show:
        grid = plotting.Grid(objs.size)
        arat = 0.5 * grid.nrow/grid.ncol
        tab = biggles.Table(grid.nrow, grid.ncol, aspect_ratio=arat)

    gmlist = []
    for object_number in range(1, objs.size+1):
        i = object_number - 1

        uberseg = get_uberseg(seg, weight, object_number)

        if show:
            grow, gcol = grid(i)
            tab[grow, gcol] = images.view_mosaic(
                [uberseg, image*uberseg], show=False,
            )

        row, col = objs['y'][i], objs['x'][i]
        tjac = jacobian.copy()
        tjac.set_cen(row=row, col=col)

        obs = ngmix.Observation(
            image,
            weight=uberseg,
            jacobian=tjac,
            psf=psf_obs,
        )

        x2 = objs['x2'][i]
        y2 = objs['y2'][i]
        Tsep = (x2 + y2)*jacobian.scale**2

        # Tguess = obs.psf.gmix.get_T()
        Tguess = Tsep
        flux = objs['flux'][i]*jacobian.scale**2

        if model == 'bdf':
            guesser = ngmix.guessers.BDFGuesser(
                Tguess,
                flux,
                prior,
            )
            runner = ngmix.bootstrap.BDFRunner(
                obs,
                max_pars,
                guesser,
                prior=prior,
            )
        elif model == 'bd':
            guesser = ngmix.guessers.BDGuesser(
                Tguess,
                flux,
                prior,
            )
            runner = ngmix.bootstrap.BDRunner(
                obs,
                max_pars,
                guesser,
                prior=prior,
            )

        else:
            guesser = ngmix.guessers.TFluxGuesser(
                Tguess,
                flux,
            )
            runner = ngmix.bootstrap.MaxRunner(
                obs,
                model,
                max_pars,
                guesser,
                prior=prior,
            )

        runner.go(ntry=2)
        fitter = runner.fitter
        res = fitter.get_result()
        if res['flags'] != 0:
            print('model fit fail, using point source')
            this_pars = guesser()
            this_pars[2:2+3] = 0.0
            print(this_pars)
            if model == 'bdf':
                this_gm = ngmix.GMixBDF(pars=this_pars)
            else:
                this_gm = ngmix.GMixModel(this_pars, model)
            print(this_gm)
        else:
            this_gm = fitter.get_gmix()

        gmlist.append(this_gm)

    if show:
        tab.show(width=width, height=arat*width)

    return gmlist


def make_coadd_obs(mbobs):

    """
    for obslist in mbobs:
        for obs in obslist:
            wbad = np.where(obs.weight <= 0.0)
            assert wbad[0].size == 0
    """
    weights = np.array([obslist[0].weight.max() for obslist in mbobs])
    wsum = weights.sum()

    nweights = weights/wsum

    coadd_image = mbobs[0][0].image.copy()
    coadd_image[:, :] = 0.0
    coadd_weight = coadd_image.copy()
    coadd_weight[:, :] = wsum

    coadd_psf = mbobs[0][0].psf.image.copy()
    coadd_psf[:, :] = 0
    coadd_psf_weight = mbobs[0][0].psf.weight.copy()*3  # not quite right

    for i, obslist in enumerate(mbobs):
        obs = obslist[0]
        coadd_image += obs.image*nweights[i]
        coadd_psf += obs.psf.image*nweights[i]

    """
    import images
    images.view(
        asinh_scale(coadd_image, np.sqrt(1/coadd_weight[0, 0]), scale=0.005),
    )
    """

    psf_obs = ngmix.Observation(
        coadd_psf,
        weight=coadd_psf_weight,
        jacobian=mbobs[0][0].psf.jacobian,
    )
    obs = ngmix.Observation(
        coadd_image,
        weight=coadd_weight,
        jacobian=mbobs[0][0].jacobian,
        psf=psf_obs,
    )
    print('noise 0:         ', np.sqrt(1/mbobs[0][0].weight[0, 0]))
    print('coadd noise:     ', np.sqrt(1/coadd_weight[0, 0]))
    return obs


def make_rgb(imlist, wtlist, nonlinear=0.12, scale=0.0005):
    import images

    # relative_scales = np.array([1.1, 1.0, 2.0])
    relative_scales = np.array([1.0, 1.0, 2.0])

    scales = scale*relative_scales

    noise_fac = 0.1
    rminval = noise_fac*np.sqrt(1/wtlist[2][0, 0])
    gminval = noise_fac*np.sqrt(1/wtlist[1][0, 0])
    bminval = noise_fac*np.sqrt(1/wtlist[0][0, 0])

    minval = min(rminval, gminval, bminval)
    minval = 0

    r = imlist[2].clip(min=minval)
    g = imlist[1].clip(min=minval)
    b = imlist[0].clip(min=minval)

    rgb = images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=nonlinear,
    )
    rgb.clip(max=1.0, out=rgb)
    return rgb


def run_sep(image, noise):
    import sep
    objs, seg = sep.extract(
        image,
        0.8,
        err=noise,
        deblend_cont=1.0e-5,
        minarea=4,
        filter_kernel=KERNEL,
        segmentation_map=True,
    )

    add_dt = [('number', 'i4')]
    objs = eu.numpy_util.add_fields(objs, add_dt)
    objs['number'] = np.arange(1, objs.size+1)
    return objs, seg


def plot_seg(segin, title=None, width=1000, rng=None, show=False):
    """
    plot the seg map with randomized ids for better display
    """

    seg = np.transpose(segin)

    cseg = np.zeros((seg.shape[0], seg.shape[1], 3))

    if rng is None:
        rng = np.random.RandomState()

    useg = np.unique(seg)[1:]

    low = 50/255
    high = 255/255

    for i, segval in enumerate(useg):

        w = np.where(seg == segval)

        r, g, b = rng.uniform(low=low, high=high, size=3)

        cseg[w[0], w[1], 0] = r
        cseg[w[0], w[1], 1] = g
        cseg[w[0], w[1], 2] = b

    plt = images.view(cseg, show=False)

    if title is not None:
        plt.title = title

    if show:
        srat = seg.shape[1]/seg.shape[0]
        fname = '/tmp/seg.png'
        plt.write_img(width, width*srat, fname)
        show_image(fname)

    return plt


@njit
def _get_seg_pairs(seg, pairs):

    nrow, ncol = seg.shape

    pi = 0
    for row in range(nrow):
        rowstart = row - 1
        rowend = row + 1

        for col in range(ncol):

            ind = seg[row, col]

            if ind == 0:
                # 0 means this pixel is not assigned to an object
                continue

            colstart = col - 1
            colend = col + 1

            for crow in range(rowstart, rowend+1):
                if crow == -1 or crow == nrow:
                    continue

                for ccol in range(colstart, colend+1):
                    if ccol == -1 or ccol == ncol:
                        continue

                    if crow == row and ccol == col:
                        continue

                    cind = seg[crow, ccol]

                    if cind != 0 and cind != ind:
                        # we found a neighboring pixel assigned
                        # to another object
                        pairs['number'][pi] = ind
                        pairs['nbr_number'][pi] = cind
                        pi += 1

    npairs = pi
    return npairs


def _get_unique_pairs(pairs):
    """
    get unique pairs, assuming max seg id number is at most
    1_000_000
    """

    tid = pairs['number']*1_000_000 + pairs['nbr_number']
    uid, uid_index = np.unique(tid, return_index=True)
    return pairs[uid_index]


class NbrsFoF(object):
    """
    extract unique groups
    """
    def __init__(self, nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(np.unique(nbrs_data['number']))

        # records fofid of entry
        self.linked = np.zeros(self.Nobj, dtype='i8')
        self.fofs = {}

        self._fof_data = None

    def get_fofs(self):
        self._make_fofs()
        return self._fof_data

    def _make_fofs(self):
        self._init_fofs()

        for i in prange(self.Nobj):
            self._link_fof(i)

        for fofid, k in enumerate(self.fofs):
            inds = np.array(list(self.fofs[k]), dtype=int)
            self.linked[inds[:]] = fofid

        self.fofs = {}

        self._make_fof_data()

    def _link_fof(self, mind):
        # get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))

        # always make a base fof
        if self.linked[mind] == -1:
            fofid = copy.copy(mind)
            self.fofs[fofid] = set([mind])
            self.linked[mind] = fofid
        else:
            fofid = copy.copy(self.linked[mind])

        # loop through nbrs
        for nbr in nbrs:
            if self.linked[nbr] == -1 or self.linked[nbr] == fofid:
                # not linked so add to current
                self.fofs[fofid].add(nbr)
                self.linked[nbr] = fofid
            else:
                # join!
                self.fofs[self.linked[nbr]] |= self.fofs[fofid]
                del self.fofs[fofid]
                fofid = copy.copy(self.linked[nbr])
                inds = np.array(list(self.fofs[fofid]), dtype=int)
                self.linked[inds[:]] = fofid

    def _make_fof_data(self):
        self._fof_data = []
        for i in range(self.Nobj):
            self._fof_data.append((self.linked[i], i+1))
        self._fof_data = np.array(
            self._fof_data,
            dtype=[('fofid', 'i8'), ('number', 'i8')]
        )
        i = np.argsort(self._fof_data['number'])
        self._fof_data = self._fof_data[i]
        assert np.all(self._fof_data['fofid'] >= 0)

    def _init_fofs(self):
        self.linked[:] = -1
        self.fofs = {}

    def _get_nbrs_index(self, mind):
        q, = np.where((self.nbrs_data['number'] == mind+1)
                      & (self.nbrs_data['nbr_number'] > 0))
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []


def get_fofs(seg):
    """
    group any objects whose seg maps touch
    """

    useg = np.unique(seg)
    w, = np.where(useg > 0)
    useg = useg[w]
    useg.sort()

    dtype = [('number', 'i4'), ('nbr_number', 'i4')]

    pairs_singleton = np.zeros(useg.size, dtype=dtype)
    pairs_singleton['number'] = useg
    pairs_singleton['nbr_number'] = useg

    pairs = np.zeros(seg.size, dtype=dtype)
    pairs['number'] = -1

    _get_seg_pairs(seg, pairs)
    w, = np.where(pairs['number'] > 0)
    pairs = pairs[w]

    pairs = np.hstack((pairs_singleton, pairs))

    pairs = _get_unique_pairs(pairs)

    nf = NbrsFoF(pairs)
    return nf.get_fofs()


def get_large_fof(fofs, minsize):
    """
    get objects in one large fof
    """

    hd = eu.stat.histogram(fofs['fofid'], more=True)
    wlarge, = np.where(hd['hist'] >= minsize)
    if wlarge.size == 0:
        raise ValueError('no fofs with size %d' % minsize)

    fofid = int(hd['center'][wlarge[0]])
    w, = np.where(fofs['fofid'] == fofid)
    return fofid, fofs['number'][w]-1


def _replace_with_noise(image, weight, indices, rng):
    noise = np.sqrt(1.0/weight.max())
    noise_image = rng.normal(
        scale=noise,
        size=image.shape
    )
    image[indices] = noise_image[indices]


def _replace_with_noise_old(obs, indices, rng):
    weight = obs.weight
    noise = np.sqrt(1.0/weight.max())
    noise_image = rng.normal(
        scale=noise,
        size=obs.image.shape
    )
    new_image = obs.image.copy()
    new_image[indices] = noise_image[indices]

    return ngmix.Observation(
        new_image,
        weight=weight,
        jacobian=obs.jacobian,
        psf=obs.psf,
    )


def get_fof_mbobs_and_objs(mbobs,
                           objs,
                           seg,
                           keep_numbers,
                           rng,
                           pixbuf=10):
    """
    get images with everythine noisy outside the
    FoF seg map
    """

    keep_indices = keep_numbers - 1

    mincol = objs['xmin'][keep_indices].min() - pixbuf
    maxcol = objs['xmax'][keep_indices].max() + pixbuf
    minrow = objs['ymin'][keep_indices].min() - pixbuf
    maxrow = objs['ymax'][keep_indices].max() + pixbuf

    if minrow < 0:
        minrow = 0
    if maxrow > seg.shape[0]:
        maxrow = seg.shape[0]

    if mincol < 0:
        mincol = 0
    if maxcol > seg.shape[1]:
        maxcol = seg.shape[1]

    newseg = seg[
        minrow:maxrow,
        mincol:maxcol,
    ].copy()

    new_objs = objs[keep_indices].copy()
    new_objs['y'] -= minrow
    new_objs['x'] -= mincol

    for i, number in enumerate(keep_numbers):
        tlogic = newseg != number
        if i == 0:
            logic = tlogic
        else:
            logic &= tlogic

    wout = np.where(logic)
    newseg[wout] = 0

    new_mbobs = ngmix.MultiBandObsList()
    for obslist in mbobs:
        new_obslist = ngmix.ObsList()
        for obs in obslist:
            new_image = obs.image[
                minrow:maxrow,
                mincol:maxcol,
            ].copy()
            new_weight = obs.weight[
                minrow:maxrow,
                mincol:maxcol,
            ].copy()

            _replace_with_noise(new_image, new_weight, wout, rng)

            new_obs = ngmix.Observation(
                new_image,
                weight=new_weight,
                jacobian=obs.jacobian,
                psf=obs.psf,
            )

            new_obslist.append(new_obs)

        new_mbobs.append(new_obslist)
    output = new_mbobs

    return output, newseg, new_objs


def show_fofs(mbobs,
              cat,
              fofs,
              minsize=2,
              viewscale=0.0005,
              width=1000):
    import pcolors
    import random

    plt = view_mbobs(mbobs, scale=viewscale)

    ufofs = np.unique(fofs['fofid'])

    print(fofs['fofid'])
    hd = eu.stat.histogram(fofs['fofid'], more=True)
    wlarge, = np.where(hd['hist'] >= minsize)
    print('found %d groups with size >= %d' % (wlarge.size, minsize))
    ngroup = wlarge.size

    if ngroup > 0:
        if ngroup > 1:
            colors = pcolors.rainbow(ngroup)
            random.shuffle(colors)
        else:
            colors = ['white']

        icolor = 0
        for fofid in ufofs:
            w, = np.where(fofs['fofid'] == fofid)
            if w.size >= minsize:
                print('plotting fof', fofid, 'size', w.size)

                mfofs, mcat = eu.numpy_util.match(
                    fofs['number'][w],
                    cat['number'],
                )

                color = colors[icolor]
                icolor += 1
                pts = biggles.Points(
                    cat['x'][mcat],
                    cat['y'][mcat],
                    type='filled circle',
                    size=1,
                    color=color,
                )
                plt.add(pts)

    fname = tempfile.mktemp(suffix='.png')
    plt.write_img(width, width, fname)
    show_image(fname)


def run_peaks(image, noise, scale, kernel_fwhm, weight_fwhm=1.2):
    import peaks

    objects1 = peaks.find_peaks(
        image=image,
        kernel_fwhm=kernel_fwhm/scale,
        noise=noise,
    )

    objects = peaks.get_moments(
        objects=objects1,
        image=image,
        fwhm=1.2,
        scale=scale,
        noise=noise,
    )

    seg = np.zeros(image.shape, dtype='i4')

    ind = np.arange(objects.size)
    for i in range(objects.size):
        row_start = objects['row_orig'][i]-1
        row_end = objects['row_orig'][i]+1
        col_start = objects['col_orig'][i]-1
        col_end = objects['col_orig'][i]+1

        seg[
            row_start:row_end+1,
            col_start:col_end+1,
        ] = 1 + ind[i]

    return objects, seg


def get_guess_from_detect(objs, scale, ur, model):
    nobj = objs.size

    guess_pars = []
    for i in range(objs.size):
        if 'T' in objs.dtype.names:
            Tguess = objs['T'][i]  # *scale**2
            row = objs['row'][i]*scale
            col = objs['col'][i]*scale
        else:
            x2 = objs['x2'][i]
            y2 = objs['y2'][i]
            Tguess = (x2 + y2)*scale**2
            row = objs['y'][i]*scale
            col = objs['x'][i]*scale

        g1, g2 = ur(low=-0.01, high=0.01, size=2)
        # print('%d (%g, %g) %g' % (i, g1, g2, Tguess))

        # our dbsim obs have jacobian "center" set to 0, 0

        if model == 'bdf':
            fracdev = ur(low=0.45, high=0.55)
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                fracdev,
                1.0/nobj,
            ]
            gm_model = ngmix.GMixBDF(pars=pars)
        elif model == 'bd':
            fracdev = ur(low=0.45, high=0.55)
            logTratio = ur(low=-0.01, high=0.01)
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                logTratio,
                fracdev,
                1.0/nobj,
            ]
            gm_model = ngmix.GMixModel(pars, model)

        else:
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                1.0/nobj,
            ]
            gm_model = ngmix.GMixModel(pars, model)

        # print('gm model guess')
        # print(gm_model)
        # perturb the models
        data = gm_model.get_data()
        for j in range(data.size):
            data['p'][j] *= (1 + ur(low=-0.05, high=0.05))

            fac = 0.01
            data['row'][j] += ur(low=-fac*scale, high=fac*scale)
            data['col'][j] += ur(low=-fac*scale, high=fac*scale)

            data['irr'][j] *= (1 + ur(low=-0.05, high=0.05))
            data['irc'][j] *= (1 + ur(low=-0.05, high=0.05))
            data['icc'][j] *= (1 + ur(low=-0.05, high=0.05))

        guess_pars += list(gm_model.get_full_pars())

    gm_guess = ngmix.GMix(pars=guess_pars)
    return gm_guess


def read_real_images(**kw):
    """
    read some example data
    """
    import psfex

    if 'row_range' not in kw:
        row_range = (1470, 1680)
    else:
        row_range = kw['row_range']

    if 'col_range' not in kw:
        col_range = (520, 790)
    else:
        col_range = kw['col_range']

    if row_range is None:
        row_range = [0, 2200]

    if col_range is None:
        col_range = [0, 4200]

    rowmin, rowmax = row_range
    colmin, colmax = col_range

    cen = (
        (rowmin + rowmax)/2,
        (colmin + colmax)/2,
    )

    dir = os.path.expandvars('$HOME/data/DES/coadd-examples')

    mbobs = ngmix.MultiBandObsList()
    for band in ['g', 'r', 'i']:
        fname = os.path.join(
            dir,
            'COSMOS_C08_r3764p01_%s.fits.fz' % band,
        )
        psf_fname = os.path.join(
            dir,
            'COSMOS_C08_r3764p01_%s_psfcat.psf' % band,
        )

        psf = psfex.PSFEx(psf_fname)
        psf_image = psf.get_rec(cen[0], cen[1])
        psf_weight = psf_image*0 + 1.0/0.001**2

        psf_cen = (np.array(psf_image.shape)-1)/2

        with fitsio.FITS(fname) as fits:
            image = fits['sci'][
                rowmin:rowmax,
                colmin:colmax,
            ]
            h = fits['sci'].read_header()

            weight = fits['wgt'][
                rowmin:rowmax,
                colmin:colmax,
            ]

        wcs = eu.wcsutil.WCS(h)
        dudcol, dudrow, dvdcol, dvdrow = wcs.get_jacobian(cen[1], cen[0])
        jacob = ngmix.Jacobian(
            # row=cen[0] - rowmin,
            # col=cen[1] - colmin,
            row=0,
            col=0,
            dudcol=dudcol,
            dudrow=dudrow,
            dvdcol=dvdcol,
            dvdrow=dvdrow,
        )

        psf_jacob = jacob.copy()
        psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

        psf_obs = ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jacob,
        )

        obs = ngmix.Observation(
            image=image,
            weight=weight,
            jacobian=jacob,
            psf=psf_obs,
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def test_real(min_fofsize=2,
              row_range=None,
              col_range=None,
              model='dev',
              tol=1.0e-3,
              maxiter=40000,
              viewscale=0.2,
              show=False,
              title=None,
              width=1000,
              seed=None):
    """
    true positions, ngauss related to T
    """
    import time

    print('seed:', seed)
    rng = np.random.RandomState(seed)

    mbobs = read_real_images(
        row_range=row_range,
        col_range=col_range,
    )

    ur = rng.uniform

    tm = 0.0

    for obslist in mbobs:
        for obs in obslist:
            do_psf_fit(obs.psf, rng)

    # coadd over bands
    coadd_obs = make_coadd_obs(mbobs)
    scale = coadd_obs.jacobian.scale
    do_psf_fit(coadd_obs.psf, rng)

    noise = np.sqrt(1/coadd_obs.weight.max())

    objs, seg = run_sep(coadd_obs.image, noise)
    print('found', objs.size, 'objects')

    if objs.size == 0:
        if show:
            view_rgb(mbobs, scale=viewscale, show=True)
        return

    fofs = get_fofs(seg)
    show_fofs(mbobs, objs, fofs, viewscale=viewscale, width=width)

    hd = eu.stat.histogram(fofs['fofid'], more=True)
    wlarge, = np.where(hd['hist'] >= min_fofsize)
    nfofs = wlarge.size
    print('found %d groups with size >= %d' % (nfofs, min_fofsize))

    rev = hd['rev']
    for ifof in range(nfofs):
        print('-'*70)
        print('FoF group %d/%d' % (ifof+1, nfofs))

        i = wlarge[ifof]
        indices = rev[rev[i]:rev[i+1]]

        # fof_objs = objs[indices]
        fofid = fofs['fofid'][indices[0]]

        numbers = indices + 1
        fof_mbobs, fof_seg, fof_objs = get_fof_mbobs_and_objs(
            mbobs,
            objs,
            seg,
            numbers,
            rng,
        )

        # plot_seg(fof_seg, rng=rng, show=True)
        # return

        fof_coadd_obs = make_coadd_obs(fof_mbobs)
        do_psf_fit(fof_coadd_obs.psf, rng)

        show_fofs(
            fof_mbobs,
            fof_objs,
            fofs[indices],
            # fof_ids=[fofid],
            viewscale=viewscale,
            width=width,
        )

        tm0 = time.time()

        imsky, sky = ngmix.em.prep_image(fof_coadd_obs.image)
        emobs = Observation(
            imsky,
            jacobian=fof_coadd_obs.jacobian,
            psf=fof_coadd_obs.psf,
        )

        for itry in range(2):

            gm_guess = get_guess_from_detect(
                fof_objs,
                scale,
                ur,
                model,
            )

            em = GMixEMFixCen(emobs)
            em.go(gm_guess, sky, miniter=50, maxiter=maxiter, tol=tol)

            res = em.get_result()
            if res['flags'] == 0:
                break

        if res['flags'] != 0:
            print('could not do fit')
            if show and input('enter a key, q to quit: ') == 'q':
                return

            continue

        # this gmix is the pre-seeing one
        gmfit = em.get_gmix()

        print('results')
        print(res)

        #
        # now fit to each band, letting the fluxes vary
        #

        imlist = []
        wtlist = []
        byband_gm = []
        byband_pars = []
        byband_imlist = []
        byband_wtlist = []
        difflist = []
        chi2 = 0.0

        for band, obslist in enumerate(fof_mbobs):
            obs = obslist[0]

            imlist.append(obs.image)
            wtlist.append(obs.weight)

            imsky, sky = ngmix.em.prep_image(obs.image)
            emobs = Observation(imsky, jacobian=obs.jacobian)
            em = GMixEMPOnly(emobs)

            for itry in range(4):
                # this is pre-psf
                gm_guess = gmfit.copy()

                # convolve with psf from this band and let things adjust
                em_convolve_1gauss(
                    gm_guess.get_data(),
                    obs.psf.gmix.get_data(),
                    fix=True,
                )

                gdata = gm_guess.get_data()
                for idata in range(gdata.size):
                    gdata['p'][idata] *= (1.0 + ur(low=-0.01, high=0.01))

                use_sky = res['sky']

                em.go(
                    gm_guess,
                    use_sky,
                    maxiter=maxiter,
                    tol=tol,
                )

                bres = em.get_result()
                if bres['flags'] == 0:
                    break

            if bres['flags'] != 0:
                if bres['message'] != 'OK':
                    raise RuntimeError('cannot proceed')

                print('did not succeed, going ahead anyway')

            bgm = em.get_gmix()
            print('band %d res:' % band)
            print(bres)

            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            bflux_tot, bflux_tot_err = get_flux(
                obs.image,
                obs.weight,
                bim,
            )
            bgm.set_flux(bflux_tot*obs.jacobian.scale**2)
            # bim *= bflux_tot/bim.sum()
            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            byband_gm.append(bgm)
            byband_pars.append(bgm.get_full_pars())
            byband_imlist.append(bim)
            byband_wtlist.append(obs.weight)

            tdiff = obs.image - bim
            difflist.append(tdiff)
            chi2 += (tdiff**2 * obs.weight).sum()

        this_tm = time.time() - tm0
        print('this time:', tm)
        tm += this_tm

        if show:
            rgb = make_rgb(imlist, wtlist, scale=viewscale)

            model_rgb = make_rgb(byband_imlist, byband_wtlist, scale=viewscale)

            diff_rgb = make_rgb(difflist, byband_wtlist, scale=viewscale)

            compare_rgb_images(
                rgb,
                model_rgb,
                diff_rgb,
                fof_seg,
                width,
                chi2,
                rng=rng,
                title=title,
                # model_noisy=model_rgb_noisy,
            )

        if nfofs > 1 and ifof < nfofs-1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return

    print('total time:', tm)
    print('time per:', tm/nfofs)


def test_fixcen(real_data=False,
                min_fofsize=40,
                row_range=None,
                col_range=None,
                sim_config='dbsim.yaml',
                det_method='sep',
                model='exp',
                nobj=None,
                ntrial=1,
                tol=1.0e-3,
                maxiter=40000,
                noise_factor=1,  # for low noise sim
                viewscale=0.0005,
                show=False,
                title=None,
                width=1000,
                seed=None):
    """
    true positions, ngauss related to T
    """
    import time

    print('seed:', seed)
    rng = np.random.RandomState(seed)

    if real_data:
        mbobs = read_real_images(
            row_range=row_range,
            col_range=col_range,
        )
        dosim = False
        ntrial = 1
    else:
        dosim = True
        sim = make_dbsim(
            rng,
            noise_factor=noise_factor,
            sim_config=sim_config,
        )

    ur = rng.uniform

    tm = 0.0
    for trial in range(ntrial):
        print('-'*70)
        print('%d/%d' % (trial+1, ntrial))

        if dosim:
            sim.make_obs(nobj=nobj)
            mbobs = sim.obs

        imlist = [o[0].image for o in mbobs]
        wtlist = [o[0].weight for o in mbobs]

        if False and show:
            rgb = make_rgb(imlist, wtlist, scale=viewscale)
            plt = images.view(rgb, show=False)
            fname = '/tmp/orig.png'
            plt.write_img(width, width, fname)
            show_image(fname)

        for obslist in mbobs:
            for obs in obslist:
                do_psf_fit(obs.psf, rng)

        # if trial == 0:
        #     psflist = [o[0].psf.image for o in mbobs]
        #     images.view_mosaic(psflist, title='psfs')

        # coadd over bands
        coadd_obs = make_coadd_obs(mbobs)
        scale = coadd_obs.jacobian.scale
        do_psf_fit(coadd_obs.psf, rng)

        noise = np.sqrt(1/coadd_obs.weight[0, 0])
        if det_method == 'sep':
            objs, seg = run_sep(coadd_obs.image, noise)

            fofs = get_fofs(seg)
            show_fofs(mbobs, objs, fofs, viewscale=viewscale, width=width)

            large_fofid, indices = get_large_fof(fofs, min_fofsize)
            objs_orig = objs
            objs = objs_orig[indices]

            numbers = indices + 1
            mbobs, seg = get_fof_mbobs(mbobs, seg, numbers, rng)

            # plot_seg(seg, rng=rng, show=True)
            # return

            imlist = [o[0].image for o in mbobs]
            wtlist = [o[0].weight for o in mbobs]

            coadd_obs = make_coadd_obs(mbobs)
            do_psf_fit(coadd_obs.psf, rng)

            show_fofs(mbobs, objs_orig, fofs,
                      fof_ids=[large_fofid],
                      viewscale=viewscale, width=width)

        else:
            psf_T = coadd_obs.psf.gmix.get_T()
            kernel_fwhm = ngmix.moments.T_to_fwhm(psf_T)
            objs, seg = run_peaks(coadd_obs.image, noise, scale, kernel_fwhm)

        print('found', objs.size, 'objects')
        if objs.size == 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        tm0 = time.time()

        imsky, sky = ngmix.em.prep_image(coadd_obs.image)
        emobs = Observation(
            imsky,
            jacobian=coadd_obs.jacobian,
            psf=coadd_obs.psf,
        )

        # fit to coadd
        for itry in range(2):

            gm_guess = get_guess_from_detect(
                objs,
                scale,
                ur,
                model,
            )
            # print('guess:')
            # print(gm_guess)

            em = GMixEMFixCen(emobs)
            em.go(gm_guess, sky, miniter=50, maxiter=maxiter, tol=tol)

            res = em.get_result()
            if res['flags'] == 0:
                break

        if res['flags'] != 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        # this gmix is the pre-seeing one
        gmfit = em.get_gmix()
        # print('best fit')
        # print(gmfit)

        print('results')
        print(res)
        # return

        #
        # now fit to each band, letting the fluxes vary
        #

        byband_gm = []
        byband_pars = []
        byband_imlist = []
        byband_wtlist = []
        difflist = []
        chi2 = 0.0

        for band, obslist in enumerate(mbobs):
            obs = obslist[0]
            imsky, sky = ngmix.em.prep_image(obs.image)
            emobs = Observation(imsky, jacobian=obs.jacobian)
            em = GMixEMPOnly(emobs)

            for itry in range(4):
                # this is pre-psf
                gm_guess = gmfit.copy()

                # convolve with psf from this band and let things adjust
                em_convolve_1gauss(
                    gm_guess.get_data(),
                    obs.psf.gmix.get_data(),
                    fix=True,
                )

                gdata = gm_guess.get_data()
                for idata in range(gdata.size):
                    gdata['p'][idata] *= (1.0 + ur(low=-0.01, high=0.01))

                use_sky = res['sky']

                em.go(
                    gm_guess,
                    use_sky,
                    maxiter=maxiter,
                    tol=tol,
                )

                bres = em.get_result()
                if bres['flags'] == 0:
                    break

            if bres['flags'] != 0:
                if bres['message'] != 'OK':
                    raise RuntimeError('cannot proceed')

                print('did not succeed, going ahead anyway')

            bgm = em.get_gmix()
            print('band %d res:' % band)
            print(bres)

            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            bflux_tot, bflux_tot_err = get_flux(
                obs.image,
                obs.weight,
                bim,
            )
            bgm.set_flux(bflux_tot*obs.jacobian.scale**2)
            # bim *= bflux_tot/bim.sum()
            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            byband_gm.append(bgm)
            byband_pars.append(bgm.get_full_pars())
            byband_imlist.append(bim)
            byband_wtlist.append(obs.weight)

            tdiff = obs.image - bim
            difflist.append(tdiff)
            chi2 += (tdiff**2 * obs.weight).sum()

        this_tm = time.time() - tm0
        print('this time:', tm)
        tm += this_tm

        if show:
            rgb = make_rgb(imlist, wtlist, scale=viewscale)

            """
            tmp_modlist = []
            for i in range(len(wtlist)):
                tnoise = np.sqrt(1/wtlist[i][0, 0])
                tmpim = byband_imlist[i].copy()
                tmpim += rng.normal(scale=tnoise, size=tmpim.shape)
                tmp_modlist.append(tmpim)
            """
            """
            model_rgb_noisy = make_rgb(
                tmp_modlist,
                byband_wtlist,
                scale=viewscale,
            )
            """
            model_rgb = make_rgb(byband_imlist, byband_wtlist, scale=viewscale)

            diff_rgb = make_rgb(difflist, byband_wtlist, scale=viewscale)

            compare_rgb_images(rgb, model_rgb, diff_rgb,
                               seg, width, chi2,
                               rng=rng,
                               # model_noisy=model_rgb_noisy,
                               title=title)

        if ntrial > 1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return

    print('total time:', tm)
    print('time per:', tm/ntrial)


def test_descwl_lin_allgauss(sim_config='dbsim.yaml',
                             model='bdf',
                             nobj=None,
                             ntrial=1,
                             tol=1.0e-3,
                             maxiter=40000,
                             scaling='linear',
                             noise_factor=1,  # for low noise sim
                             viewscale=0.0005,
                             show=False,
                             title=None,
                             width=1000,
                             seed=None):
    """
    true positions, ngauss related to T
    """
    import time

    rng = np.random.RandomState(seed)
    sim = make_dbsim(
        rng,
        noise_factor=noise_factor,
        sim_config=sim_config,
    )

    tm = 0.0
    for trial in range(ntrial):
        print('-'*70)
        print('%d/%d' % (trial+1, ntrial))

        sim.make_obs(nobj=nobj)
        mbobs = sim.obs

        imlist = [o[0].image for o in mbobs]
        wtlist = [o[0].weight for o in mbobs]

        for obslist in mbobs:
            for obs in obslist:
                do_psf_fit(obs.psf, rng)

        # coadd over bands
        coadd_obs = make_coadd_obs(mbobs)
        do_psf_fit(coadd_obs.psf, rng)

        noise = np.sqrt(1/coadd_obs.weight[0, 0])
        objs, seg = run_sep(coadd_obs.image, noise)

        print('found', objs.size, 'objects')
        if objs.size == 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        tm0 = time.time()

        gm_model_list = get_model_fits(
            model,
            coadd_obs.image,
            coadd_obs.weight,
            coadd_obs.psf,
            coadd_obs.jacobian,
            objs,
            seg,
            rng,
            # show=show,
            show=False,
            width=width,
        )
        if gm_model_list is None:
            if show:
                tab = biggles.Table(1, 2, aspect_ratio=0.5)
                rgb = make_rgb(imlist, wtlist, scale=viewscale)
                tab[0, 0] = images.view(rgb, show=False)
                tab[0, 1] = images.view(seg, show=False)
                tab.show(width=width, height=width*0.5)
            break

        byband_gm = []
        byband_pars = []
        byband_imlist = []
        byband_wtlist = []
        difflist = []
        chi2 = 0.0

        ngauss_per = len(gm_model_list[0])
        model_data = np.zeros((coadd_obs.image.size, objs.size*ngauss_per))
        for band, obslist in enumerate(mbobs):
            obs = obslist[0]
            jacob = obs.jacobian
            scale = jacob.scale

            bgm_list = []
            for i in range(objs.size):

                this_gm0 = gm_model_list[i].copy()

                row = objs['y'][i]*scale
                col = objs['x'][i]*scale
                this_gm0.set_cen(row=row, col=col)

                start = i*ngauss_per

                this_pars = this_gm0.get_full_pars()
                for j in range(ngauss_per):
                    gstart = j*6
                    gend = (j+1)*6
                    jpars = this_pars[gstart:gend]
                    jgm0 = ngmix.GMix(pars=jpars)
                    jgm0.set_flux(1.0)

                    jgm = jgm0.convolve(obs.psf.gmix)
                    try:
                        jgm.set_norms()
                    except GMixRangeError:
                        tdata = jgm0.get_data()
                        tdata['irr'] = 0.0
                        tdata['irc'] = 0.0
                        tdata['icc'] = 0.0
                        jgm = jgm0.convolve(obs.psf.gmix)
                        jgm.set_norms()

                    jim = jgm.make_image(
                        obs.image.shape,
                        jacobian=jacob,
                    )
                    model_data[:, start+j] = jim.ravel()

                    bgm_list.append(jgm0)

            fluxes, resid, rank, s = np.linalg.lstsq(
                model_data,
                obs.image.ravel(),
                rcond=None,
            )

            bpars = []
            for i, this_gm in enumerate(bgm_list):
                this_gm.set_flux(fluxes[i])
                bpars += list(this_gm.get_full_pars())

            bgm0 = ngmix.GMix(pars=bpars)
            bgm = bgm0.convolve(obs.psf.gmix)

            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            byband_gm.append(bgm)
            byband_pars.append(bgm.get_full_pars())
            byband_imlist.append(bim)
            byband_wtlist.append(obs.weight)

            tdiff = obs.image - bim
            difflist.append(tdiff)
            chi2 += (tdiff**2 * obs.weight).sum()

        this_tm = time.time() - tm0
        print('this time:', tm)
        tm += this_tm

        if show:
            rgb = make_rgb(imlist, wtlist, scale=viewscale)

            tmp_modlist = []
            for i in range(len(wtlist)):
                tnoise = np.sqrt(1/wtlist[i][0, 0])
                tmpim = byband_imlist[i].copy()
                tmpim += rng.normal(scale=tnoise, size=tmpim.shape)
                tmp_modlist.append(tmpim)
            model_rgb_noisy = make_rgb(
                tmp_modlist,
                byband_wtlist,
                scale=viewscale,
            )
            model_rgb = make_rgb(byband_imlist, byband_wtlist, scale=viewscale)

            diff_rgb = make_rgb(difflist, byband_wtlist, scale=viewscale)

            compare_rgb_images(rgb, model_rgb, diff_rgb,
                               seg, width, chi2,
                               model_noisy=model_rgb_noisy,
                               title=title)

        if ntrial > 1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return

    print('total time:', tm)
    print('time per:', tm/ntrial)


def test_descwl_lin(sim_config='dbsim.yaml',
                    model='bdf',
                    nobj=None,
                    ntrial=1,
                    tol=1.0e-3,
                    maxiter=40000,
                    scaling='linear',
                    noise_factor=1,  # for low noise sim
                    viewscale=0.0005,
                    show=False,
                    title=None,
                    width=1000,
                    seed=None):
    """
    true positions, ngauss related to T
    """
    import time

    rng = np.random.RandomState(seed)
    sim = make_dbsim(
        rng,
        noise_factor=noise_factor,
        sim_config=sim_config,
    )

    tm = 0.0
    for trial in range(ntrial):
        print('-'*70)
        print('%d/%d' % (trial+1, ntrial))

        sim.make_obs(nobj=nobj)
        mbobs = sim.obs

        imlist = [o[0].image for o in mbobs]
        wtlist = [o[0].weight for o in mbobs]

        for obslist in mbobs:
            for obs in obslist:
                do_psf_fit(obs.psf, rng)

        # coadd over bands
        coadd_obs = make_coadd_obs(mbobs)
        do_psf_fit(coadd_obs.psf, rng)

        noise = np.sqrt(1/coadd_obs.weight[0, 0])
        objs, seg = run_sep(coadd_obs.image, noise)

        print('found', objs.size, 'objects')
        if objs.size == 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        tm0 = time.time()

        gm_model_list = get_model_fits(
            model,
            coadd_obs.image,
            coadd_obs.weight,
            coadd_obs.psf,
            coadd_obs.jacobian,
            objs,
            seg,
            rng,
            # show=show,
            show=False,
            width=width,
        )
        if gm_model_list is None:
            if show:
                tab = biggles.Table(1, 2, aspect_ratio=0.5)
                rgb = make_rgb(imlist, wtlist, scale=viewscale)
                tab[0, 0] = images.view(rgb, show=False)
                tab[0, 1] = images.view(seg, show=False)
                tab.show(width=width, height=width*0.5)
            break

        byband_gm = []
        byband_pars = []
        byband_imlist = []
        byband_wtlist = []
        difflist = []
        chi2 = 0.0

        model_data = np.zeros((coadd_obs.image.size, objs.size))
        for band, obslist in enumerate(mbobs):
            obs = obslist[0]
            jacob = obs.jacobian
            scale = jacob.scale

            bgm_list = []
            for i in range(objs.size):

                this_gm0 = gm_model_list[i].copy()

                row = objs['y'][i]*scale
                col = objs['x'][i]*scale
                this_gm0.set_cen(row=row, col=col)
                this_gm0.set_flux(1.0)

                this_gm = this_gm0.convolve(obs.psf.gmix)
                try:
                    this_gm.set_norms()
                except GMixRangeError:
                    print('seeting zero')
                    tdata = this_gm0.get_data()
                    tdata['irr'] = 0.0
                    tdata['irc'] = 0.0
                    tdata['icc'] = 0.0
                    this_gm = this_gm0.convolve(obs.psf.gmix)
                    this_gm.set_norms()

                # print('%g %g %g' % this_gm.get_g1g2T())
                this_im = this_gm.make_image(
                    obs.image.shape,
                    jacobian=jacob,
                )
                model_data[:, i] = this_im.ravel()

                bgm_list.append(this_gm0)

            fluxes, resid, rank, s = np.linalg.lstsq(
                model_data,
                obs.image.ravel(),
                rcond=None,
            )

            bpars = []
            for i, this_gm0 in enumerate(bgm_list):
                this_gm0.set_flux(fluxes[i])
                bpars += list(this_gm0.get_full_pars())

            bgm0 = ngmix.GMix(pars=bpars)
            bgm = bgm0.convolve(obs.psf.gmix)

            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            byband_gm.append(bgm)
            byband_pars.append(bgm.get_full_pars())
            byband_imlist.append(bim)
            byband_wtlist.append(obs.weight)

            tdiff = obs.image - bim
            difflist.append(tdiff)
            chi2 += (tdiff**2 * obs.weight).sum()

        this_tm = time.time() - tm0
        print('this time:', tm)
        tm += this_tm

        if show:
            rgb = make_rgb(imlist, wtlist, scale=viewscale)

            tmp_modlist = []
            for i in range(len(wtlist)):
                tnoise = np.sqrt(1/wtlist[i][0, 0])
                tmpim = byband_imlist[i].copy()
                tmpim += rng.normal(scale=tnoise, size=tmpim.shape)
                tmp_modlist.append(tmpim)
            model_rgb_noisy = make_rgb(
                tmp_modlist,
                byband_wtlist,
                scale=viewscale,
            )
            model_rgb = make_rgb(byband_imlist, byband_wtlist, scale=viewscale)

            diff_rgb = make_rgb(difflist, byband_wtlist, scale=viewscale)

            compare_rgb_images(rgb, model_rgb, diff_rgb,
                               seg, width, chi2,
                               model_noisy=model_rgb_noisy,
                               title=title)

        if ntrial > 1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return

    print('total time:', tm)
    print('time per:', tm/ntrial)


def test_descwl(sim_config='dbsim.yaml',
                ponly=False,
                model='exp',
                nobj=None,
                ntrial=1,
                tol=1.0e-3,
                maxiter=40000,
                noise_factor=1,  # for low noise sim
                viewscale=0.0005,
                show=False,
                title=None,
                width=1000,
                seed=None):
    """
    true positions, ngauss related to T
    """
    import time
    import sep

    rng = np.random.RandomState(seed)
    sim = make_dbsim(
        rng,
        noise_factor=noise_factor,
        sim_config=sim_config,
    )

    ur = rng.uniform

    tm = 0.0
    for trial in range(ntrial):
        print('-'*70)
        print('%d/%d' % (trial+1, ntrial))

        sim.make_obs(nobj=nobj)
        mbobs = sim.obs

        imlist = [o[0].image for o in mbobs]
        wtlist = [o[0].weight for o in mbobs]

        # if show:
        #     dbsim.visualize.view_mbobs(sim.obs, scale=viewscale)

        for obslist in mbobs:
            for obs in obslist:
                do_psf_fit(obs.psf, rng)

        # coadd over bands
        coadd_obs = make_coadd_obs(mbobs)
        do_psf_fit(coadd_obs.psf, rng)

        noise = np.sqrt(1/coadd_obs.weight[0, 0])
        objs, seg = sep.extract(
            coadd_obs.image,
            0.8,
            err=noise,
            deblend_cont=1.0e-5,
            minarea=4,
            filter_kernel=KERNEL,
            segmentation_map=True,
        )

        print('found', objs.size, 'objects')
        if objs.size == 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        tm0 = time.time()

        gm_model_list = get_model_fits(
            model,
            coadd_obs.image,
            coadd_obs.weight,
            coadd_obs.psf,
            coadd_obs.jacobian,
            objs,
            seg,
            rng,
            # show=show,
            show=False,
            width=width,
        )
        if gm_model_list is None:
            if show:
                tab = biggles.Table(1, 2, aspect_ratio=0.5)
                rgb = make_rgb(imlist, wtlist, scale=viewscale)
                tab[0, 0] = images.view(rgb, show=False)
                tab[0, 1] = images.view(seg, show=False)
                tab.show(width=width, height=width*0.5)
            break

        imsky, sky = ngmix.em.prep_image(coadd_obs.image)
        emobs = Observation(imsky, jacobian=coadd_obs.jacobian)

        for itry in range(2):
            guess_pars = []

            scale = coadd_obs.jacobian.scale
            coff = 0.01*scale

            for i in range(objs.size):
                tgm = gm_model_list[i]
                T = tgm.get_T()
                if T <= 0.0:
                    T = 0.0
                    g1 = 0.0
                    g2 = 0.0
                else:
                    try:
                        g1, g2, T = tgm.get_g1g2T()
                    except GMixRangeError:
                        T = 0.0
                        g1 = 0.0
                        g2 = 0.0

                print('%d %g %g %g' % (i, g1, g2, T))

                # our dbsim obs have jacobian "center" set to 0, 0
                # row = objs['y'][i] - cen[0]
                # col = objs['x'][i] - cen[1]
                row = objs['y'][i]*scale
                col = objs['x'][i]*scale

                """
                fracdev = 0.5
                pars = [
                    row,
                    col,
                    g1,
                    g2,
                    T,
                    fracdev,
                    1.0,
                ]
                gm_model = ngmix.GMixBDF(pars=pars)
                """

                if model == 'bdf':
                    fracdev = tgm._pars[5]
                    pars = [
                        row,
                        col,
                        g1,
                        g2,
                        T,
                        fracdev,
                        1.0,
                    ]
                    gm_model = ngmix.GMixBDF(pars=pars)
                elif model == 'bd':
                    fracdev = tgm._pars[5]
                    logTratio = tgm._pars[6]
                    pars = [
                        row,
                        col,
                        g1,
                        g2,
                        T,
                        logTratio,
                        fracdev,
                        1.0,
                    ]
                    gm_model = ngmix.GMixModel(pars, model)

                else:
                    pars = [
                        row,
                        col,
                        g1,
                        g2,
                        T,
                        1.0,
                    ]
                    gm_model = ngmix.GMixModel(pars, model)

                gm_model = gm_model.convolve(coadd_obs.psf.gmix)

                # perturb the centers
                data = gm_model.get_data()

                if not ponly:
                    for j in range(data.size):
                        rowoff = ur(low=-coff, high=coff),
                        coloff = ur(low=-coff, high=coff),
                        data['row'][j] += rowoff
                        data['col'][j] += coloff

                guess_pars += list(gm_model.get_full_pars())

            gm_guess = ngmix.GMix(pars=guess_pars)
            # print('guess:')
            # print(gm_guess)

            if ponly:
                break

            em = GMixEMFixSize(emobs)
            em.go(gm_guess, sky, maxiter=maxiter, tol=tol)

            res = em.get_result()
            if res['flags'] == 0:
                break

        if not ponly and res['flags'] != 0:
            if show:
                view_rgb(imlist, wtlist, viewscale)
            break

        if ponly:
            gmfit = gm_guess
        else:
            gmfit = em.get_gmix()

            """
            print('best fit:')
            print(gmfit)
            """

            print('results')
            print(res)

        byband_gm = []
        byband_pars = []
        byband_imlist = []
        byband_wtlist = []
        difflist = []
        chi2 = 0.0

        for band, obslist in enumerate(mbobs):
            obs = obslist[0]
            imsky, sky = ngmix.em.prep_image(obs.image)
            emobs = Observation(imsky, jacobian=obs.jacobian)
            em = GMixEMPOnly(emobs)

            for itry in range(4):
                gm_guess = gmfit.copy()
                gdata = gm_guess.get_data()
                for idata in range(gdata.size):
                    gdata['p'][idata] *= (1.0 + ur(low=-0.01, high=0.01))

                if ponly:
                    use_sky = sky
                    print('raw use sky:', use_sky)
                    use_sky = use_sky*imsky.size*obs.jacobian.scale**2
                    print('use sky:', use_sky)
                else:
                    use_sky = res['sky']

                em.go(
                    gm_guess,
                    use_sky,
                    maxiter=500,
                    tol=0.001,
                    # tol=0.0001,
                )

                bres = em.get_result()
                if bres['flags'] == 0:
                    break

            if bres['flags'] != 0:
                if bres['message'] != 'OK':
                    raise RuntimeError('cannot proceed')

                print('did not succeed, going ahead anyway')

            bgm = em.get_gmix()
            print('band %d res:' % band)
            print(bres)

            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            bflux_tot, bflux_tot_err = get_flux(
                obs.image,
                obs.weight,
                bim,
            )
            bgm.set_flux(bflux_tot*obs.jacobian.scale**2)
            # bim *= bflux_tot/bim.sum()
            bim = bgm.make_image(
                obs.image.shape,
                jacobian=obs.jacobian,
            )

            byband_gm.append(bgm)
            byband_pars.append(bgm.get_full_pars())
            byband_imlist.append(bim)
            byband_wtlist.append(obs.weight)

            tdiff = obs.image - bim
            difflist.append(tdiff)
            chi2 += (tdiff**2 * obs.weight).sum()

        this_tm = time.time() - tm0
        print('this time:', tm)
        tm += this_tm

        if show:
            # rgb = dbsim.visualize.make_rgb(
            #     mbobs,
            #     scale=viewscale,
            # )
            rgb = make_rgb(imlist, wtlist, scale=viewscale)

            tmp_modlist = []
            for i in range(len(wtlist)):
                tnoise = np.sqrt(1/wtlist[i][0, 0])
                tmpim = byband_imlist[i].copy()
                tmpim += rng.normal(scale=tnoise, size=tmpim.shape)
                tmp_modlist.append(tmpim)
            model_rgb_noisy = make_rgb(
                tmp_modlist,
                byband_wtlist,
                scale=viewscale,
            )
            model_rgb = make_rgb(byband_imlist, byband_wtlist, scale=viewscale)

            diff_rgb = make_rgb(difflist, byband_wtlist, scale=viewscale)

            compare_rgb_images(rgb, model_rgb, diff_rgb,
                               seg, width, chi2,
                               model_noisy=model_rgb_noisy,
                               title=title)

            """
            grid = plotting.Grid(objs.size)
            trat = grid.nrow/(grid.ncol*2)
            eachtab = biggles.Table(grid.nrow, grid.ncol,
                                    aspect_ratio=trat)

            for i in range(objs.size):

                beg = i*ngauss_per*6
                end = (i+1)*ngauss_per*6

                gp = byband_pars[0][beg:end]
                rp = byband_pars[1][beg:end]
                ip = byband_pars[2][beg:end]

                # pravel = p.reshape(ngauss_per*ngauss_per)

                ggm = GMix(pars=gp)
                rgm = GMix(pars=rp)
                igm = GMix(pars=ip)

                gim = ggm.make_image(
                    coadd_obs.image.shape,
                    jacobian=coadd_obs.jacobian,
                )
                rim = rgm.make_image(
                    coadd_obs.image.shape,
                    jacobian=coadd_obs.jacobian,
                )
                iim = igm.make_image(
                    coadd_obs.image.shape,
                    jacobian=coadd_obs.jacobian,
                )

                gimn = gim + rng.normal(
                     scale=np.sqrt(1.0/wtlist[0][0, 0]),
                     size=gim.shape,
                )
                rimn = rim + rng.normal(
                     scale=np.sqrt(1.0/wtlist[1][0, 0]),
                     size=rim.shape,
                )
                iimn = iim + rng.normal(
                     scale=np.sqrt(1.0/wtlist[2][0, 0]),
                     size=iim.shape,
                )

                rgb = make_rgb([gim, rim, iim], byband_wtlist, scale=viewscale)
                rgb_noisy = make_rgb([gimn, rimn, iimn],
                                     byband_wtlist, scale=viewscale)

                grow, gcol = grid(i)
                title = 'model %d' % i
                eachtab[grow, gcol] = images.view_mosaic(
                    [rgb_noisy, rgb],
                    show=False,
                    title=title,
                )

            wfac = 1
            # eachtab.show(width=width*wfac, height=width*wfac*trat)
            fname = '/tmp/each.png'
            eachtab.write_img(width*wfac, width*wfac*trat, fname)
            show_image(fname)
            """

        if ntrial > 1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return

    print('total time:', tm)
    print('time per:', tm/ntrial)


def test_multi_sep_fromT(nobj=4,
                         ntrial=1,
                         off=10,
                         dim=75,
                         noise=0.1,
                         tol=1.0e-3,
                         maxiter=40000,
                         Tmean=4,
                         Tsig=10,
                         Fmin=10.0,
                         Fmax=90.0,
                         frac_stars=0.0,
                         sim_model='bd',
                         scaling='linear',
                         show=False,
                         width=1000,
                         seed=None):
    """
    true positions, ngauss related to T
    """
    import time
    import sep
    scale = 1.0

    rng = np.random.RandomState(seed)

    Tpdf = ngmix.priors.LogNormal(Tmean, Tsig, rng=rng)
    gpdf = ngmix.priors.GPriorBA(0.3, rng=rng)

    ur = rng.uniform

    dims = [dim]*2

    cen = (np.array(dims)-1.0)/2.0
    jacob = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

    Tpsf = 4.0
    psf = ngmix.GMixModel(
        [0.0, 0.0, 0.0, 0.0, Tpsf, 1.0],
        'gauss',
    )
    psf_dims = [25]*2
    psf_cen = (np.array(psf_dims)-1)/2
    psf_jac = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    psf_im = psf.make_image(psf_dims, jacobian=psf_jac)
    psf_noise = 0.0001
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)

    psf_weight = psf_im + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_weight,
        jacobian=psf_jac,
    )
    do_psf_fit(psf_obs, rng)

    for trial in range(ntrial):
        true_flux_tot = 0.0
        for i in range(nobj):

            F = rng.uniform(low=Fmin, high=Fmax)
            true_flux_tot += F

            tcen = ur(low=-off, high=+off, size=2)

            num = rng.uniform()
            if num < frac_stars:
                print('%d star' % i)
                gm = psf.copy()
                gm.set_cen(row=tcen[0], col=tcen[1])
                gm.set_flux(F)
            else:

                g1, g2 = gpdf.sample2d()

                # T = rng.uniform(low=Tmin, high=Tmax)
                T = Tpdf.sample()
                print('%d %g' % (i, T))

                if sim_model == 'bd':
                    fracdev = rng.uniform(low=0.0, high=1.0)
                    TdByTe = rng.uniform(0.2, 1.8)

                    print('%d fracdev: %.3f '
                          'Td/Te: %.3f' % (i, fracdev, TdByTe))

                    pars = [tcen[0], tcen[1], g1, g2, T, fracdev, F]
                    gm0 = ngmix.GMixBDF(pars=pars, TdByTe=TdByTe)
                else:
                    pars = [tcen[0], tcen[1], g1, g2, T, F]
                    gm0 = ngmix.GMixModel(pars, sim_model)

                # perturb the centers significantly
                data0 = gm0.get_data()

                model_sigma = np.sqrt(T/2)
                coff = 0.1*model_sigma
                for j in range(data0.size):
                    rowoff = ur(low=-coff, high=coff),
                    coloff = ur(low=-coff, high=coff),
                    data0['row'][j] += rowoff
                    data0['col'][j] += coloff

                gm = gm0.convolve(psf)

            sim = gm.make_image(dims, jacobian=jacob)

            if i == 0:
                im0 = sim.copy()
            else:
                im0 += sim

        noise_image = rng.normal(scale=noise, size=im0.shape)
        im = im0 + noise_image
        weight = im*0 + 1.0/noise**2

        objs, seg = sep.extract(
            im,
            0.8,
            err=noise,
            deblend_cont=1.0e-5,
            minarea=4,
            filter_kernel=KERNEL,
            segmentation_map=True,
        )

        print('found', objs.size, 'objects')
        if objs.size == 0:
            if show:
                images.view_mosaic([im, seg])
            break

        gm_model_list = get_model_fits(
            im,
            weight,
            psf_obs,
            jacob,
            objs,
            seg,
            rng,
            show=show,
            width=width,
        )
        if gm_model_list is None:
            if show:
                images.view_mosaic([im, seg])
            break

        imsky, sky = ngmix.em.prep_image(im)
        obs = Observation(imsky, jacobian=jacob)

        guess_pars = []

        coff = 0.01*scale

        for i in range(objs.size):
            tgm = gm_model_list[i]
            g1, g2, T = tgm.get_g1g2T()
            print('%d %g %g %g' % (i, g1, g2, T))

            row = objs['y'][i] - cen[0]
            col = objs['x'][i] - cen[1]

            pars = [
                row,
                col,
                g1,
                g2,
                T,
                1.0,
            ]
            gm_model = ngmix.GMixModel(pars, 'exp')

            gm_model = gm_model.convolve(psf)

            # perturb the centers
            data = gm_model.get_data()
            ngauss_per = data.size

            for j in range(data.size):
                rowoff = ur(low=-coff, high=coff),
                coloff = ur(low=-coff, high=coff),
                data['row'][j] += rowoff
                data['col'][j] += coloff

            guess_pars += list(gm_model.get_full_pars())

        gm_guess = ngmix.GMix(pars=guess_pars)
        print('guess:')
        print(gm_guess)

        tm0 = time.time()
        em = GMixEMFixSize(obs)
        em.go(gm_guess, sky, maxiter=maxiter, tol=tol)

        gmfit = em.get_gmix()
        res = em.get_result()
        print('best fit:')
        print(gmfit)
        print('results')
        print(res)

        imfit = gmfit.make_image(im.shape)

        flux_tot, flux_tot_err = get_flux(im, weight, imfit)
        print('true flux tot:', true_flux_tot)
        print('flux: %g +/- %g' % (flux_tot, flux_tot_err))
        gmfit.set_flux(flux_tot)

        imfit *= flux_tot/imfit.sum()

        tm = time.time() - tm0
        print('time:', tm)

        if show:
            tab, maxval = compare_images(im, weight, imfit, scaling=scaling)
            tab[1, 1] = images.view(seg, show=False, title='seg')

            tab.show(width=width, height=width)

            full_pars = gmfit.get_full_pars().reshape(
                (ngauss_per*objs.size, 6),
            )

            grid = plotting.Grid(objs.size)

            trat = grid.nrow/grid.ncol
            eachtab = biggles.Table(grid.nrow, grid.ncol,
                                    aspect_ratio=trat)

            for i in range(objs.size):

                beg = i*ngauss_per
                end = (i+1)*ngauss_per

                p = full_pars[beg:end]

                pravel = p.reshape(ngauss_per*6)

                gmi = GMix(pars=pravel)
                tflux = gmi.get_flux()

                ptup = (i, tflux, flux_tot_err*tflux/flux_tot)
                print('flux %d %g +/- %g' % ptup)

                imfiti = gmi.make_image(im.shape, jacobian=jacob)
                imfiti += noise_image

                grow, gcol = grid(i)
                title = 'model %d' % i
                eachtab[grow, gcol] = images.view(imfiti, show=False,
                                                  title=title)

            eachtab.show(width=width, height=width*trat)

        if ntrial > 1 and show:
            if input('enter a key, q to quit: ') == 'q':
                return


def test_multi_truepos_fromT(nobj=4,
                             off=10,
                             dim=75,
                             noise=0.1,
                             tol=1.0e-3,
                             maxiter=40000,
                             Tmean=4,
                             Tsig=10,
                             Fmin=10.0,
                             Fmax=90.0,
                             sim_model='bd',
                             model='bdf',
                             scaling='linear',
                             show=False,
                             width=1000,
                             seed=None):
    """
    true positions, ngauss related to T
    """
    import time
    scale = 1.0

    rng = np.random.RandomState(seed)

    Tpdf = ngmix.priors.LogNormal(Tmean, Tsig, rng=rng)
    gpdf = ngmix.priors.GPriorBA(0.3, rng=rng)

    ur = rng.uniform

    dims = [dim]*2

    cen = (np.array(dims)-1.0)/2.0
    jacob = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

    Tpsf = 4.0
    psf = ngmix.GMixModel(
        [0.0, 0.0, 0.0, 0.0, Tpsf, 1.0],
        'gauss',
    )

    simlist = []
    gmlist = []
    for i in range(nobj):
        tcen = ur(low=-off, high=+off, size=2)

        g1, g2 = gpdf.sample2d()

        # T = rng.uniform(low=Tmin, high=Tmax)
        T = Tpdf.sample()

        F = rng.uniform(low=Fmin, high=Fmax)

        if sim_model == 'bd':
            fracdev = rng.uniform(low=0.0, high=1.0)
            TdByTe = rng.uniform(0.2, 1.8)

            print('%d fracdev: %.3f Td/Te: %.3f' % (i+1, fracdev, TdByTe))

            pars = [tcen[0], tcen[1], g1, g2, T, fracdev, F]
            gm0 = ngmix.GMixBDF(pars=pars, TdByTe=TdByTe)
        else:
            pars = [tcen[0], tcen[1], g1, g2, T, F]
            gm0 = ngmix.GMixModel(pars, sim_model)

        # perturb the centers significantly
        data0 = gm0.get_data()

        model_sigma = np.sqrt(T/2)
        coff = 0.1*model_sigma
        for j in range(data0.size):
            rowoff = ur(low=-coff, high=coff),
            coloff = ur(low=-coff, high=coff),
            data0['row'][j] += rowoff
            data0['col'][j] += coloff

        gm = gm0.convolve(psf)

        sim = gm.make_image(dims, jacobian=jacob)
        simlist.append(sim)

        if i == 0:
            im0 = sim.copy()
        else:
            im0 += sim

        gmlist.append(gm0)

    noise_image = rng.normal(scale=noise, size=im0.shape)
    im = im0 + noise_image

    guess_pars = []

    coff = 0.01*scale
    for i in range(nobj):
        gm = gmlist[i]
        g1, g2, T = gm.get_g1g2T()
        row, col = gm.get_cen()

        g1, g2 = get_shape_guess(g1, g2, 0.05, rng)

        if model == 'bdf':
            fracdev = 0.5
            pars = [
                row,
                col,
                g1,
                g2,
                T,
                fracdev,
                1.0,
            ]
            gm_model = ngmix.GMixBDF(pars=pars)
        else:
            pars = [
                row,
                col,
                g1,
                g2,
                T,
                1.0,
            ]
            gm_model = ngmix.GMixModel(pars, model)

        gm_model = gm_model.convolve(psf)

        # perturb the centers
        data = gm_model.get_data()
        ngauss_per = data.size

        for j in range(data.size):
            rowoff = ur(low=-coff, high=coff),
            coloff = ur(low=-coff, high=coff),
            data['row'][j] += rowoff
            data['col'][j] += coloff

        guess_pars += list(gm_model.get_full_pars())

    gm_guess = ngmix.GMix(pars=guess_pars)
    print('guess:')
    print(gm_guess)

    imsky, sky = ngmix.em.prep_image(im)

    obs = Observation(imsky, jacobian=jacob)

    tm0 = time.time()

    em = GMixEMFixSize(obs)
    em.go(gm_guess, sky, maxiter=maxiter, tol=tol)

    gmfit = em.get_gmix()
    res = em.get_result()
    print('best fit:')
    print(gmfit)
    print('results')
    print(res)

    imfit = gmfit.make_image(im.shape)

    weight = im*0 + 1.0/noise**2
    flux_tot, flux_tot_err = get_flux(im, weight, imfit)
    print('flux: %g +/- %g' % (flux_tot, flux_tot_err))
    gmfit.set_flux(flux_tot)

    imfit *= flux_tot/imfit.sum()

    tm = time.time() - tm0
    print('time:', tm)

    if show:
        import biggles
        import plotting

        tab, maxval = compare_images(im, weight, imfit, scaling=scaling)

        tab.show(width=width, height=width)

        full_pars = gmfit.get_full_pars().reshape((ngauss_per*nobj, 6))

        grid = plotting.Grid(nobj)

        trat = grid.nrow/grid.ncol
        eachtab = biggles.Table(grid.nrow, grid.ncol,
                                aspect_ratio=trat)

        beg = 0
        for i in range(nobj):

            end = beg + ngauss_per

            p = full_pars[beg:end]

            pravel = p.reshape(ngauss_per*6)

            gmi = GMix(pars=pravel)
            tflux = gmi.get_flux()

            ptup = (i, tflux, flux_tot_err*tflux/flux_tot)
            print('flux %d %g +/- %g' % ptup)

            imfiti = gmi.make_image(im.shape, jacobian=jacob)
            sim = simlist[i]

            tabi, _ = compare_images(
                sim + noise_image,
                weight,
                imfiti,
                maxval=maxval,
                scaling=scaling,
                label='image %d' % (i+1),
            )

            grow, gcol = grid(i)
            eachtab[grow, gcol] = tabi

            beg += ngauss_per

        eachtab.show(width=width, height=width*trat)


def test_multi_truepos_ps(nobj=4,
                          off=10,
                          dim=75,
                          cen_dist='psf',
                          min_ngauss=1,
                          noise=0.1,
                          tol=1.0e-3,
                          maxiter=40000,
                          Tmin=0.0,
                          Tmax=48.0,
                          Fmin=10.0,
                          Fmax=90.0,
                          show=False,
                          width=1000,
                          seed=None):
    """
    true positions, ngauss related to T
    """
    import time

    rng = np.random.RandomState(seed)

    gpdf = ngmix.priors.GPriorBA(0.3, rng=rng)

    ur = rng.uniform

    dims = [dim]*2

    cen = (np.array(dims)-1.0)/2.0
    jacob = UnitJacobian(row=cen[0], col=cen[1])

    Tpsf = 4.0
    sigma_psf = np.sqrt(Tpsf/2)
    psf = ngmix.GMixModel(
        [0.0, 0.0, 0.0, 0.0, Tpsf, 1.0],
        'gauss',
    )

    area_fac = 2
    psf_area = area_fac*Tpsf
    ngausses = np.zeros(nobj, dtype='i4')

    simlist = []
    gmlist = []
    for i in range(nobj):
        tcen = ur(low=-off, high=+off, size=2)

        g1, g2 = gpdf.sample2d()

        T = rng.uniform(low=Tmin, high=Tmax)

        F = rng.uniform(low=Fmin, high=Fmax)

        pars = [tcen[0], tcen[1], g1, g2, T, F]

        val = rng.uniform()
        if val < 0.5:
            ms = 'exp'
        else:
            ms = 'dev'

        print(i, ms)
        gm = ngmix.GMixModel(
            pars,
            ms,
        )

        gm = gm.convolve(psf)

        sim = gm.make_image(dims, jacobian=jacob)
        simlist.append(sim)

        if i == 0:
            im0 = sim.copy()
        else:
            im0 += sim

        obsT = gm.get_T()
        area = area_fac*obsT
        ngausses[i] = int(2*area/psf_area)

        if ngausses[i] < min_ngauss:
            ngausses[i] = min_ngauss

        gmlist.append(gm)

    noise_image = rng.normal(scale=noise, size=im0.shape)
    im = im0 + noise_image

    print('ngauss:', ngausses)

    ngauss = ngausses.sum()

    guess_pars = np.zeros((ngauss, 6))

    Tguess = Tpsf
    beg = 0
    for i in range(nobj):
        tngauss = ngausses[i]
        end = beg + tngauss

        gm = gmlist[i]
        row, col = gm.get_cen()

        fac = (1.0 + ur(low=-0.05, high=0.05, size=tngauss))
        guess_pars[beg:end, 0] = 1.0/ngauss * fac

        if cen_dist == 'tight':
            coff = 0.01
            rowoff = ur(low=-coff, high=coff, size=tngauss)
            coloff = ur(low=-coff, high=coff, size=tngauss)
            guess_pars[beg:end, 1] = row + rowoff
            guess_pars[beg:end, 2] = col + coloff
        elif cen_dist == 'psf':
            rowoff = rng.normal(scale=sigma_psf, size=tngauss)
            coloff = rng.normal(scale=sigma_psf, size=tngauss)
            guess_pars[beg:end, 1] = row + rowoff
            guess_pars[beg:end, 2] = col + coloff
        else:
            raise ValueError('bad cen dist: %s' % cen_dist)

        guess_pars[beg:end, 3] = Tguess/2
        guess_pars[beg:end, 4] = 0.0
        guess_pars[beg:end, 5] = Tguess/2

        beg += tngauss

    guess_pars = guess_pars.ravel()

    gm_guess = ngmix.GMix(pars=guess_pars)
    print('guess:')
    print(gm_guess)

    imsky, sky = ngmix.em.prep_image(im)

    obs = Observation(imsky, jacobian=jacob)

    tm0 = time.time()

    em = GMixEMFixSize(obs)
    em.go(gm_guess, sky, maxiter=maxiter, tol=tol)

    gmfit = em.get_gmix()
    res = em.get_result()
    print('best fit:')
    print(gmfit)
    print('results')
    print(res)

    imfit = gmfit.make_image(im.shape)

    weight = im*0 + 1.0/noise**2
    flux_tot, flux_tot_err = get_flux(im, weight, imfit)
    print('flux: %g +/- %g' % (flux_tot, flux_tot_err))
    gmfit.set_flux(flux_tot)

    imfit *= flux_tot/imfit.sum()

    tm = time.time() - tm0
    print('time:', tm)

    if show:
        import biggles
        import plotting

        tab, maxval = compare_images(im, imfit)

        tab.show(width=width, height=width*2/3)

        full_pars = gmfit.get_full_pars().reshape((ngauss, 6))

        grid = plotting.Grid(nobj)

        trat = grid.nrow/grid.ncol
        eachtab = biggles.Table(grid.nrow, grid.ncol,
                                aspect_ratio=trat)

        beg = 0
        for i in range(nobj):
            tngauss = ngausses[i]

            end = beg + tngauss

            p = full_pars[beg:end]

            pravel = p.reshape(tngauss*6)
            gmi = GMix(pars=pravel)
            tflux = gmi.get_flux()

            ptup = (i, tflux, flux_tot_err*tflux/flux_tot)
            print('flux %d %g +/- %g' % ptup)

            imfiti = gmi.make_image(im.shape, jacobian=jacob)
            sim = simlist[i]

            tabi, _ = compare_images(
                sim + noise_image,
                imfiti,
                maxval=maxval,
                label='image %d' % (i+1),
            )

            grow, gcol = grid(i)
            eachtab[grow, gcol] = tabi

            beg += tngauss

        eachtab.show(width=width, height=width*trat)

    print('ngauss:', ngausses)


def test_multi_sep_ps(nobj=4,
                      off=10,
                      cen_dist='psf',
                      min_ngauss=1,
                      noise=0.1,
                      tol=1.0e-3,
                      maxiter=40000,
                      Tmin=0.0,
                      Tmax=48.0,
                      Tfac=1.0,
                      Fmin=10.0,
                      Fmax=90.0,
                      show=False,
                      title=None,
                      viewscale=0.0005,
                      width=1000,
                      seed=None):
    import time
    import sep

    rng = np.random.RandomState(seed)

    gpdf = ngmix.priors.GPriorBA(0.3, rng=rng)

    ur = rng.uniform

    dims = [75, 75]

    cen = (np.array(dims)-1.0)/2.0
    jacob = UnitJacobian(row=cen[0], col=cen[1])

    Tpsf = 4.0
    sigma_psf = np.sqrt(Tpsf/2)
    psf = ngmix.GMixModel(
        [0.0, 0.0, 0.0, 0.0, Tpsf, 1.0],
        'gauss',
    )

    for i in range(nobj):
        tcen = ur(low=-off, high=+off, size=2)

        g1, g2 = gpdf.sample2d()

        T = rng.uniform(low=Tmin, high=Tmax)
        F = rng.uniform(low=Fmin, high=Fmax)

        pars = [tcen[0], tcen[1], g1, g2, T, F]

        val = rng.uniform()
        if val < 0.5:
            ms = 'exp'
        else:
            ms = 'dev'

        # print(i, ms)
        gm = ngmix.GMixModel(
            pars,
            ms,
        )

        gm = gm.convolve(psf)

        sim = gm.make_image(dims, jacobian=jacob)
        if i == 0:
            im0 = sim.copy()
        else:
            im0 += sim

    noise_image = rng.normal(scale=noise, size=im0.shape)
    im = im0 + noise_image

    # wt = im0*0 + 1.0/noise**2

    objs, seg = sep.extract(
        im,
        0.8,
        err=noise,
        deblend_cont=1.0e-5,
        minarea=4,
        filter_kernel=KERNEL,
        segmentation_map=True,
    )
    if objs.size == 0:
        print('no objects')
        return

    add_dt = [('isoarea_image', 'f4'), ('ngauss', 'i4')]
    objs = eu.numpy_util.add_fields(objs, add_dt)

    psf_area = 2*Tpsf
    # psf_area = 4*Tpsf
    for i in range(objs.size):
        w = np.where(seg == (i+1))
        area = w[0].size
        tngauss = int(area/psf_area)
        if tngauss < min_ngauss:
            tngauss = min_ngauss
        objs['isoarea_image'][i] = area
        objs['ngauss'][i] = tngauss

    print('ngauss:', objs['ngauss'])
    # return

    nobj_det = objs.size

    ngauss = objs['ngauss'].sum()

    guess_pars = np.zeros((ngauss, 6))

    Tguess = Tpsf*Tfac
    beg = 0
    for i in range(nobj_det):
        # beg = i*ngauss_per
        # end = (i+1)*ngauss_per
        tngauss = objs['ngauss'][i]
        end = beg + tngauss

        row = objs['y'][i] - cen[0]
        col = objs['x'][i] - cen[1]

        fac = (1.0 + ur(low=-0.05, high=0.05, size=tngauss))
        guess_pars[beg:end, 0] = 1.0/ngauss * fac

        if cen_dist == 'tight':
            coff = 0.01
            rowoff = ur(low=-coff, high=coff, size=tngauss)
            coloff = ur(low=-coff, high=coff, size=tngauss)
            guess_pars[beg:end, 1] = row + rowoff
            guess_pars[beg:end, 2] = col + coloff
        elif cen_dist == 'psf':
            rowoff = rng.normal(scale=sigma_psf, size=tngauss)
            coloff = rng.normal(scale=sigma_psf, size=tngauss)
            guess_pars[beg:end, 1] = row + rowoff
            guess_pars[beg:end, 2] = col + coloff
        else:
            raise ValueError('bad cen dist: %s' % cen_dist)

        guess_pars[beg:end, 3] = Tguess/2
        guess_pars[beg:end, 4] = 0.0
        guess_pars[beg:end, 5] = Tguess/2

        beg += tngauss

    guess_pars = guess_pars.ravel()

    gm_guess = ngmix.GMix(pars=guess_pars)
    # print('guess:')
    # print(gm_guess)

    imsky, sky = ngmix.em.prep_image(im)

    obs = Observation(imsky, jacobian=jacob)

    tm0 = time.time()

    em = GMixEMFixSize(obs)
    em.go(gm_guess, sky, maxiter=maxiter, tol=tol)

    gmfit = em.get_gmix()
    res = em.get_result()
    # print('best fit:')
    # print(gmfit)
    print('results')
    print(res)

    imfit = gmfit.make_image(im.shape)

    weight = im*0 + 1.0/noise**2
    flux_tot, flux_tot_err = get_flux(im, weight, imfit)
    print('flux: %g +/- %g' % (flux_tot, flux_tot_err))
    gmfit.set_flux(flux_tot)

    flux_fac = flux_tot/imfit.sum()
    imfit *= flux_fac

    tm = time.time() - tm0
    print('time:', tm)

    if show:

        tab = compare_images(
            im,
            weight,
            imfit,
            seg,
            scale=viewscale,
        )
        if title is not None:
            tab.title = title

        tab.show(width=width, height=width)
