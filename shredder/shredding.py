import numpy as np
import logging
import ngmix
from ngmix.em import EM_MAXITER

from .em import (
    GMixEMFixCen,
    GMixEMPOnly,
)
from . import procflags
from . import coadding
from . import vis
from .psf_fitting import do_psf_fit

logger = logging.getLogger(__name__)


class Shredder(object):
    def __init__(self,
                 mbobs,
                 miniter=10,
                 maxiter=1000,
                 fill_zero_weight=False,
                 tol=0.001,
                 rng=None):
        """
        Parameters
        ----------
        obs: observations
            Typcally an ngmix.MultiBandObsList, but can also
            be a simple Observation

            The image should not have zero or negative pixels. You can
            use the ngmix.em.prep_image() function to ensure this.
        miniter: int, optional
            Mininum number of iterations, default 10
        maxiter: int, optional
            Maximum number of iterations, default 1000
        tol: number, optional
            The tolerance in the weighted logL, default 1.e-3
        rng: random number generator
            E.g. np.random.RandomState.
        """

        # TODO deal with Observation input, which would only use
        # the "coadd" result and would not actually coadd

        if fill_zero_weight:
            self._ignore_zero_weight = False
        else:
            self._ignore_zero_weight = True

        self.mbobs = mbobs
        do_psf_fit(self.mbobs)

        self.coadd_obs = coadding.make_coadd_obs(
            mbobs,
            ignore_zero_weight=self._ignore_zero_weight,
        )
        do_psf_fit(self.coadd_obs)

        self.miniter = miniter
        self.maxiter = maxiter
        self.tol = tol

        if rng is None:
            rng = np.random.RandomState()

        self._rng = rng

    def get_result(self):
        """
        get the result dictionary
        """
        if not hasattr(self, '_result'):
            raise RuntimeError('run shred() first')
        return self._result

    def get_model_images(self):
        """
        get a list of model images for all bands
        """
        imlist = []

        for band in range(len(self.mbobs)):
            image = self.get_model_image(band=band)
            imlist.append(image)
        return imlist

    def get_model_image(self, band=None):
        """
        get a model image for the specified band
        """

        res = self.get_result()
        gm = res['band_results'][band]['gmix_convolved']

        dims = self.mbobs[band][0].image.shape
        jacob = self.mbobs[band][0].jacobian
        return gm.make_image(dims, jacobian=jacob)

    def plot_comparison(self, **kw):
        """
        visualize a comparison of the model and data
        """
        models = self.get_model_images()
        return vis.compare_mbobs_and_models(
            self.mbobs,
            models,
            **kw
        )

    def shred(self, gmix_guess):
        """
        perform deblending

        Parameters
        ----------
        gmix_guess: ngmix.GMix
            A guess for the deblending
        """

        em_coadd = self._do_coadd_fit(
            gmix_guess,
        )

        self._result = {
            'flags': 0,
        }
        res = self._result

        cres = em_coadd.get_result()
        res['coadd_result'] = cres
        res['coadd_fitter'] = em_coadd

        res['flags'] != cres['flags']

        if cres['flags'] != 0 and cres['flags'] & EM_MAXITER == 0:
            # we failed in such a way that we cannot proceed
            res['flags'] |= procflags.COADD_FAILURE
        else:
            cres['gmix'] = em_coadd.get_gmix()

            self._do_multiband_fit()

    def _do_coadd_fit(self, gmix_guess):
        """
        run the fixed-center em fitter on the coadd image
        """

        coadd_obs = self.coadd_obs

        imsky, sky = ngmix.em.prep_image(coadd_obs.image)

        emobs = ngmix.Observation(
            imsky,
            weight=coadd_obs.weight,
            jacobian=coadd_obs.jacobian,
            psf=coadd_obs.psf,
            ignore_zero_weight=self._ignore_zero_weight,
        )

        em = GMixEMFixCen(
            emobs,
            miniter=self.miniter,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        em.go(gmix_guess, sky)

        return em

    def _do_multiband_fit(self):
        """
        tweak the mixture for each band and set the total flux
        """
        res = self._result

        reslist = []
        fitters = []

        for band, obslist in enumerate(self.mbobs):
            obs = obslist[0]

            em = self._get_band_fit(obs)
            fitters.append(em)

            bres = em.get_result()

            logger.debug('band %d: %s' % (band, repr(bres)))

            if bres['flags'] != 0 and bres['flags'] & EM_MAXITER == 0:
                logger.info('could not get flux fit for band %d' % band)
                res['flags'] |= procflags.BAND_FAILURE
            else:
                bres = self._set_band_flux_and_gmix(obs, em)

            reslist.append(bres)

        res['band_results'] = reslist
        res['band_fitters'] = fitters

    def _get_band_fit(self, obs):
        """
        get the flux-only fit for a band
        """

        imsky, sky = ngmix.em.prep_image(obs.image)
        emobs = ngmix.Observation(
            imsky,
            weight=obs.weight,
            jacobian=obs.jacobian,
            psf=obs.psf,
            ignore_zero_weight=self._ignore_zero_weight,
        )
        em = GMixEMPOnly(
            emobs,
            miniter=self.miniter,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        gm_guess = self._get_band_guess()

        use_sky = sky
        em.go(gm_guess, use_sky)

        return em

    def _set_band_flux_and_gmix(self, obs, em):
        """
        get the total flux and set it in the mixture

        also set the mixture in the result
        """

        res = em.get_result()

        gm = em.get_gmix()
        gm_convolved = em.get_convolved_gmix()

        bim = gm_convolved.make_image(
            obs.image.shape,
            jacobian=obs.jacobian,
        )

        flux, flux_err = get_flux(
            obs.image,
            obs.weight,
            bim,
        )

        flux = flux*obs.jacobian.scale**2
        flux_err = flux_err*obs.jacobian.scale**2
        gm.set_flux(flux)
        gm_convolved.set_flux(flux)

        res['gmix'] = gm
        res['gmix_convolved'] = gm_convolved
        res['total_flux'] = flux
        res['total_flux_err'] = flux

        return res

    def _get_band_guess(self):
        """
        get a guess for the band based on the coadd mixture
        """
        rng = self._rng
        gmix_guess = self._result['coadd_result']['gmix'].copy()

        gdata = gmix_guess.get_data()
        for idata in range(gdata.size):
            rnum = rng.uniform(low=-0.01, high=0.01)
            gdata['p'][idata] *= (1.0 + rnum)

        return gmix_guess


def get_flux_fill(im, wt, model_in):
    """
    use the input model as a template and infer the total
    flux

    Parameters
    ----------
    im: array
        The data image
    wt: array
        The data weight map
    model: array
        The input model image

    Returns
    -------
    flux, flux_err.  Estimates of the total flux and uncertainty
    """

    flux, flux_err = get_flux(im, wt, model_in)
    print('first pass:', flux, flux_err)

    wbad = np.where(wt <= 0.0)
    if wbad[0].size > 0:
        imfill = im.copy()
        wtfill = wt.copy()

        modfill = model_in.copy()
        modfill[:, :] *= flux
        imfill[wbad] = modfill[wbad]
        wtfill[wbad] = wt.max()

        flux, flux_err = get_flux(imfill, wtfill, model_in)
        print('second pass:', flux, flux_err)

    return flux, flux_err


def get_flux(im, wt, model_in):
    """
    use the input model as a template and infer the total
    flux

    Parameters
    ----------
    im: array
        The data image
    wt: array
        The data weight map
    model: array
        The input model image

    Returns
    -------
    flux, flux_err.  Estimates of the total flux and uncertainty
    """
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
        logger.info('cannot calculate err')
    else:

        arg = chi2/msq_sum/(im.size-1)
        if arg >= 0.0:
            flux_err = np.sqrt(arg)
        else:
            logger.info('cannot calculate err')

    return flux, flux_err
