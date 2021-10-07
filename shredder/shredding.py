import logging
import ngmix
from ngmix.flags import EM_MAXITER

from . import procflags
from . import coadding
from . import vis
from .psf_fitting import do_psf_fit
from .sexceptions import PSFFailure

logger = logging.getLogger(__name__)


class Shredder(object):
    def __init__(
        self,
        obs,
        psf_ngauss,
        rng,
        miniter=40,
        maxiter=500,
        flux_miniter=20,
        flux_maxiter=500,
        tol=0.001,
        vary_sky=False,
    ):
        """
        Parameters
        ----------
        obs: observations
            Typcally an ngmix.MultiBandObsList
        psf_ngauss: int
            Number of gaussians for psf
        rng: random number generator
            E.g. np.random.RandomState.
        miniter: int, optional
            Mininum number of iterations, default 40
        maxiter: int, optional
            Maximum number of iterations, default 1000
        flux_miniter: int, optional
            Mininum number of iterations for flux fits, default 20
        flux_maxiter: int, optional
            Maximum number of iterations for flux fits, default 1000
        tol: number, optional
            The tolerance in the weighted logL, default 1.e-3
        vary_sky: bool, optional
            If True, vary the sky
        """

        # TODO deal with Observation input, which would only use
        # the "coadd" result and would not actually coadd

        self.mbobs = obs

        self.rng = rng

        self.miniter = miniter
        self.maxiter = maxiter

        self.flux_miniter = flux_miniter
        self.flux_maxiter = flux_maxiter

        self.tol = tol
        self.vary_sky = vary_sky

        self.coadd_psfres = self._do_psf_fits(self.mbobs, psf_ngauss)

        self.coadd_obs = coadding.make_coadd_obs(self.mbobs)

        self.band_psfres = self._do_psf_fits(self.coadd_obs, psf_ngauss)

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
        gm = res['band_gmix_convolved'][band]

        dims = self.mbobs[band][0].image.shape
        jacob = self.mbobs[band][0].jacobian
        return gm.make_image(dims, jacobian=jacob)

    def plot(self, **kw):
        """
        make a plot of the image
        """
        return vis.view_mbobs(self.mbobs, **kw)

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

        self._result = {'flags': 0}
        res = self._result

        if self.coadd_psfres['flags'] != 0:
            res['flags'] = self.coadd_psfres['flags']
            return

        elif self.band_psfres['flags'] != 0:
            res['flags'] = self.band_psfres['flags']
            return

        coadd_result = self._do_coadd_fit(gmix_guess)

        logger.info('coadd: %s' % repr(coadd_result))

        res['coadd_result'] = coadd_result
        res['coadd_psf_gmix'] = self.coadd_obs.psf.gmix.copy()

        if coadd_result.has_gmix():
            res['coadd_gmix'] = coadd_result.get_gmix()
            res['coadd_gmix_convolved'] = coadd_result.get_convolved_gmix()
        else:
            res['coadd_gmix'] = None
            res['coadd_gmix_convolved'] = None

        if (
            coadd_result['flags'] != 0 and
            coadd_result['flags'] & EM_MAXITER == 0
        ):
            # we cannot proceed without the coadd fit
            res['flags'] |= procflags.COADD_FAILURE
        else:
            self._do_multiband_fit()

    def _do_coadd_fit(self, gmix_guess):
        """
        run the fixed-center em fitter on the coadd image
        """

        emobs, sky = ngmix.em.prep_obs(self.coadd_obs)

        em = ngmix.em.EMFitterFixCen(
            miniter=self.miniter,
            maxiter=self.maxiter,
            tol=self.tol,
            vary_sky=self.vary_sky,
        )

        return em.go(obs=emobs, guess=gmix_guess, sky=sky)

    def _do_multiband_fit(self):
        """
        tweak the mixture for each band and set the total flux
        """
        res = self._result

        reslist = []
        pgmlist = []
        gmlist = []
        gmclist = []

        for band, obslist in enumerate(self.mbobs):
            obs = obslist[0]

            pgmlist.append(obs.psf.gmix.copy())

            band_result = self._get_band_fit(obs)

            logger.info('band %d: %s' % (band, repr(band_result)))

            if (
                band_result['flags'] != 0 and
                band_result['flags'] & EM_MAXITER == 0
            ):
                logger.info('could not get flux fit for band %d' % band)
                res['flags'] |= procflags.BAND_FAILURE

            reslist.append(band_result)

            if band_result.has_gmix():
                gm = band_result.get_gmix()
                gm_convolved = band_result.get_convolved_gmix()
            else:
                gm = None
                gm_convolved = None

            gmlist.append(gm)
            gmclist.append(gm_convolved)

        res['band_results'] = reslist
        res['band_psf_gmix'] = pgmlist
        res['band_gmix'] = gmlist
        res['band_gmix_convolved'] = gmclist

    def _get_band_fit(self, obs):
        """
        get the flux-only fit for a band
        """

        emobs, sky = ngmix.em.prep_obs(obs)

        em = ngmix.em.EMFitterFluxOnly(
            miniter=self.flux_miniter,
            maxiter=self.flux_maxiter,
            tol=self.tol,
            vary_sky=self.vary_sky,
        )

        gm_guess = self._get_band_guess()

        return em.go(obs=emobs, guess=gm_guess, sky=sky)

    def _get_band_guess(self):
        """
        get a guess for the band based on the coadd mixture
        """
        rng = self.rng
        gmix_guess = self._result['coadd_gmix'].copy()

        gdata = gmix_guess.get_data()
        for idata in range(gdata.size):
            rnum = rng.uniform(low=-0.01, high=0.01)
            gdata['p'][idata] *= (1.0 + rnum)

        return gmix_guess

    def _do_psf_fits(self, mbobs, psf_ngauss):
        """
        perform psf fits, catching errors
        """

        try:
            do_psf_fit(mbobs, psf_ngauss, rng=self.rng)
            flags = 0
        except PSFFailure as err:
            logger.info(str(err))
            flags = procflags.PSF_FAILURE

        return {'flags': flags}
