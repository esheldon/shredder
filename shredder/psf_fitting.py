"""
TODO make sure we have a good center for the em fitting before
we run
"""

import logging
import ngmix
from ngmix.gexceptions import BootPSFFailure

logger = logging.getLogger(__name__)


def do_psf_fit(obs, ngauss, ntry=4, rng=None):
    """
    fit the obs to a psf model and set the mixture

    Parameters
    ----------
    obs: MultiBandObsList, ObsList, Observation
        Observations to fit
    ngauss: int
        Number of coelliptical gaussians to fit
    rng: np.random.RandomState
        random number generator
    """

    if isinstance(obs, ngmix.MultiBandObsList):
        for tobslist in obs:
            for tobs in tobslist:
                do_psf_fit(tobs, ngauss, rng=rng)

    elif isinstance(obs, ngmix.ObsList):
        for tobs in obs:
            do_psf_fit(tobs, ngauss, rng=rng)

    else:

        psf_obs = obs.psf

        scale = psf_obs.jacobian.scale
        lm_pars = {
            'xtol': 1.0e-5,
            'ftol': 1.0e-5,
            'maxfev': 2000,
        }

        if ngauss == 1:
            # guesser = ngmix.guessers.SimplePSFGuesser(
            #     rng=rng,
            #     guess_from_moms=True,
            # )
            # fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=lm_pars)
            fitter = ngmix.admom.AdmomFitter(rng=rng)
            guesser = ngmix.guessers.GMixPSFGuesser(
                rng=rng, ngauss=1, guess_from_moms=True,
            )
            runner = ngmix.runners.PSFRunner(
                fitter=fitter,
                guesser=guesser,
                ntry=ntry,
            )
            res = runner.go(psf_obs)
        else:
            do_offset = True

            emobs, sky = ngmix.em.prep_obs(psf_obs)

            guesser = ngmix.guessers.GMixPSFGuesser(
                rng=rng,
                ngauss=ngauss,
                guess_from_moms=True,
            )

            for i in range(ntry):
                guess = guesser(psf_obs)
                gdata = guess.get_data()

                # we assume it is centered; we make our guesses
                # centered, used fixed cen fitter, but add
                # slight offsets to avoid degeneracies
                # first we must put all at the same center

                gdata['row'] = 0.0
                gdata['col'] = 0.0

                # tweaks to offsets to avoid degeneracies
                if do_offset:
                    off = 0.001*scale
                    n = gdata.size
                    gdata['row'] += rng.uniform(low=-off, high=off, size=n)
                    gdata['col'] += rng.uniform(low=-off, high=off, size=n)

                fitter = ngmix.em.EMFitterFixCen(
                    miniter=20,
                    tol=1.0e-4,
                )

                res = fitter.go(obs=emobs, guess=guess, sky=sky)

                logger.debug(res)

                if res['flags'] == 0:
                    break
                else:
                    logger.info('retrying psf fit')

        if res['flags'] != 0:
            logger.info('psf fitting failed: %s' % str(res))
            raise BootPSFFailure('psf fitting failed: %s' % str(res))

        gmix = res.get_gmix()
        psf_obs.set_gmix(gmix)

        logger.debug('psf T: %g' % gmix.get_T())
        logger.debug('psf pars:')
        logger.debug('%s' % str(gmix))
