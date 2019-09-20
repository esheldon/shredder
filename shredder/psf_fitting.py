"""
TODO make sure we have a good center for the em fitting before
we run
"""

import logging
import ngmix
from .em import GMixEMFixCen

logger = logging.getLogger(__name__)


def do_psf_fit(obs, ngauss, rng=None):
    """
    fit the obs to a psf modle and set the mixture

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
        Tguess = 4.0*scale
        lm_pars = {
            'xtol': 1.0e-5,
            'ftol': 1.0e-5,
            'maxfev': 2000,
        }

        if ngauss == 1:
            runner = ngmix.bootstrap.PSFRunner(
                psf_obs,
                'gauss',
                Tguess,
                lm_pars,
                rng=rng,
            )
            runner.go(ntry=4)
            fitter = runner.fitter
        else:
            do_cen = False
            do_offset = True

            if do_cen:
                runner1 = ngmix.bootstrap.PSFRunner(
                    psf_obs,
                    'gauss',
                    Tguess,
                    lm_pars,
                    rng=rng,
                )
                runner1.go(ntry=4)
                gmix1 = runner1.fitter.get_gmix()
                row, col = gmix1.get_cen()

            orunner = ngmix.bootstrap.EMRunner(
                psf_obs,
                Tguess,
                ngauss,
                {},
                rng=rng,
            )
            guess = orunner.get_guess()
            gdata = guess.get_data()

            # first we must put all at the same center
            gdata['row'] = 0.0
            gdata['col'] = 0.0

            # now offset the centers all together
            if do_cen:
                print('setting cen:', row, col)
                guess.set_cen(row, col)

            if do_offset:
                off = 0.001*scale
                n = gdata.size
                gdata['row'] += rng.uniform(low=-off, high=off, size=n)
                gdata['col'] += rng.uniform(low=-off, high=off, size=n)

            imsky, sky = ngmix.em.prep_image(psf_obs.image)
            emobs = ngmix.Observation(
                imsky,
                weight=psf_obs.weight,
                jacobian=psf_obs.jacobian,
            )
            fitter = GMixEMFixCen(
                emobs,
                miniter=20,
                tol=1.0e-4,
            )

            fitter.go(guess, sky)

        res = fitter.get_result()
        logger.debug(res)

        if res['flags'] != 0:
            raise RuntimeError('psf fitting failed')

        gmix = fitter.get_gmix()
        psf_obs.set_gmix(gmix)

        logger.debug('psf T: %g' % gmix.get_T())
        logger.debug('psf pars:')
        logger.debug('%s' % str(gmix))
