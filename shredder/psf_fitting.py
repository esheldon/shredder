import ngmix


def do_psf_fit(obs, rng=None):
    """
    fit the obs to a psf modle and set the mixture

    Parameters
    ----------
    obs: MultiBandObsList, ObsList, Observation
        Observations to fit
    """

    if isinstance(obs, ngmix.MultiBandObsList):
        for tobslist in obs:
            for tobs in tobslist:
                do_psf_fit(tobs, rng=rng)

    elif isinstance(obs, ngmix.ObsList):
        for tobs in obs:
            do_psf_fit(tobs, rng=rng)

    else:

        psf_obs = obs.psf

        Tguess = 4.0*psf_obs.jacobian.scale
        lm_pars = {
            'xtol': 1.0e-5,
            'ftol': 1.0e-5,
            'maxfev': 2000,
        }
        runner = ngmix.bootstrap.PSFRunner(
            psf_obs,
            'gauss',
            Tguess,
            lm_pars,
            rng=rng,
        )

        runner.go(ntry=4)
        fitter = runner.fitter

        res = fitter.get_result()
        if res['flags'] != 0:
            raise RuntimeError('psf fitting failed')

        gmix = fitter.get_gmix()
        psf_obs.set_gmix(gmix)
