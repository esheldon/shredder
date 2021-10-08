import numpy as np
import shredder
import esutil as eu
import pytest
import logging

logger = logging.getLogger(__name__)


def _add_T_and_scale(obj_data, scale):
    """
    for guesses we need T and scaled flux to surface brightness
    """

    add_dt = [('T', 'f4')]
    objs = eu.numpy_util.add_fields(obj_data, add_dt)

    # fluxes are in image units not surface brightness
    objs['flux'] *= scale**2

    # bad for non-gaussians, but what else do we have?
    min_sigma = scale
    min_T = 2*min_sigma**2
    T = 2*objs['hlr']**2
    T = T.clip(min=min_T)
    objs['T'] = T

    return objs


@pytest.mark.parametrize('seed', [91, 22])
def test_subtractor_smoke(seed, show=False):

    rng = np.random.RandomState(seed)

    sim = shredder.sim.Sim(rng=rng)
    mbobs = sim()
    model = 'dev'

    psf_ngauss = 2
    # miniter = 10
    # tol = 1.0e-4

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model=model,
        rng=rng,
    )

    s = shredder.Shredder(
        obs=mbobs,
        psf_ngauss=psf_ngauss,
        rng=rng,
    )
    s.shred(gm_guess)

    subtractor = shredder.ModelSubtractor(shredder=s, nobj=objs.size)

    if show:
        subtractor.plot_comparison(titles=['image', 'all subtracted'])

    for iobj in range(objs.size):
        with subtractor.add_source(iobj):
            if show:
                subtractor.plot_comparison(titles=['image', f'{iobj} added'])


@pytest.mark.parametrize('seed', [125, 871])
def test_subtractor(seed):
    """
    test that the fit is pretty good; this should be equivalent
    to the test_shredder test in test_shredding.py
    """
    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)
    mbobs = sim()

    guess_model = 'dev'
    psf_ngauss = 2

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model=guess_model,
        rng=rng,
    )

    s = shredder.Shredder(obs=mbobs, psf_ngauss=psf_ngauss, rng=rng)
    s.shred(gm_guess)

    res = s.get_result()
    logger.info('coadd: %s', res['coadd_result'])
    for band, band_result in enumerate(res['band_results']):
        logger.info('%s %s', band, band_result)

    assert res['flags'] == 0

    subtractor = shredder.ModelSubtractor(shredder=s, nobj=objs.size)

    chi2 = 0.0
    dof = 0
    for band, obslist in enumerate(subtractor.mbobs):
        obs = obslist[0]
        diffim = obs.image
        dof += diffim.size

        weight = obs.weight
        chi2 += (diffim**2 * weight).sum()

    dof = dof - 3
    chi2per = chi2/dof

    assert chi2per < 1.05


if __name__ == '__main__':
    shredder.setup_logging('info')

    show = True
    seed = 53
    rng = np.random.RandomState(seed)
    for i in range(100):
        # test_shredder_stars_moffat(rng.randint(0, 2**16))
        test_subtractor_smoke(rng.randint(0, 2**16), show=show)
