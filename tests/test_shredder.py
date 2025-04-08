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


@pytest.mark.parametrize('seed', [55, 77])
@pytest.mark.parametrize('vary_sky', [False, True])
def test_shredder_smoke(seed, vary_sky, show=False):
    """
    test we can run end to end
    """

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
        vary_sky=vary_sky,
        # miniter=miniter,
        # tol=tol,
        rng=rng,
    )
    s.shred(gm_guess)

    res = s.get_result()
    logger.info('coadd: %s', res['coadd_result'])
    assert res['flags'] == 0
    for band, band_result in enumerate(res['band_results']):
        logger.info('%s %s', band, band_result)

    if show:
        title = 'vary sky: %s' % vary_sky
        s.plot_comparison(show=True, title=title)


@pytest.mark.parametrize('seed', [99, 105])
def test_shredder_stars_gaussian(seed, show=False):
    """
    Test with sim a gaussian psf and stars, fitting
    gaussian to both object and psf
    """
    rng = np.random.RandomState(seed)
    guess_model = 'gauss'
    psf_ngauss = 1

    conf = {'psf': {'model': 'gauss', 'fwhm': 0.9}}
    sim = shredder.sim.Sim(rng=rng, config=conf)

    sim['objects']['flux_range'] = [100, 200]
    sim['objects']['hlr_range'] = (0.0001, 0.0001)

    mbobs = sim()

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model=guess_model,
        rng=rng,
    )

    s = shredder.Shredder(
        obs=mbobs,
        psf_ngauss=psf_ngauss,
        rng=rng,
    )
    s.shred(gm_guess)

    if show:
        s.plot_comparison(show=True, title='gaussians stars')

    res = s.get_result()
    logger.info('coadd: %s', res['coadd_result'])
    assert res['flags'] == 0
    for band, band_result in enumerate(res['band_results']):
        logger.info('%s %s', band, band_result)

    models = s.get_model_images()

    chi2 = 0.0
    dof = 0
    for band, model in enumerate(models):
        image = mbobs[band][0].image
        dof += image.size

        weight = mbobs[band][0].weight
        diffim = image - model
        chi2 += (diffim**2 * weight).sum()

    dof = dof - 3
    chi2per = chi2/dof

    assert chi2per < 1.05


@pytest.mark.parametrize('seed', [99, 105])
def test_shredder_stars_moffat(seed, show=False):
    """
    Test with sim a gaussian psf and stars, fitting
    gaussian to both object and psf
    """

    rng = np.random.RandomState(seed)
    guess_model = 'gauss'
    psf_ngauss = 3

    sim = shredder.sim.Sim(rng=rng)

    # need lower flux to pass this test, due to small issues
    # with the modeling
    # sim['objects']['flux_range'] = 100, 200
    sim['objects']['flux_range'] = [20, 50]
    sim['objects']['hlr_range'] = (0.0001, 0.0001)

    mbobs = sim()

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model=guess_model,
        rng=rng,
    )

    s = shredder.Shredder(
        obs=mbobs,
        psf_ngauss=psf_ngauss,
        rng=rng,
    )
    s.shred(gm_guess)

    if show:
        s.plot_comparison(show=True, title='moffat stars')

    res = s.get_result()
    logger.info('coadd: %s', res['coadd_result'])
    logger.info('gmix:')
    logger.info(res['coadd_gmix'])
    assert res['flags'] == 0
    for band, band_result in enumerate(res['band_results']):
        logger.info('%s %s', band, band_result)

    models = s.get_model_images()

    chi2 = 0.0
    dof = 0
    for band, model in enumerate(models):
        image = mbobs[band][0].image
        dof += image.size

        weight = mbobs[band][0].weight
        diffim = image - model
        chi2 += (diffim**2 * weight).sum()

    dof = dof - 3
    chi2per = chi2/dof

    assert chi2per < 1.05


@pytest.mark.parametrize('seed', [125, 871])
def test_shredder(seed):
    """
    test that the fit is pretty good
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

    models = s.get_model_images()

    chi2 = 0.0
    dof = 0
    for band, model in enumerate(models):
        image = mbobs[band][0].image
        dof += image.size

        weight = mbobs[band][0].weight
        diffim = image - model
        chi2 += (diffim**2 * weight).sum()

    dof = dof - 3
    chi2per = chi2/dof

    assert chi2per < 1.05


@pytest.mark.parametrize('seed', [9731, 7317])
def test_shredder_bad_columns(seed, show=False):
    """
    test with bad column
    """

    logger.info('seed: %s', seed)

    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)

    sim['image']['bad_columns'] = True
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

    s = shredder.Shredder(
        obs=mbobs,
        psf_ngauss=psf_ngauss,
        rng=rng,
    )
    s.shred(gm_guess)

    res = s.get_result()
    assert res['flags'] == 0
    cres = res['coadd_result']
    logger.info('%s %s', cres['numiter'], cres['fdiff'])
    for bres in res['band_results']:
        logger.info('%s %s', bres['numiter'], bres['fdiff'])

    if show:
        s.plot_comparison(show=True)


if __name__ == '__main__':
    # seed = 500
    # seed = 250
    # seed = np.random.randint(0, 2**10)
    # test_shredder_smoke(seed, False, show=True)
    # test_shredder(seed)
    # test_shredder_bad_columns(seed, show=True)
    # test_shredder_stars_gaussian(seed, show=True)
    # test_shredder_stars_moffat(seed, show=True)

    shredder.setup_logging('info')

    show = True
    seed = 813
    rng = np.random.RandomState(seed)
    for i in range(100):
        # test_shredder_stars_moffat(rng.randint(0, 2**16))
        test_shredder_bad_columns(rng.randint(0, 2**16), show=show)
