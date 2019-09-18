import numpy as np
import shredder
import esutil as eu
import pytest


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
    """
    objs['T'] = rng.uniform(
        low=2.0*scale**2,
        high=4.0*scale**2,
        size=objs.size,
    )
    """

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

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model='dev',
        rng=rng,
    )

    s = shredder.Shredder(
        mbobs,
        vary_sky=vary_sky,
        rng=rng,
    )
    s.shred(gm_guess)

    res = s.get_result()
    print(res['coadd_result'])
    assert res['flags'] == 0

    if show:
        title = 'vary sky: %s' % vary_sky
        s.plot_comparison(show=True, title=title)


@pytest.mark.parametrize('seed', [125, 871])
def test_shredder(seed):
    """
    test that the fit is pretty good
    """
    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)
    mbobs = sim()

    scale = sim['image']['pixel_scale']

    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model='dev',
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng)
    s.shred(gm_guess)

    res = s.get_result()
    print(res['coadd_result'])
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

    print('seed:', seed)

    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)

    sim['image']['bad_columns'] = True
    mbobs = sim()

    scale = sim['image']['pixel_scale']
    obj_data = mbobs.meta['obj_data']
    objs = _add_T_and_scale(obj_data, scale)

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model='dev',
        rng=rng,
    )

    for fill_zero_weight in [False, True]:
        s = shredder.Shredder(
            mbobs, rng=rng, fill_zero_weight=fill_zero_weight,
        )
        s.shred(gm_guess)

        res = s.get_result()
        assert res['flags'] == 0
        cres = res['coadd_result']
        print(cres['numiter'], cres['fdiff'])
        for bres in res['band_results']:
            print(bres['numiter'], bres['fdiff'])

        if show:
            title = 'fill: %s' % fill_zero_weight
            s.plot_comparison(show=True, title=title)


if __name__ == '__main__':
    seed = 278
    # seed = np.random.randint(0, 2**10)
    test_shredder_smoke(seed, False, show=True)
    test_shredder_bad_columns(seed, show=True)
