import numpy as np
import shredder
import esutil as eu
import pytest


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

    centers = mbobs.meta['centers']
    add_dt = [('T', 'f8')]
    objs = eu.numpy_util.add_fields(centers, add_dt)

    # fake size guesses
    objs['T'] = rng.uniform(
        low=2.0*scale**2,
        high=4.0*scale**2,
        size=objs.size,
    )

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

    centers = mbobs.meta['centers']
    add_dt = [('T', 'f8')]
    objs = eu.numpy_util.add_fields(centers, add_dt)

    # fake size guesses
    objs['T'] = rng.uniform(
        low=2.0*scale**2,
        high=4.0*scale**2,
        size=objs.size,
    )

    gm_guess = shredder.get_guess(
        objs,
        jacobian=mbobs[0][0].jacobian,
        model='dev',
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng, miniter=50)  # tol=1.0e-5)
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

    centers = mbobs.meta['centers']
    add_dt = [('T', 'f8')]
    objs = eu.numpy_util.add_fields(centers, add_dt)

    # fake size guesses
    objs['T'] = rng.uniform(
        low=2.0*scale**2,
        high=4.0*scale**2,
        size=objs.size,
    )

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
    # seed = 15575
    # seed = 278
    # seed = np.random.randint(0, 2**10)
    # test_shredder_bad_columns(seed, show=True)
    seed = 125
    test_shredder_smoke(seed, False, show=True)
