import numpy as np
import shredder
import esutil as eu
import pytest


@pytest.mark.parametrize('seed', [55, 77])
def test_shredder_smoke(seed, show=False):
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

    gm_guess = shredder.get_guess_from_cat(
        objs,
        pixel_scale=scale,
        model='dev',
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng)
    s.shred(gm_guess)

    if show:
        s.plot_comparison(show=True)


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

    gm_guess = shredder.get_guess_from_cat(
        objs,
        pixel_scale=scale,
        model='dev',
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng)
    s.shred(gm_guess)

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
