import numpy as np
import shredder
import pytest


@pytest.mark.parametrize('seed', [8, 314159])
def test_sim_smoke(seed):
    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)
    mbobs = sim()  # noqa


@pytest.mark.parametrize('seed', [75, 817, 213])
def test_sim(seed):
    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)
    mbobs = sim()

    assert mbobs.meta['centers'].size == sim['objects']['nobj']
