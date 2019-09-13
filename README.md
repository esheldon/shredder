# Shredder
An astronomical image deblender

## Examples

An example using the built-in simulation
```python
import numpy as np
import shredder
import esutil as eu

rng = np.random.RandomState()
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
    rng=rng,
)

s = shredder.Shredder(mbobs, rng=rng)
s.shred(gm_guess)

# result contains the best fit mixture for each band
res = s.get_result()

# if you ahve installed the optional packags you can view
# the comparison of model and data
s.plot_comparison(show=True)
```

## Requirements

- python 3.  python 2 should also work, but this hasn't been tested
- numpy
- numba
- ngmix
- esutil
- pytest (optional for unit tests)
- galsim (optional for simulations and tests)
- biggles (optional for visualization)
- images (optional for visualization)
