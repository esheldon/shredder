# Shredder
An astronomical image deblender

![meme](https://i.imgflip.com/5qa8b2.jpg)

## Examples

An example using the built-in simulation.  This example requires galsim to be
installed.

```python
import numpy as np
import shredder
import esutil as eu

rng = np.random.RandomState()
sim = shredder.sim.Sim(rng=rng)
mbobs = sim()

scale = sim['image']['pixel_scale']

# make a guess based on the true centers and
# arbitrary sizes

centers = mbobs.meta['centers']
add_dt = [('T', 'f8')]
objs = eu.numpy_util.add_fields(centers, add_dt)

# arbitrary sizes
objs['T'] = rng.uniform(
    low=2.0*scale**2,
    high=4.0*scale**2,
    size=objs.size,
)

# generate guesses from the catalog
gm_guess = shredder.get_guess_from_cat(
    objs,
    pixel_scale=scale,
    rng=rng,
)

# run the deblender
s = shredder.Shredder(mbobs, rng=rng)
s.shred(gm_guess)

# result contains the best fit mixture for each band and information
# about the processing
res = s.get_result()
assert res['flags'] == 0

# if you have installed the optional packags you can view
# the comparison of model and data
s.plot_comparison(show=True)
```

## Deblend Example On Real Data

Example deblending an image from the Dark Energy Survey.  There
are 48 objects included in the blend.  The total time to run
was 35 seconds.

![Example From DES](https://raw.githubusercontent.com/esheldon/shredder/master/data/example-blend01.png)

## Requirements

- python 3.  python 2 should also work, but this hasn't been tested
- numpy
- numba
- ngmix
- esutil
- pytest (optional for unit tests)
- galsim (optional for simulations and tests)
- images (optional for visualization)
