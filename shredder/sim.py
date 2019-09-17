import numpy as np
import ngmix
import copy

DEFAULT_CONFIG = {
    'image': {
        'dim_pixels': 100,
        'noise': 0.1,
        # 'pixel_scale': 0.263,
        'pixel_scale': 0.26227,  # to match example wcs in shredx
        'bad_columns': False,
    },
    'positions': {
        'width_pixels': 50,
    },
    'psf': {
        'model': 'moffat',
        'fwhm': 0.9,
        'beta': 3.5,
    },
    'objects': {
        'nobj': 5,
        'fracdev_range': [0.0, 1.0],

        'hlr_range': [0.01, 1.5],
        # 'flux_range': [0.1, 300.0],
        'flux_range': 'track_hlr',

        'bulge_sizefrac_range': [0.1, 0.5],
        'bulge_angle_offet_range': [-30, 30],  # degrees

        'disk_color': [1.0, 1.0, 1.0],
        # 'disk_color': [1.25, 1.0, 1.0/1.25],
        # 'disk_color': [1.25, 1.0, 0.75],
        'bulge_color': [0.35, 1.0, 1.0/0.35],
        # 'bulge_color': [0.5, 1.0, 1.5],

        'gsigma': 0.2,
    },
}


def get_default_config():
    """
    get a copy of the default config
    """
    return copy.deepcopy(DEFAULT_CONFIG)


class Sim(dict):
    """
    simple sim to generate objects scattered about an image
    """
    def __init__(self, rng=None, config=None):
        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng

        dconf = get_default_config()
        self.update(dconf)

        if config is not None:
            self.update(config)

        self._set_psf()

        self.gpdf = ngmix.priors.GPriorBA(
            self['objects']['gsigma'],
            rng=rng,
        )

    def __call__(self):
        """
        get a simulated ngmix.MultiBandObsList
        """
        iconf = self['image']

        band_images, band_weights, shifts = self._get_noisy_images()

        jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=iconf['pixel_scale'],
        )

        mbobs = ngmix.MultiBandObsList()

        for image, weight, shift in zip(band_images, band_weights, shifts):
            obs = ngmix.Observation(
                image,
                weight=weight,
                jacobian=jacobian,
                psf=self.get_psf_obs(),
                ignore_zero_weight=False,
            )
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        centers = self._get_centers(obs, shifts)
        mbobs.meta['centers'] = centers
        return mbobs

    def get_psf(self):
        """
        get the galsim psf objects
        """
        return self._psf

    def get_psf_obs(self):
        """
        get a copy of the psf observation
        """
        return self._psf_obs.copy()

    def _get_centers(self, obs, shifts):
        """
        get the true centers for each simulated object
        """
        dt = [
            ('row', 'f4'), ('col', 'f4'),
        ]
        centers = np.zeros(self['objects']['nobj'], dtype=dt)

        ccen = (np.array(obs.image.shape)-1)/2

        scale = self['image']['pixel_scale']
        for i, shift in enumerate(shifts):
            row = ccen[0] + shift[1]/scale
            col = ccen[1] + shift[0]/scale

            centers['row'][i] = row
            centers['col'][i] = col

        return centers

    def _get_noisy_images(self):
        """
        get noisy images for each band
        """
        band_images, shifts = self._get_images()

        band_weights = []
        for image in band_images:
            weight = self._get_weight_and_badpix(image)
            band_weights.append(weight)

            noise = self.rng.normal(
                size=image.shape,
                scale=self['image']['noise'],
            )
            image += noise

        return band_images, band_weights, shifts

    def _get_weight_and_badpix(self, image):
        """
        get a weight image.  Mark some of the pixels to
        have zero weight
        """

        iconf = self['image']

        dims = [iconf['dim_pixels']]*2
        weight = np.zeros(dims)
        weight[:, :] = 1.0/iconf['noise']**2

        if iconf['bad_columns']:
            badcol = self.rng.randint(0, dims[1])
            weight[:, badcol] = 0
            image[:, badcol] = 0.0
            # weight[:, badcol-1:badcol+2] = 0
            # image[:, badcol-1:badcol+2] = 0

        return weight

    def _get_images(self):
        """
        get images without noise for each band
        """
        import galsim

        gmodels = []
        rmodels = []
        imodels = []
        shifts = []
        for i in range(self['objects']['nobj']):
            band_models, shift = self._get_convolved_models()
            gmodels.append(band_models[0])
            rmodels.append(band_models[1])
            imodels.append(band_models[2])
            shifts.append(shift)

        band_models = [
            galsim.Add(gmodels),
            galsim.Add(rmodels),
            galsim.Add(imodels),
        ]

        iconf = self['image']
        ny, nx = [iconf['dim_pixels']]*2

        band_images = []
        for model in band_models:
            image = model.drawImage(
                nx=nx,
                ny=ny,
                scale=iconf['pixel_scale'],
            ).array

            band_images.append(image)

        return band_images, shifts

    def _get_convolved_models(self):
        """
        get models convolved by the psf for each band
        """
        import galsim
        band_models0, shift = self._get_models()

        band_models = []
        for model0 in band_models0:
            model = galsim.Convolve(
                model0,
                self.get_psf(),
            )
            band_models.append(model)

        return band_models, shift

    def _get_models(self):
        """
        get models for each band
        """
        import galsim
        rng = self.rng

        o = self['objects']
        shift = self._get_shift()

        fracdev = rng.uniform(
            low=o['fracdev_range'][0],
            high=o['fracdev_range'][1],
        )

        disk_hlr = rng.uniform(
            low=o['hlr_range'][0],
            high=o['hlr_range'][1],
        )

        bulge_sizefrac = rng.uniform(
            low=o['bulge_sizefrac_range'][0],
            high=o['bulge_sizefrac_range'][1],
        )
        bulge_hlr = disk_hlr*bulge_sizefrac

        if o['flux_range'] == 'track_hlr':
            flux = disk_hlr**2 * 100
        else:
            flux = rng.uniform(
                low=o['flux_range'][0],
                high=o['flux_range'][1],
            )

        g1disk, g2disk = self.gpdf.sample2d()

        bulge_angle_offset = rng.uniform(
                low=o['bulge_angle_offet_range'][0],
                high=o['bulge_angle_offet_range'][1],
        )
        bulge_angle_offset = np.deg2rad(bulge_angle_offset)
        disk_shape = ngmix.Shape(g1disk, g2disk)
        bulge_shape = disk_shape.get_rotated(bulge_angle_offset)

        disk = galsim.Exponential(
            half_light_radius=disk_hlr,
            flux=(1-fracdev)*flux,
        ).shear(
            g1=disk_shape.g1,
            g2=disk_shape.g2,
        ).shift(
            *shift,
        )

        bulge = galsim.DeVaucouleurs(
            half_light_radius=bulge_hlr,
            flux=fracdev*flux,
        ).shear(
            g1=bulge_shape.g1,
            g2=bulge_shape.g2,
        ).shift(
            *shift,
        )

        disk_color = o['disk_color']
        bulge_color = o['bulge_color']

        gdisk = disk*disk_color[0]
        rdisk = disk*disk_color[1]
        idisk = disk*disk_color[2]

        gbulge = bulge*bulge_color[0]
        rbulge = bulge*bulge_color[1]
        ibulge = bulge*bulge_color[2]

        gmodel = galsim.Add(gdisk, gbulge)
        rmodel = galsim.Add(rdisk, rbulge)
        imodel = galsim.Add(idisk, ibulge)

        return (gmodel, rmodel, imodel), shift

    def _get_shift(self):
        """
        get shift within the image, relative to the canonical
        center, in arcseconds
        """
        rng = self.rng

        iconf = self['image']
        pconf = self['positions']

        radius = pconf['width_pixels']/2 * iconf['pixel_scale']

        return rng.uniform(
            low=-radius,
            high=+radius,
            size=2,
        )

    def _set_psf(self):
        """
        set the psf and psf observation
        """
        import galsim

        pconf = self['psf']

        if pconf['model'] == 'moffat':
            self._psf = galsim.Moffat(
                fwhm=pconf['fwhm'],
                beta=pconf['beta'],
            )
        elif pconf['model'] == 'gauss':
            self._psf = galsim.Gaussian(
                fwhm=self['psf']['fwhm'],
            )
        else:
            raise ValueError('bad psf model: "%s"' % pconf['model'])

        psf_image = self._psf.drawImage(
            scale=self['image']['pixel_scale'],
        ).array

        psf_noise = 0.0001
        psf_image += self.rng.normal(
            size=psf_image.shape,
            scale=psf_noise,
        )
        psf_weight = psf_image*0 + 1.0/psf_noise**2

        cen = (np.array(psf_image.shape)-1)/2
        jac = ngmix.DiagonalJacobian(
            row=cen[0],
            col=cen[1],
            scale=self['image']['pixel_scale'],
        )

        self._psf_obs = ngmix.Observation(
            psf_image,
            psf_weight,
            jacobian=jac,
        )


def test(ntrial=1, seed=None, show=False, scale=2):
    """
    quick test viewing an example sim
    """
    from . import vis

    rng = np.random.RandomState(seed)
    sim = Sim(rng)

    for i in range(ntrial):
        mbobs = sim()

        if show:
            vis.view_mbobs(mbobs, scale=scale, show=show)
            if 'q' == input('hit a key (q to quit): '):
                break

    return mbobs
