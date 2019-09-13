import numpy as np
import ngmix

DEFAULT_CONFIG = {
    'image': {
        'dim_pixels': 50,
        'noise': 0.1,
        'pixel_scale': 0.263,
    },
    'positions': {
        'width_pixels': 35,
    },
    'psf': {
        'model': 'moffat',
        'fwhm': 0.9,
        'beta': 3.5,
    },
    'objects': {
        'hlr_range': [0.1, 1.5],
        'flux_range': [50, 150],
        'disk_color': [1.25, 1.0, 0.75],
        'bulge_color': [0.5, 1.0, 1.5],
        'gsigma': 0.2,
    },
}


class Sim(dict):
    """
    simple sim to generate objects scattered about an image
    """
    def __init__(self, rng, config=None):
        self.rng = rng

        if config is None:
            self.update(DEFAULT_CONFIG)
        else:
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

        band_images = self._get_noisy_images()
        band_weights = [
            image*0 + 1.0/iconf['noise']**2 for image in band_images
        ]

        jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=iconf['pixel_scale'],
        )

        mbobs = ngmix.MultiBandObsList()

        for image, weight in zip(band_images, band_weights):
            obs = ngmix.Observation(
                image,
                weight=weight,
                jacobian=jacobian,
                psf=self.get_psf_obs(),
            )
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

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

    def _get_noisy_images(self):
        """
        get noisy images for each band
        """
        band_images = self._get_images()

        for image in band_images:
            noise = self.rng.normal(
                size=image.shape,
                scale=self['image']['noise'],
            )
            image += noise

        return band_images

    def _get_images(self):
        """
        get images without noise for each band
        """
        band_models = self._get_convolved_models()

        iconf = self['image']
        ny, nx = iconf['dims']

        band_images = []
        for model in band_models:
            image = model.drawImage(
                nx=nx,
                ny=ny,
                scale=iconf['pixel_scale'],
            ).array

            band_images.append(image)

        return band_images

    def _get_convolved_models(self):
        """
        get models convolved by the psf for each band
        """
        import galsim
        band_models0 = self._get_models()

        band_models = []
        for model0 in band_models0:
            model = galsim.Convolve(
                model0,
                self.get_psf(),
            )
            band_models.append(model)

        return band_models

    def _get_models(self):
        """
        get models for each band
        """
        import galsim
        rng = self.rng

        o = self['object']
        shift = self._get_shift()

        fracdev = rng.uniform(low=0, high=1)
        bulge_sizefrac = rng.uniform(low=0.5, high=1)

        g1bulge, g2bulge = self.gpdf.sample2d()
        g1disk, g2disk = self.gpdf.sample2d()

        disk = galsim.Exponential(
            half_light_radius=o['hlr'],
            flux=(1-fracdev)*o['flux'],
        ).shear(
            g1=g1disk,
            g2=g2disk,
        ).shift(
            *shift,
        )

        bulge = galsim.DeVaucouleurs(
            half_light_radius=o['hlr']*bulge_sizefrac,
            flux=fracdev*o['flux'],
        ).shear(
            g1=g1bulge,
            g2=g2bulge,
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

        return gmodel, rmodel, imodel

    def _get_shift(self):
        """
        get shift within the image, relative to the canonical
        center, in arcseconds
        """
        rng = self.rng

        iconf = self['image']

        radius = iconf['width_pixels']/2 * iconf['pixel_scale']

        return rng.uniform(
            low=-radius,
            high=-radius,
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
