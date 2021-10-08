from contextlib import contextmanager
import numpy as np
import ngmix


class ModelSubtractor(object):
    """
    Class to produce neighbor model subtracted images

    Parameters
    ----------
    shredder: Shredder
        The shredder used for deblending
    nobj: int
        Number of objects represented in the results.
    """
    def __init__(self, shredder, nobj):
        self.shredder = shredder
        self.nobj = nobj

        self._set_ngauss_per()
        self._build_models()
        self._build_subtracted_mbobs()

    @contextmanager
    def add_source(self, index):
        """
        Open a with context with all objects subtracted except the specified
        one.

        since the image had data-model this restores the pixels for the object
        of interest, minus models of other objects

        with subtractor.add_source(index):
            # do something with subtractor.mbobs

        Parameters
        ----------
        index: int
            The index of the source

        Yields
        -------
        mbobs, although more typically one uses the .mbobs attribute
        """
        imax = self.nobj - 1
        if index < 0 or index > imax:
            raise IndexError('index {index} out range [0, {imax}]')

        self._add_or_subtract_model_image(index, 'add')

        try:
            # usually won't use this yielded value
            yield self.mbobs
        finally:
            self._add_or_subtract_model_image(index, 'subtract')

    def _add_or_subtract_model_image(self, index, type):
        mbobs = self.mbobs
        model_images = self.model_images

        for obslist, band_model_images in zip(mbobs, model_images):
            obs = obslist[0]

            model_image = band_model_images[index]

            with obs.writeable():
                if type == 'add':
                    obs.image += model_image
                else:
                    obs.image -= model_image

    def _set_ngauss_per(self):
        res = self.shredder.result
        ngauss = len(res['band_gmix_convolved'][0])
        if ngauss % self.nobj != 0:
            raise ValueError('found ngauss % nobj != 0')

        self.ngauss = ngauss
        self.ngauss_per = ngauss // self.nobj

    def _build_models(self):
        res = self.shredder.result
        mbobs_orig = self.shredder.mbobs

        self.model_images = []

        for band, obslist in enumerate(mbobs_orig):
            obs = obslist[0]
            band_gm = res['band_gmix_convolved'][band].get_data()

            band_model_images = []

            coords = ngmix.pixels.make_coords(obs.image.shape, obs.jacobian)

            for iobj in range(self.nobj):
                start = self.ngauss_per * iobj
                end = self.ngauss_per * (iobj + 1)

                gm = band_gm[start:end]

                model_image = np.zeros_like(obs.image)
                ngmix.gmix.render_nb.render(
                    gm, coords, model_image.ravel(), fast_exp=1,
                )
                band_model_images.append(model_image)

            self.model_images.append(band_model_images)

    def _build_subtracted_mbobs(self):

        mbobs_orig = self.shredder.mbobs
        model_images = self.model_images
        self.mbobs = ngmix.MultiBandObsList()

        for obslist, band_model_images in zip(mbobs_orig, model_images):
            obs = obslist[0]

            diff_obs = obs.copy()

            with diff_obs.writeable():
                for model_image in band_model_images:
                    diff_obs.image -= model_image

            diff_obslist = ngmix.ObsList()
            diff_obslist.append(diff_obs)

            self.mbobs.append(diff_obslist)
