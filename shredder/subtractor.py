from contextlib import contextmanager
import numpy as np
import ngmix
from . import vis


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

    def plot_object(self, index, stamp_size):
        """
        plot a postage stamp for the indicated object

        Parameters
        ----------
        index: int
            The index of the source
        stamp_size: int
            The stamp size; note the actual returned stamp size is always odd
            so the object center is in the center pixel
        """
        mbobs = self.get_object_mbobs(index=index, stamp_size=stamp_size)
        row, col = mbobs[0][0].jacobian.get_cen()
        objs = np.zeros(1, dtype=[('row', 'f4'), ('col', 'f4')])
        objs['row'] = row
        objs['col'] = col
        return vis.view_mbobs(
            mbobs,
            title=f'object {index+1}',
            objs=objs,
        )

    def get_object_mbobs(self, index, stamp_size):
        """
        get a postage stamp MultiBandObsList for the indicated object

        Parameters
        ----------
        index: int
            The index of the source
        stamp_size: int
            The stamp size; note the actual returned stamp size is always odd
            so the object center is in the center pixel
        """

        # all bands share the same center
        gm = self.get_object_gmix(index, band=0)
        v_orig, u_orig = gm.get_cen()

        obs0_orig = self.shredder.mbobs[0][0]
        jacobian = obs0_orig.jacobian.copy()
        row_orig, col_orig = jacobian.get_rowcol(v=v_orig, u=u_orig)

        # this may trim off one at the beginning or end in some cases
        # seems unavoidable
        row_start, row_end, col_start, col_end = _get_bbox(
            image_shape=obs0_orig.image.shape,
            row=row_orig, col=col_orig,
            stamp_size=stamp_size,
        )

        jacobian.set_cen(
            row=row_orig-row_start,
            col=col_orig-col_start,
        )
        stamp_mbobs = ngmix.MultiBandObsList()

        for band, obslist in enumerate(self.mbobs):
            obs_orig = self.mbobs[band][0]
            stampim = obs_orig.image[row_start:row_end, col_start:col_end]
            stampwt = obs_orig.weight[row_start:row_end, col_start:col_end]
            # stampbmask = obs_orig.bmask[row_start:row_end, col_start:col_end]

            stampobs = ngmix.Observation(
                image=stampim,
                weight=stampwt,
                # bmask=stampbmask,
                jacobian=jacobian,
                psf=obs_orig.psf,
            )
            stamp_obslist = ngmix.ObsList()
            stamp_obslist.append(stampobs)
            stamp_mbobs.append(stamp_obslist)

        return stamp_mbobs

    def get_object_gmix(self, index, band):
        """
        Get a copy of the pre-psf gmix for the indicated object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        ngmix.GMix
        """
        gmdata = self.get_object_gmix_data(index, band)
        gm = ngmix.GMix(ngauss=self.ngauss_per)
        gm._data[:] = gmdata[:]

        return gm

    def get_object_gmix_convolved(self, index, band):
        """
        Get a copy of the psf convolved gmix for the indicated object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        ngmix.GMix
        """
        gmdata = self.get_object_gmix_data_convolved(index, band)
        gm = ngmix.GMix(ngauss=self.ngauss_per_convolved)
        gm._data[:] = gmdata[:]

        return gm

    def get_object_gmix_data(self, index, band):
        """
        Get a copy of the pre-psf gmix data for the indicated object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        ngmix.GMix
        """
        start, end = self.get_object_index_range(index)

        res = self.shredder.result

        band_gm = res['band_gmix'][band].get_data()
        return band_gm[start:end].copy()

    def get_object_gmix_data_convolved(self, index, band):
        """
        Get a copy of the psf convolved gmix data for the indicated object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        ngmix.GMix
        """
        start, end = self.get_object_index_range_convolved(index)

        res = self.shredder.result

        band_gm = res['band_gmix_convolved'][band].get_data()
        return band_gm[start:end].copy()

    def get_object_index_range(self, index):
        """
        Get the gaussian index range for the requested pre-psf object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        start, end to be used as a slice range
        """
        if index < 0 or index > self.nobj-1:
            raise IndexError(f'no such object index {index}')

        start = self.ngauss_per * index
        end = self.ngauss_per * (index + 1)
        return start, end

    def get_object_index_range_convolved(self, index):
        """
        Get the gaussian index range for the requested convolved object

        Paramters
        ---------
        index: int
            the object index

        Returns
        -------
        start, end to be used as a slice range
        """
        if index < 0 or index > self.nobj-1:
            raise IndexError(f'no such object index {index}')

        start = self.ngauss_per_convolved * index
        end = self.ngauss_per_convolved * (index + 1)
        return start, end

    def plot_comparison(self, titles=None, show=True, **kw):
        """
        visualize a comparison of the model and data
        """
        if titles is None:
            titles = ('image', 'subtracted')

        subimages = [obslist[0].image for obslist in self.mbobs]
        objs = self.get_positions()
        return vis.compare_mbobs_and_models(
            self.shredder.mbobs,
            subimages,
            titles=titles,
            objs=objs,
            show=show,
            **kw
        )

    def get_positions(self):
        """
        get array with row/col for each object
        """
        dt = [('row', 'f4'), ('col', 'f4')]
        objs = np.zeros(self.nobj, dtype=dt)

        obs0_orig = self.shredder.mbobs[0][0]
        for index in range(self.nobj):
            gm = self.get_object_gmix(index, band=0)
            v_orig, u_orig = gm.get_cen()

            jacobian = obs0_orig.jacobian.copy()
            row_orig, col_orig = jacobian.get_rowcol(v=v_orig, u=u_orig)
            objs['row'][index] = row_orig
            objs['col'][index] = col_orig

        return objs

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

        ngauss = len(res['band_gmix'][0])
        ngauss_convolved = len(res['band_gmix_convolved'][0])

        if ngauss_convolved % self.nobj != 0:
            raise ValueError('found ngauss % nobj != 0')

        self.ngauss_convolved = ngauss_convolved
        self.ngauss_per_convolved = ngauss_convolved // self.nobj

        self.ngauss = ngauss
        self.ngauss_per = ngauss // self.nobj

    def _build_models(self):
        mbobs_orig = self.shredder.mbobs

        self.model_images = []

        for band, obslist in enumerate(mbobs_orig):
            obs = obslist[0]

            band_model_images = []

            coords = ngmix.pixels.make_coords(obs.image.shape, obs.jacobian)

            for iobj in range(self.nobj):
                gm = self.get_object_gmix_data_convolved(iobj, band)

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


def _get_bbox(image_shape, row, col, stamp_size):

    rad = int(stamp_size) // 2

    row_start, row_end = _get_start_end(
        image_dim=image_shape[0],
        cen=row, rad=rad,
    )
    col_start, col_end = _get_start_end(
        image_dim=image_shape[1],
        cen=col, rad=rad,
    )

    row_start, row_end, col_start, col_end = _trim_one_maybe(
        image_shape,
        row_start, row_end, col_start, col_end,
    )

    _check_start_end(
        start=row_start, end=row_end, image_dim=image_shape[0], type='row',
    )
    _check_start_end(
        start=col_start, end=col_end, image_dim=image_shape[1], type='col',
    )
    assert row_start - row_end == col_start - col_end, 'non round found'

    return row_start, row_end, col_start, col_end


def _trim_one_maybe(image_shape, row_start, row_end, col_start, col_end):
    if row_start < 0:
        if row_start == -1:
            # maintain square image
            row_start += 1
            col_start += 1

    if col_start < 0:
        if col_start == -1:
            # maintain square image
            col_start += 1
            row_start += 1

    if row_end > image_shape[0]:
        if row_end == image_shape[0] + 1:
            # maintain square image
            row_end -= 1
            col_end -= 1

    if col_end > image_shape[1]:
        if col_end == image_shape[1] + 1:
            # maintain square image
            col_end -= 1
            row_end -= 1

    return row_start, row_end, col_start, col_end


def _check_start_end(start, end, image_dim, type):
    if start < 0 or start > image_dim:
        raise IndexError(
            f'requested bbox {type} range [{start}:{end}) is '
            f'out of bounds [0:{image_dim})'
        )

    return start, end


def _get_start_end(image_dim, cen, rad):
    # this is how the stack code is currently working
    icen = int(round(cen))
    start = icen - rad
    end = icen + rad + 1
    return start, end
