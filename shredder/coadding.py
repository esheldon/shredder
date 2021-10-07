import numpy as np
import ngmix


def make_coadd_obs(mbobs):
    """
    perform a simple coadd assuming the images all already align

    This is used for combining the coadds from multiple bands

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        An ngmix multi band observation list to be coadded.  The
        images must align perfectly and have the same wcs
    """

    weights = np.array([obslist[0].weight.max() for obslist in mbobs])
    wsum = weights.sum()

    nweights = weights/wsum

    coadd_image = mbobs[0][0].image.copy()
    coadd_image[:, :] = 0.0

    coadd_weight = coadd_image.copy()
    coadd_weight[:, :] = wsum

    coadd_psf = mbobs[0][0].psf.image.copy()
    coadd_psf[:, :] = 0
    coadd_psf_weight = mbobs[0][0].psf.weight.copy()*3  # not quite right

    for i, obslist in enumerate(mbobs):
        obs = obslist[0]

        wbad = np.where(obs.weight <= 0.0)
        if wbad[0].size > 0:
            coadd_weight[wbad] = 0.0

        coadd_image += obs.image*nweights[i]
        coadd_psf += obs.psf.image*nweights[i]

    psf_obs = ngmix.Observation(
        coadd_psf,
        weight=coadd_psf_weight,
        jacobian=mbobs[0][0].psf.jacobian,
    )

    # we don't ingore zero weight pixels because we will
    # fill them in during processing
    obs = ngmix.Observation(
        coadd_image,
        weight=coadd_weight,
        jacobian=mbobs[0][0].jacobian,
        psf=psf_obs,
        ignore_zero_weight=False,
    )

    # import images
    # images.view(coadd_weight, title='weight')
    return obs
