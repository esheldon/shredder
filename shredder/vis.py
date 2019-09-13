import os
import numpy as np


def view_mbobs(mbobs, scale, show=False):
    """
    make a color image of the MultiBandObsList
    """
    import images

    imlist = [olist[0].image for olist in mbobs]
    wtlist = [olist[0].weight for olist in mbobs]

    rgb = make_rgb(imlist, wtlist, scale=scale)
    plt = images.view(rgb, show=show)
    return plt


def compare_rgb_images(image, model, diffim,
                       seg, width, chi2,
                       rng=None,
                       title=None,
                       model_noisy=None):
    """
    make a comparison of the image with the model
    """
    import biggles
    import images

    dof = image.size-3
    chi2per = chi2/dof

    if model_noisy is not None:
        arat = 2.0/3.0
        tab = biggles.Table(2, 3, aspect_ratio=arat)
    else:
        arat = image.shape[1]/image.shape[0]
        tab = biggles.Table(2, 2, aspect_ratio=arat)

    tab[0, 0] = images.view(
        image,  # /maxval,
        show=False,
        title='image',
    )
    if model_noisy is not None:
        tab[0, 1] = images.view(
            model_noisy,  # /maxval,
            show=False,
            title='model+noise',
        )
        tab[0, 2] = images.view(
            model,  # /maxval,
            show=False,
            title='model',
        )
    else:
        tab[0, 1] = images.view(
            model,  # /maxval,
            show=False,
            title='model',
        )

    tab[1, 0] = images.view(
        diffim,
        show=False,
        title='chi2/dof: %.2f' % chi2per,
    )

    tab[1, 1] = plot_seg(seg, rng=rng, width=width, title='seg')

    if title is not None:
        tab.title = title

    fname = '/tmp/tmp-comp.png'
    tab.write_img(width, width*arat, fname)
    show_image(fname)


def show_image(fname):
    """
    show the image using an external viewer
    """
    os.system('feh --force-aliasing -B black %s &' % fname)


def make_rgb(imlist, wtlist, nonlinear=0.12, scale=0.0005):
    """
    make an rgb image using the input images and weights
    """
    import images

    # relative_scales = np.array([1.1, 1.0, 2.0])
    relative_scales = np.array([1.0, 1.0, 2.0])

    scales = scale*relative_scales

    noise_fac = 0.1
    rminval = noise_fac*np.sqrt(1/wtlist[2][0, 0])
    gminval = noise_fac*np.sqrt(1/wtlist[1][0, 0])
    bminval = noise_fac*np.sqrt(1/wtlist[0][0, 0])

    minval = min(rminval, gminval, bminval)
    minval = 0

    r = imlist[2].clip(min=minval)
    g = imlist[1].clip(min=minval)
    b = imlist[0].clip(min=minval)

    rgb = images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=nonlinear,
    )
    rgb.clip(max=1.0, out=rgb)
    return rgb


def plot_seg(segin, title=None, width=1000, rng=None, show=False):
    """
    plot the seg map with randomized colors
    """
    import images

    seg = np.transpose(segin)

    cseg = np.zeros((seg.shape[0], seg.shape[1], 3))

    if rng is None:
        rng = np.random.RandomState()

    useg = np.unique(seg)[1:]

    low = 50/255
    high = 255/255

    for i, segval in enumerate(useg):

        w = np.where(seg == segval)

        r, g, b = rng.uniform(low=low, high=high, size=3)

        cseg[w[0], w[1], 0] = r
        cseg[w[0], w[1], 1] = g
        cseg[w[0], w[1], 2] = b

    plt = images.view(cseg, show=False)

    if title is not None:
        plt.title = title

    if show:
        srat = seg.shape[1]/seg.shape[0]
        fname = '/tmp/seg.png'
        plt.write_img(width, width*srat, fname)
        show_image(fname)

    return plt
