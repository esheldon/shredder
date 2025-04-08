import numpy as np

DEFAULT_STRETCH = 1.25
DEFAULT_Q = 7.5
SIZE = 16
COLOR = 'red'
EDGECOLOR = 'white'


def view_mbobs(
    mbobs,
    title=None,
    stretch=DEFAULT_STRETCH,
    q=DEFAULT_Q,
    show=True,
    objs=None,
):
    """
    make a color image of the MultiBandObsList
    """

    imlist = [olist[0].image for olist in mbobs]

    return view_image(
        imlist, stretch=stretch, q=q, show=show, title=title, objs=objs,
    )


def view_image(
    imlist, ax=None, stretch=DEFAULT_STRETCH, q=DEFAULT_Q,
    title=None,
    objs=None,
    show=True,
):
    """
    view rgb data
    """
    from matplotlib import pyplot as mplt

    from astropy.visualization import (
        AsinhStretch,
        imshow_norm,
    )

    if ax is None:
        fig, ax = mplt.subplots()

    if len(imlist) >= 3:
        image = make_rgb(imlist, stretch=stretch, q=q)
        ax.imshow(image)
    elif len(imlist) == 1:
        imshow_norm(imlist[0], ax=ax, stretch=AsinhStretch())
    else:
        raise ValueError('can only do 3d or 1d')

    if objs is not None:
        ax.scatter(
            objs['col'], objs['row'],
            s=SIZE, color=COLOR, edgecolor=EDGECOLOR,
        )

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    if show:
        mplt.show()
        mplt.close()
    else:
        return fig


def compare_mbobs_and_models(
    mbobs,
    models,
    seg=None,
    rng=None,
    title=None,
    titles=('image', 'model'),
    # scale=0.5,
    stretch=DEFAULT_STRETCH,
    q=DEFAULT_Q,
    objs=None,
    show=True,
):
    """
    generate rgb images for data and model and make
    a comparison plot
    """
    imlist = [olist[0].image for olist in mbobs]
    wtlist = [olist[0].weight for olist in mbobs]

    difflist = []
    chi2 = 0.0

    npix = 0
    wtall = None
    for im, wt, model in zip(imlist, wtlist, models):

        if wtall is None:
            wtall = wt.copy()
        else:
            wtall *= wt

        diffim = im - model

        difflist.append(diffim)
        chi2 += (diffim**2 * wt).sum()

        wbad = np.where(wt <= 0.0)
        npix += im.size - wbad[0].size

    dof = npix-3
    chi2per = chi2/dof

    if len(imlist) >= 3:
        image = make_rgb(
            imlist,
            stretch=stretch,
            q=q,
        )
        model_image = make_rgb(
            models,
            stretch=stretch,
            q=q,
        )
        diff_image = make_rgb(
            difflist,
            stretch=stretch,
            q=q,
        )
    else:
        image = imlist[0]
        model_image = models[0]
        diff_image = difflist[0]

    return compare_images(
        image,
        model_image,
        diff_image,
        seg=seg,
        weight=wtall,
        chi2per=chi2per,
        rng=rng,
        title=title,
        titles=titles,
        objs=objs,
        show=show,
    )


def compare_images(
    image,
    model,
    diffim,
    titles=('image', 'model'),
    seg=None,
    weight=None,
    chi2per=None,
    rng=None,
    title=None,
    objs=None,
    show=True,
):
    """
    make a comparison of the image with the model
    """
    from matplotlib import pyplot as mplt

    if chi2per is not None:
        diff_title = 'chi2/dof: %.2f' % chi2per
    else:
        diff_title = None

    imrow, imcol = 0, 0
    modrow, modcol = 0, 1

    nrows = 2
    if seg is not None and weight is not None:
        ncols = 3
        # arat = image.shape[1]/image.shape[0] * 2/3

        diffrow, diffcol = 0, 2
        segrow, segcol = 1, 0
        wtrow, wtcol = 1, 1
    else:
        ncols = 2
        # arat = image.shape[1]/image.shape[0]

        diffrow, diffcol = 1, 0

        if seg is not None:
            segrow, segcol = 1, 1
        elif weight is not None:
            wtrow, wtcol = 1, 1

    fig, axs = mplt.subplots(nrows=nrows, ncols=ncols)

    imax = axs[imrow, imcol]
    modax = axs[modrow, modcol]
    diffax = axs[diffrow, diffcol]

    imax.imshow(image)
    imax.set_title(titles[0])

    modax.imshow(model)
    modax.set_title(titles[1])

    diffax.imshow(diffim)
    diffax.set_title(diff_title)

    if objs is not None:
        for tax in [imax, modax, diffax]:
            tax.scatter(
                # objs['x'], objs['y'],
                objs['col'], objs['row'],
                s=SIZE, color=COLOR, edgecolor=EDGECOLOR,
            )

    if seg is not None:
        plot_seg(seg, ax=axs[segrow, segcol], rng=rng)
        axs[segrow, segcol].set_title('seg')

    if weight is not None:
        axs[wtrow, wtcol].imshow(weight)
        axs[wtrow, wtcol].set_title('weight')

    if seg is not None and weight is not None:
        axs[-1, -1].axis('off')

    if title is not None:
        fig.suptitle(title)

    if show:
        mplt.show()
        mplt.close(fig)
    else:
        return fig


def make_rgb(imlist, stretch=DEFAULT_STRETCH, q=DEFAULT_Q, minval=0):
    """
    make an rgb image using the input images and weights
    """
    from astropy.visualization.lupton_rgb import AsinhMapping
    # from astropy.visualization import (
    #     AsinhStretch,
    #     imshow_norm,
    # )
    asinh = AsinhMapping(
        minimum=0,
        stretch=stretch,
        Q=q,
    )

    ny, nx = imlist[0].shape

    rgb = np.zeros((3, ny, nx), dtype='f4')
    rgb[0, :, :] = imlist[2].clip(min=minval)
    rgb[1, :, :] = imlist[1].clip(min=minval)
    rgb[2, :, :] = imlist[0].clip(min=minval)

    rgb = asinh.make_rgb_image(*rgb)

    return rgb


'''
def view_mbobs_old(mbobs, scale=0.5, show=True, **kw):
    """
    make a color image of the MultiBandObsList
    """

    imlist = [olist[0].image for olist in mbobs]

    return view_rgb(imlist, scale=scale, show=show, **kw)


def view_rgb_old(imlist, scale=2, show=True, **kw):
    """
    view rgb data
    """
    from espy import images

    rgb = make_rgb(imlist, scale=scale)
    plt = images.view(rgb, show=show, **kw)
    return plt


def make_rgb_old(imlist, nonlinear=0.12, scale=0.5):
    """
    make an rgb image using the input images and weights
    """
    from espy import images

    # relative_scales = np.array([1.1, 1.0, 2.0])
    relative_scales = np.array([1.0, 1.0, 2.0])

    scales = scale*relative_scales

    # minval = min(rminval, gminval, bminval)
    minval = 0

    r = imlist[2].clip(min=minval)
    g = imlist[1].clip(min=minval)
    b = imlist[0].clip(min=minval)

    rgb = images.get_color_image(
        r, g, b, scales=scales,
        nonlinear=nonlinear,
    )
    rgb.clip(max=1.0, out=rgb)
    return rgb
'''


def plot_seg(seg, ax, rng=None):
    """
    plot the seg map with randomized colors
    """

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

    ax.imshow(cseg)
