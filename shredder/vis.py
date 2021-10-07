import os
import numpy as np
import tempfile
from matplotlib import pyplot as mplt


def view_mbobs(mbobs, scale=2, show=False, **kw):
    """
    make a color image of the MultiBandObsList
    """

    imlist = [olist[0].image for olist in mbobs]
    wtlist = [olist[0].weight for olist in mbobs]

    return view_rgb(imlist, wtlist, scale=scale, show=show, **kw)


def view_rgb(imlist, wtlist, scale=2, show=False, **kw):
    """
    view rgb data
    """
    from espy import images

    rgb = make_rgb(imlist, wtlist, scale=scale)
    plt = images.view(rgb, show=show, **kw)
    return plt


def compare_mbobs_and_models(mbobs,
                             models,
                             seg=None,
                             width=1000,
                             rng=None,
                             title=None,
                             scale=2,
                             show=False):
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

    rgb = make_rgb(
        imlist,
        wtlist,
        scale=scale,
    )
    model_rgb = make_rgb(
        models,
        wtlist,
        scale=scale,
    )
    diff_rgb = make_rgb(
        difflist,
        wtlist,
        scale=scale,
    )

    return compare_rgb_images(
        rgb,
        model_rgb,
        diff_rgb,
        seg=seg,
        weight=wtall,
        width=width,
        chi2per=chi2per,
        rng=rng,
        title=title,
        show=show,
    )


def compare_rgb_images(image,
                       model,
                       diffim,
                       seg=None,
                       weight=None,
                       width=1000,
                       chi2per=None,
                       rng=None,
                       title=None,
                       show=False):
    """
    make a comparison of the image with the model
    """

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

    axs[imrow, imcol].imshow(image)
    axs[imrow, imcol].set_title('image')

    axs[modrow, modcol].imshow(model)
    axs[modrow, modcol].set_title('model')

    axs[diffrow, diffcol].imshow(diffim)
    axs[diffrow, diffcol].set_title(diff_title)

    if seg is not None:
        plot_seg(seg, ax=axs[segrow, segcol], rng=rng)
        axs[segrow, segcol].set_title('seg')

    if weight is not None:
        axs[wtrow, wtcol].imshow(weight)
        axs[wtrow, wtcol].set_title('weight')

    if title is not None:
        fig.suptitle(title)

    if show:
        fname = tempfile.mktemp(suffix='.png')
        fig.savefig(fname, dpi=150)
        show_image(fname)

    return fig


def show_image(fname):
    """
    show the image using an external viewer
    """
    os.system('feh --force-aliasing -B black %s &' % fname)


def make_rgb(imlist, wtlist, nonlinear=0.12, scale=0.0005):
    """
    make an rgb image using the input images and weights
    """
    from espy import images

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


def plot_seg(segin, ax, rng=None):
    """
    plot the seg map with randomized colors
    """

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

    ax.imshow(cseg)
