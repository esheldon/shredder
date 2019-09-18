import logging
import numpy as np
import ngmix

logger = logging.getLogger(__name__)


def get_guess(objs,
              jacobian=None,
              pixel_scale=1.0,
              model='dev',
              rng=None):
    """
    get a full gaussian mixture guess based on an input object list

    Parameters
    -----------
    objs: array
        Should have either
            - row, col in pixels, T in arcsec^2 and flux in surface brightness
              TODO may want to make row, col in arcsec
            - x, y, x2, y2 all in pixel units and flux in flux units
    pixel_scale: float
        The pixel scale, default 1
    model: string, optional
        model for distribution gaussian sizes around each object
        center.  'exp', 'dev', 'bdf', 'bd'
    rng: np.random.RandomState, optional
        optional random number generator
    """

    if rng is None:
        ur = np.random.uniform
    else:
        ur = rng.uniform

    if jacobian is not None:
        scale = jacobian.scale

    guess_pars = []
    for i in range(objs.size):
        if 'T' in objs.dtype.names:
            Tguess = objs['T'][i]  # *pixel_scale**2
            row = objs['row'][i]
            col = objs['col'][i]
            flux = objs['flux'][i] # *pixel_scale**2
        else:
            x2 = objs['x2'][i]
            y2 = objs['y2'][i]
            Tguess = (x2 + y2)*pixel_scale**2

            row = objs['y'][i]
            col = objs['x'][i]

            flux = objs['flux'][i]*pixel_scale**2

        if jacobian is not None:
            row, col = jacobian.get_vu(row, col)
        else:
            row = row*pixel_scale
            col = col*pixel_scale

        g1, g2 = ur(low=-0.01, high=0.01, size=2)

        # our dbsim obs have jacobian "center" set to 0, 0

        if model == 'bdf':
            fracdev = ur(low=0.45, high=0.55)
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                fracdev,
                # 1.0/nobj,
                flux,
            ]
            gm_model = ngmix.GMixBDF(pars=pars)
        elif model == 'bd':
            fracdev = ur(low=0.45, high=0.55)
            logTratio = ur(low=-0.01, high=0.01)
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                logTratio,
                fracdev,
                # 1.0/nobj,
                flux,
            ]
            gm_model = ngmix.GMixModel(pars, model)

        else:
            pars = [
                row,
                col,
                g1,
                g2,
                Tguess,
                # 1.0/nobj,
                flux,
            ]
            gm_model = ngmix.GMixModel(pars, model)

        logger.debug('gm model guess')
        logger.debug('\n%s' % str(gm_model))

        # perturb the models
        data = gm_model.get_data()
        for j in range(data.size):
            data['p'][j] *= (1 + ur(low=-0.05, high=0.05))

            fac = 0.01
            data['row'][j] += ur(low=-fac*pixel_scale, high=fac*pixel_scale)
            data['col'][j] += ur(low=-fac*pixel_scale, high=fac*pixel_scale)

            data['irr'][j] *= (1 + ur(low=-0.05, high=0.05))
            data['irc'][j] *= (1 + ur(low=-0.05, high=0.05))
            data['icc'][j] *= (1 + ur(low=-0.05, high=0.05))

        guess_pars += list(gm_model.get_full_pars())

    gm_guess = ngmix.GMix(pars=guess_pars)
    return gm_guess
