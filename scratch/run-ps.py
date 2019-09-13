import testem

tol = 1.0e-6

for seed in range(3005,100000):
    print('-'*70)
    print('seed:', seed)

    # for Tfac in [1.0, 1.05, 1.10]:

    for Tfac in [1.0]:
        title = 'Tfac: %g' % Tfac
        testem.test_multi_sep_ps(
            nobj=1,
            noise=.1,
            off=10,
            show=True,
            # Tmax=0,
            Tfac=Tfac,
            width=2000,
            seed=seed,
            tol=tol,
            title=title,
        )
    if 'q' == input('hit a key (q to quit): '):
        break

