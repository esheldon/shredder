import testem

for seed in range(2000,3000):
     testem.test_descwl_fixcen_fromcoadd(
         nobj=1,
         show=True,
         noise_factor=.01,
         viewscale=0.05,
         width=2000,
         model='exp',
         seed=seed,
         title='1.0e-3',
         sim_config='dbsim-small.yaml',
         tol=1.0e-3,
     )

     if 'q' == input('hit a key (q to quit): '):
         break

