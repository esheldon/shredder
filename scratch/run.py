import testem

for seed in range(3001,100000):
     testem.test_fixcen(
         nobj=200,
         show=True,
         noise_factor=1,
         viewscale=0.0005,
         width=2000,
         model='exp',
         seed=seed,
         title='1.0e-3',
         sim_config='dbsim-huge.yaml',
         tol=1.0e-3,
     )

     if 'q' == input('hit a key (q to quit): '):
         break

