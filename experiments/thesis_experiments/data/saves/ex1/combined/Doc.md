Combine Suitesparse Matrices & generated and add more data to get at least 250 solver wins per class.
1. N_SAMPLES=20006 MAX_ITER=5000 STORE_MATRIX=1 SEED=1234 docker compose run -d datagen
2. SOLVER_TIMEOUT=60 MAX_N=100000 DATA_DIR=./data/saves/ex1/suite CACHE_DIR=../shared/cache docker compose run -d ingest
3. Merge: SRC_DIRS="./data/saves/ex1/ ./data/saves/ex1/suite/" OUT_DIR="./data/saves/ex1/combined/" docker compose run --rm merge
```cmd 
    Solver win distribution  (20565 samples, ../data/saves/ex1/combined/)
  Synthetic    :  20006  (97.3%)
  SuiteSparse  :    559  (2.7%)
Rank  Solver                      Wins       %
──────────────────────────────────────────────
  1   fbcgsr+ilu                  3190   15.5%  ███████
  2   fbcgsr+jacobi               2974   14.5%  ███████
  3   cg+eisenstat                2710   13.2%  ██████
  4   cr+jacobi                   1885    9.2%  ████
  5   cg+ilu                      1834    8.9%  ████
  6   cg+bjacobi                  1324    6.4%  ███
  7   minres+gamg                 1289    6.3%  ███
  8   cr+ilu                       985    4.8%  ██
  9   gmres+gamg                   887    4.3%  ██
  10  bcgsl+none                   828    4.0%  ██
  11  cr+eisenstat                 823    4.0%  ██
  12  symmlq+icc                   463    2.3%  █
  13  fgmres+gamg                  343    1.7%  
  14  bcgsl+asm                    231    1.1%  
  15  symmlq+jacobi                213    1.0%  
  16  cgs+gamg                     204    1.0%  
  17  fcg+gamg                     171    0.8%  
  18  symmlq+sor                   110    0.5%  
  19  dgmres+none                  101    0.5%  

── Matrix sizes (n = number of rows)  overall: min=99  median=7134  mean=12894  max=89999

  Solver                          N    min n    med n   mean n    max n
  -------------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi                2974       99     6946     8208    19991
  bcgsl+none                    828       99     2116     2796    27000
  symmlq+icc                    463      900    18769    20680    42874
  symmlq+jacobi                 213      138     2000     4194    46771
  dgmres+none                   101      586     8031     9349    52328
  gmres+gamg                    887      781    25280    24373    83521
  cr+eisenstat                  823      273     2196     4074    39304
  symmlq+sor                    110       99     5050     5029    13965
  fbcgsr+ilu                   3190       99     4954     8336    84617
  minres+gamg                  1289      362    59049    58946    89999
  fcg+gamg                      171     1357    25921    27978    89400
  cr+jacobi                    1885       99     2219     3174    15439
  cg+ilu                       1834       99     6889    10511    42874
  fgmres+gamg                   343     5928    18496    20567    87616
  cg+eisenstat                 2710       99     6859    12072    42874
  cg+bjacobi                   1324     1138    10609    15352    42874
  cr+ilu                        985       99     2024     3643    33124
  cgs+gamg                      204     1041    10000    11790    32040
  bcgsl+asm                     231      485     8281    12088    39999

```
4. N_SAMPLES=1027 STORE_MATRIX=1 docker compose run -d datagen
5. Now add up data to have at least 250 per class, for this the generator is changes so it uses better generator functions for missing classes to improve generation speed (also buckets over 19 are added from here on)
6. DATA_DIR=./data/saves/ex1/combined/ MIN_PER_CLASS=250 N_SAMPLES=1000 STORE_MATRIX=1 SEED=11111 MAX_ITER=4000 STATS_FILE=../shared/bucket_distribution.json docker compose run --rm datagen
7. DATA_DIR=./data/saves/ex1/combined/ MIN_PER_CLASS=250 N_SAMPLES=3783 STORE_MATRIX=1 SEED=11111 MAX_ITER=4000 STATS_FILE=../shared/bucket_distribution.json docker compose run --rm datagen
8. DATA_DIR=./data/saves/ex1/combined/ MIN_PER_CLASS=250 N_SAMPLES=50000 STORE_MATRIX=1 SEED=11112 MAX_ITER=4000 STATS_FILE=../shared/bucket_distribution.json docker compose run --rm datagen
9. now trim to max 600 (in the bak the full dataset is stored, to add other solvers quicker later prob.)
10. DATA_DIR=./data/saves/ex1/combined/ MAX_PER_CLASS=600 SUITESPARSE_ONLY=1 docker compose run --rm trim
```cmd 
    Solver win distribution  (9711 samples, ../data/saves/ex1/combined/)
  Synthetic    :   9152  (94.2%)
  SuiteSparse  :    559  (5.8%)
Rank  Solver                      Wins       %
──────────────────────────────────────────────
  1   cr+ilu                       600    6.2%  ███
  2   cg+eisenstat                 600    6.2%  ███
  3   cg+bjacobi                   600    6.2%  ███
  4   fbcgsr+jacobi                600    6.2%  ███
  5   gmres+gamg                   600    6.2%  ███
  6   fgmres+gamg                  600    6.2%  ███
  7   cg+ilu                       600    6.2%  ███
  8   cr+jacobi                    600    6.2%  ███
  9   minres+gamg                  600    6.2%  ███
  10  fbcgsr+ilu                   600    6.2%  ███
  11  cr+eisenstat                 600    6.2%  ███
  12  bcgsl+none                   600    6.2%  ███
  13  symmlq+icc                   600    6.2%  ███
  14  bcgsl+asm                    428    4.4%  ██
  15  dgmres+none                  354    3.6%  █
  16  cgs+gamg                     336    3.5%  █
  17  fcg+gamg                     290    3.0%  █
  18  symmlq+jacobi                253    2.6%  █
  19  symmlq+sor                   250    2.6%  █

── Matrix sizes (n = number of rows)  overall: min=99  median=8099  mean=15046  max=125000

  Solver                          N    min n    med n   mean n    max n
  -------------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi                 600      130     4265     6511    19960
  bcgsl+none                    600       99     2209     3173    39304
  symmlq+icc                    600      900    19683    27708   125000
  symmlq+jacobi                 253      138     3465     4508    46771
  dgmres+none                   354      421     2481     5192    52328
  gmres+gamg                    600      781    22200    22623    83521
  cr+eisenstat                  600      208     2597     5294   117648
  symmlq+sor                    250       99     4965     5290    64000
  fbcgsr+ilu                    600       99     4395     7917    84617
  minres+gamg                   600      362    58081    57861    89999
  fcg+gamg                      290     1357    26406    29279    89400
  cr+jacobi                     600       99     2531     3445    19683
  cg+ilu                        600       99     6874    11239   125000
  fgmres+gamg                   600     5928    21170    25816    87616
  cg+eisenstat                  600       99     5831    14986   125000
  cg+bjacobi                    600     1138    11342    15349    59318
  cr+ilu                        600       99     4038     4665    17779
  cgs+gamg                      336     1041    11024    14676    84681
  bcgsl+asm                     428      225     7921    10376    42874

Saved → /shared/Idea/PycharmProjects/IterativeSolverSelectionWithCNNs/experiments/thesis_experiments/viz/solver_wins.png

```

