N_SAMPLES=20006 MAX_ITER=5000 STORE_MATRIX=1 SEED=1234 docker compose run -d datagen
```cmd
Solver win distribution  (20006 samples, ../data/saves/ex1/)
Rank  Solver                      Wins       %
──────────────────────────────────────────────
  1   fbcgsr+ilu                  3131   15.7%  ███████
  2   fbcgsr+jacobi               2945   14.7%  ███████
  3   cg+eisenstat                2692   13.5%  ██████
  4   cr+jacobi                   1833    9.2%  ████
  5   cg+ilu                      1827    9.1%  ████
  6   cg+bjacobi                  1323    6.6%  ███
  7   minres+gamg                 1268    6.3%  ███
  8   cr+ilu                       975    4.9%  ██
  9   gmres+gamg                   824    4.1%  ██
  10  cr+eisenstat                 817    4.1%  ██
  11  bcgsl+none                   801    4.0%  ██
  12  symmlq+icc                   460    2.3%  █
  13  fgmres+gamg                  343    1.7%  
  14  cgs+gamg                     199    1.0%  
  15  bcgsl+asm                    182    0.9%  
  16  fcg+gamg                     168    0.8%  
  17  symmlq+sor                    90    0.4%  
  18  symmlq+jacobi                 79    0.4%  
  19  dgmres+none                   49    0.2%  

── Matrix sizes (n = number of rows)  overall: min=99  median=7420  mean=13156  max=89999

  Solver                          N    min n    med n   mean n    max n
  -------------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi                2945       99     7083     8267    19991
  bcgsl+none                    801       99     2116     2792    27000
  symmlq+icc                    460     1599    18769    20773    42874
  symmlq+jacobi                  79      729     6240     6606     9801
  dgmres+none                    49      586    12688    11531    19774
  gmres+gamg                    824     7056    26244    26085    83521
  cr+eisenstat                  817      273     2140     4077    39304
  symmlq+sor                     90      140     5394     5268     9823
  fbcgsr+ilu                   3131       99     5093     8427    39999
  minres+gamg                  1268    17956    59536    59834    89999
  fcg+gamg                      168    17689    26244    28413    89400
  cr+jacobi                    1833       99     2209     3161     9999
  cg+ilu                       1827       99     6889    10545    42874
  fgmres+gamg                   343     5928    18496    20567    87616
  cg+eisenstat                 2692       99     6859    12140    42874
  cg+bjacobi                   1323     1225    10609    15363    42874
  cr+ilu                        975       99     2024     3646    33124
  cgs+gamg                      199     4356    10201    12048    32040
  bcgsl+asm                     182     2304     9024    14178    39999

```