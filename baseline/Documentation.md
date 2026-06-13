#Data Generation (experiment 1: baseline)

1. Generate 50000 data entries with the mm_generate.py (seed: 42)
   1. mm_generate N_SAMPLES=50000 SEED=42
2. add the matrices specified in the github repos from suitesparse to have some 
   2. mm_ingest MODE=githubdata N_ITER = 20000
4. add larger matrices
   3. SEED=0 MIN_N=100 MAX_N=50000 MAX_ITERS=2000 N_MATRICES=27 docker compose run -d mm_ingest
   4. SEED=1 MIN_N=40000 MAX_N=140000 MAX_ITER=40000 N_MATRICES=100 SOLVER_TIMEOUT=400 docker compose run -d mm_ingest
6. Shortened exceeding data branches to prevent model from just guessing the most successful one
   5. just executed mm_trim.py
8. Now start training on laptop
   6. docker compose run --no-deps -d mm_trainer
9. as this does not reach 70% acc i will try to get more suitesparse matrices in there, maybe this helps
    7. results where this: 
##### Dataset
```cmd 
── Dataset distribution  (./data/dataset.h5) ──────────────────
  Total samples : 23167  |  Paper total : 11,623

  Solver                  Ours      %   Paper      %  Bar (ours)
  --------------------  ------  -----  ------  -----  -------------------------
  fbcgsr+jacobi           2519  10.9%    2173  18.7%  ██
  bcgsl+none              2502  10.8%    2054  17.7%  ██
  symmlq+icc               162   0.7%    1201  10.3%  
  symmlq+jacobi            211   0.9%     923   7.9%  
  dgmres+none               43   0.2%     650   5.6%    ⚠ low
  gmres+gamg               394   1.7%     640   5.5%  
  cr+eisenstat            2505  10.8%     598   5.1%  ██
  symmlq+sor               109   0.5%     582   5.0%  
  fbcgsr+ilu              1247   5.4%     562   4.8%  █
  minres+gamg              412   1.8%     524   4.5%  
  fcg+gamg                 143   0.6%     342   2.9%  
  cr+jacobi               2503  10.8%     310   2.7%  ██
  cg+ilu                  2511  10.8%     275   2.4%  ██
  fgmres+gamg              146   0.6%     226   1.9%  
  cg+eisenstat            2525  10.9%     224   1.9%  ██
  cg+bjacobi              2503  10.8%     193   1.7%  ██
  cr+ilu                  2500  10.8%      68   0.6%  ██
  cgs+gamg                 106   0.5%      49   0.4%  
  bcgsl+asm                126   0.5%      29   0.2%  

  TOTAL                  23167   100%   11623   100%

  Empty classes : 0   Low (<50) classes : 1

── Matrix sizes (n = number of rows) ───────────────────────────────
  Overall  min=100  median=1485  mean=8186  max=84617

  Solver                     N    min n    med n   mean n    max n
  --------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi           2519      102    14668    14425    29993
  bcgsl+none              2502      100      321      440    16609  ⚠ small
  symmlq+icc               162      400     1728     3144    39304
  symmlq+jacobi            211      168     1813     1542     9000
  dgmres+none               43      341     3240     4632    27276
  gmres+gamg               394     1220    28900    26201    57735
  cr+eisenstat            2505      100     1339     2384    39304
  symmlq+sor               109      100      343      940     8192  ⚠ small
  fbcgsr+ilu              1247      100      272      582    84617  ⚠ small
  minres+gamg              412     1036    33489    31536    39601
  fcg+gamg                 143     4257    32761    31243    39601
  cr+jacobi               2503      101      647      720    46772
  cg+ilu                  2511      100     9261    12300    39601
  fgmres+gamg              146     2000    28392    28504    39601
  cg+eisenstat            2525      100     2744    11226    39304
  cg+bjacobi              2503      127    15625    17610    39601
  cr+ilu                  2500      100      729     1627    39304
  cgs+gamg                 106     1182    28224    28237    39601
  bcgsl+asm                126      348     1350     2025    10000

```
##### Trainin Result
```cmd
python mm_evaluate.py 
2026-06-12 13:43:46,132 INFO Checkpoint: epoch=256  val_acc=0.4326

── MM-AutoSolver Baseline Results ──────────────────────────────
  Checkpoint   : epoch 256
  Val samples  : 2316  (of 23167 total)
  Classes      : 19

  Accuracy (Acc) : 43.26%   (paper: 78.54%)
  Macro Precision: 41.26%   (paper: 63.41%)
  Macro Recall   : 38.68%   (paper: 62.81%)
  Macro F1       : 34.30%   (paper: 62.53%)

  Top-2 Accuracy : 68.39%
  Top-3 Accuracy : 81.48%

  Near-optimal Acc (±5%)  : 52.63%
  Near-optimal Acc (±10%) : 60.15%
  Near-optimal Acc (±20%) : 70.81%

  Solver                 Our N  Paper N      F1    Prec     Rec
  --------------------  -------  -------  ------  ------  ------
  fbcgsr+jacobi           2519     2173   86.2%   89.4%   83.3%
  bcgsl+none              2502     2054   63.8%   62.0%   65.7%
  symmlq+icc               162     1201    8.1%    4.3%   72.2%
  symmlq+jacobi            211      923   36.8%   38.9%   35.0%
  dgmres+none               43      650   61.5%   80.0%   50.0%
  gmres+gamg               394      640   26.7%   60.0%   17.1%
  cr+eisenstat            2505      598   51.3%   54.5%   48.4%
  symmlq+sor               109      582    7.3%    4.3%   22.2%
  fbcgsr+ilu              1247      562   33.2%   35.3%   31.3%
  minres+gamg              412      524   20.4%   55.6%   12.5%
  fcg+gamg                 143      342   18.2%   11.1%   50.0%
  cr+jacobi               2503      310   52.0%   52.0%   52.0%
  cg+ilu                  2511      275    5.5%   33.3%    3.0%
  fgmres+gamg              146      226   12.0%    9.1%   17.6%
  cg+eisenstat            2525      224   40.4%   45.5%   36.2%
  cg+bjacobi              2503      193   48.4%   43.7%   54.3%
  cr+ilu                  2500       68   34.2%   51.6%   25.6%
  cgs+gamg                 106       49    5.8%    3.3%   25.0%
  bcgsl+asm                126       29   40.0%   50.0%   33.3%
─────────────────────────────────────────────────────────────────

```
7. further ingestion:
   1. MIN_N=5000 MAX_N=20000 N_MATRICES=1000 SEED=11 MAX_ITER=20000 docker compose run -d mm_ingest
2. add weight decay + fix splitting by class and run train on new dataset this time try first with the whole dataset 50k+ matrices
##### Dataset
```cmd 
── Dataset distribution  (./data/dataset.h5) ──────────────────
  Total samples : 50569  |  Paper total : 11,623

  Solver                  Ours      %   Paper      %  Bar (ours)
  --------------------  ------  -----  ------  -----  -------------------------
  fbcgsr+jacobi           7484  14.8%    2173  18.7%  ███
  bcgsl+none              5909  11.7%    2054  17.7%  ██
  symmlq+icc               161   0.3%    1201  10.3%  
  symmlq+jacobi            218   0.4%     923   7.9%  
  dgmres+none               64   0.1%     650   5.6%  
  gmres+gamg               395   0.8%     640   5.5%  
  cr+eisenstat            3734   7.4%     598   5.1%  █
  symmlq+sor               120   0.2%     582   5.0%  
  fbcgsr+ilu              1264   2.5%     562   4.8%  
  minres+gamg              416   0.8%     524   4.5%  
  fcg+gamg                 142   0.3%     342   2.9%  
  cr+jacobi               5384  10.6%     310   2.7%  ██
  cg+ilu                  6846  13.5%     275   2.4%  ███
  fgmres+gamg              144   0.3%     226   1.9%  
  cg+eisenstat            2523   5.0%     224   1.9%  █
  cg+bjacobi              3967   7.8%     193   1.7%  █
  cr+ilu                 11555  22.8%      68   0.6%  █████
  cgs+gamg                 109   0.2%      49   0.4%  
  bcgsl+asm                134   0.3%      29   0.2%  

  TOTAL                  50569   100%   11623   100%

  Empty classes : 0   Low (<50) classes : 0

── Matrix sizes (n = number of rows) ───────────────────────────────
  Overall  min=100  median=1331  mean=7202  max=84617

  Solver                     N    min n    med n   mean n    max n
  --------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi           7484      100    14722    14404    29994
  bcgsl+none              5909      100      323      445    26003  ⚠ small
  symmlq+icc               161      400     1728     3018    39304
  symmlq+jacobi            218      168     1813     1867    18368
  dgmres+none               64      341     5904     6924    27276
  gmres+gamg               395     1220    28900    25963    57735
  cr+eisenstat            3734      100     1341     2392    39304
  symmlq+sor               120      100      398     1634    17361  ⚠ small
  fbcgsr+ilu              1264      100      275      714    84617  ⚠ small
  minres+gamg              416     1036    33306    31323    39601
  fcg+gamg                 142     4257    32761    31290    39601
  cr+jacobi               5384      100      655      749    46772
  cg+ilu                  6846      100    10201    12605    39601
  fgmres+gamg              144     2000    28392    28559    39601
  cg+eisenstat            2523      100     3375    11314    39304
  cg+bjacobi              3967      127    15625    17428    39601
  cr+ilu                 11555      100      729     1661    39601
  cgs+gamg                 109     1182    28224    27714    39601
  bcgsl+asm                134      348     1409     2614    18588

```

##### Trainin Result : LEARNING_RATE=5e-4 docker compose run --no-deps -d mm_trainer
```cmd 
2026-06-13 07:29:28,774 INFO Checkpoint: epoch=256  val_acc=0.4481

── MM-AutoSolver Baseline Results ──────────────────────────────
  Checkpoint   : epoch 256
  Val samples  : 5056  (of 50569 total)
  Classes      : 19

  Accuracy (Acc) : 62.38%   (paper: 78.54%)
  Macro Precision: 57.10%   (paper: 63.41%)
  Macro Recall   : 63.57%   (paper: 62.81%)
  Macro F1       : 51.35%   (paper: 62.53%)

  Top-2 Accuracy : 80.52%
  Top-3 Accuracy : 90.21%

  Near-optimal Acc (±5%)  : 68.73%
  Near-optimal Acc (±10%) : 73.42%
  Near-optimal Acc (±20%) : 78.20%

  Solver                 Our N  Paper N      F1    Prec     Rec
  --------------------  -------  -------  ------  ------  ------
  fbcgsr+jacobi           7484     2173   98.7%   98.6%   98.7%
  bcgsl+none              5909     2054   96.7%   97.1%   96.3%
  symmlq+icc               161     1201    3.3%    1.7%   86.7%
  symmlq+jacobi            218      923   94.3%   89.3%  100.0%
  dgmres+none               64      650   85.7%   81.8%   90.0%
  gmres+gamg               395      640   20.4%   33.3%   14.7%
  cr+eisenstat            3734      598   85.5%   95.3%   77.5%
  symmlq+sor               120      582   12.6%    6.9%   75.0%
  fbcgsr+ilu              1264      562   94.1%   92.2%   96.0%
  minres+gamg              416      524   35.6%   23.0%   78.4%
  fcg+gamg                 142      342    9.1%   11.1%    7.7%
  cr+jacobi               5384      310   95.6%   96.1%   95.2%
  cg+ilu                  6846      275    0.9%  100.0%    0.4%
  fgmres+gamg              144      226    8.2%    4.4%   50.0%
  cg+eisenstat            2523      224   44.2%   34.8%   60.5%
  cg+bjacobi              3967      193   42.5%   36.8%   50.4%
  cr+ilu                 11555       68   52.8%   82.5%   38.8%
  cgs+gamg                 109       49    0.0%    0.0%    0.0%
  bcgsl+asm                134       29   95.7%  100.0%   91.7%
─────────────────────────────────────────────────────────────────

```
3. than try with downsampled one again, only remove matrices that are not suitesparse ones. (MAX_PER_CLASS=2000 SUITESPARSE_ONLY=1 docker compose run mm_trim)
##### Dataset
```cmd 
── Dataset distribution  (./data/dataset.h5) ──────────────────
  Total samples : 19167  |  Paper total : 11,623

  Solver                  Ours      %   Paper      %  Bar (ours)
  --------------------  ------  -----  ------  -----  -------------------------
  fbcgsr+jacobi           2000  10.4%    2173  18.7%  ██
  bcgsl+none              2000  10.4%    2054  17.7%  ██
  symmlq+icc               161   0.8%    1201  10.3%  
  symmlq+jacobi            218   1.1%     923   7.9%  
  dgmres+none               64   0.3%     650   5.6%  
  gmres+gamg               395   2.1%     640   5.5%  
  cr+eisenstat            2000  10.4%     598   5.1%  ██
  symmlq+sor               120   0.6%     582   5.0%  
  fbcgsr+ilu              1264   6.6%     562   4.8%  █
  minres+gamg              416   2.2%     524   4.5%  
  fcg+gamg                 142   0.7%     342   2.9%  
  cr+jacobi               2000  10.4%     310   2.7%  ██
  cg+ilu                  2000  10.4%     275   2.4%  ██
  fgmres+gamg              144   0.8%     226   1.9%  
  cg+eisenstat            2000  10.4%     224   1.9%  ██
  cg+bjacobi              2000  10.4%     193   1.7%  ██
  cr+ilu                  2000  10.4%      68   0.6%  ██
  cgs+gamg                 109   0.6%      49   0.4%  
  bcgsl+asm                134   0.7%      29   0.2%  

  TOTAL                  19167   100%   11623   100%

  Empty classes : 0   Low (<50) classes : 0

── Matrix sizes (n = number of rows) ───────────────────────────────
  Overall  min=100  median=1518  mean=8252  max=84617

  Solver                     N    min n    med n   mean n    max n
  --------------------  ------  -------  -------  -------  -------
  fbcgsr+jacobi           2000      102    14633    14297    29952
  bcgsl+none              2000      100      326      502    17922  ⚠ small
  symmlq+icc               161      400     1728     3018    39304
  symmlq+jacobi            218      168     1813     1867    18368
  dgmres+none               64      341     5904     6924    27276
  gmres+gamg               395     1220    28900    25963    57735
  cr+eisenstat            2000      100     1344     2431    39304
  symmlq+sor               120      100      398     1634    17361  ⚠ small
  fbcgsr+ilu              1264      100      275      714    84617  ⚠ small
  minres+gamg              416     1036    33306    31323    39601
  fcg+gamg                 142     4257    32761    31290    39601
  cr+jacobi               2000      105      661      870    46772
  cg+ilu                  2000      100     9261    12304    39601
  fgmres+gamg              144     2000    28392    28559    39601
  cg+eisenstat            2000      100     3254    11239    39304
  cg+bjacobi              2000      127    15625    17121    39601
  cr+ilu                  2000      100      729     1499    37636
  cgs+gamg                 109     1182    28224    27714    39601
  bcgsl+asm                134      348     1409     2614    18588

```

##### Trainin Result : LEARNING_RATE=1e-4 docker compose run --no-deps -d mm_trainer
```cmd 
```
4. than try with the distribution from the paper
##### Dataset
```cmd 
```

##### Trainin Result
```cmd 
```