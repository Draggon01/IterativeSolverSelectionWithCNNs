1. Generate 20000 syntetic matrices (with the whole matrice stored for easy downsampling later) : N_SAMPLES=20000 STORE_MATRIX=1 SEED=42
 docker compose run -d datagen
2. Add githubdata matrices
3. Add further suitesparse matrices.
4. Run the run_experiment script