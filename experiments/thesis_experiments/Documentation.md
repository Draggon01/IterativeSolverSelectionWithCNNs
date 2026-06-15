1. Generate 20000 syntetic matrices (with the whole matrice stored for easy downsampling later) : VERBOSE=1 N_SAMPLES=30000 MAX_ITE
R=5000 STORE_MATRIX=1 SEED=1234 docker compose run -d datagen
2. Add githubdata matrices
3. Add further suitesparse matrices.
4. Run the run_experiment script