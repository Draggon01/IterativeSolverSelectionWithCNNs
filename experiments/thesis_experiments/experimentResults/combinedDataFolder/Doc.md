Command for experiment execution:
this is for the testing of 

SKIP_DATAGEN=1 CACHE_DIR=../shared/cache/ RESULTS_FILE=./experimentResults/combinedDataFolder/combined_summary.txt MODES="binary density log_density magnitude symmetry diagonal rcm_binary rcm_density rcm_log_density rcm_magnitude" SIZES="64 128 256" ./run_experiments.sh 

2 new modes and time is now also documented:
SKIP_DATAGEN=1 CACHE_DIR=../shared/cache/ RESULTS_FILE=./experimentResults/combinedDataFolder/combined_summary.txt MODES="binary density log_density magnitude symmetry diagonal sign signed_magnitude rcm_binary rcm_density rcm_log_density rcm_magnitude rcm_sign rcm_signed_magnitude" SIZES="64 128 256" BATCH_SIZE=128 ./run_experiments.sh 
