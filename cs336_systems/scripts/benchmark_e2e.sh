python -m cs336_systems.benchmark --output_file e2e.csv --device auto --n_layers 12 --n_heads 12 --d_ff 3072 --d_model 768
python -m cs336_systems.benchmark --output_file e2e.csv --device auto --n_layers 24 --n_heads 16 --d_ff 4096 --d_model 1024
python -m cs336_systems.benchmark --output_file e2e.csv --device auto --n_layers 36 --n_heads 20 --d_ff 5120 --d_model 1280
python -m cs336_systems.benchmark --output_file e2e.csv --device auto --n_layers 48 --n_heads 25 --d_ff 6400 --d_model 1600
python -m cs336_systems.benchmark --output_file e2e.csv --device auto --n_layers 32 --n_heads 32 --d_ff 10240 --d_model 2560