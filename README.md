# gnn_control

Proof-of-concept for training distributed controllers using graph neural networks (GNN)

1. Modify training / evaluation configuration file:
```
    python scripts/config.py
```

2. Train GNN controller:
```
    python scripts/train.py
```
By default, trained models are saved in `./results/gnc/`

3. Evaluate GNN controller:
```
    python scripts/train.py --eval --dir={path_to_model}
```