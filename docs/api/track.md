# Experiment Tracking

`Tracker` is a small SQLite-backed experiment store. It lets you tag each training run with a name and config, log scalar metrics, save the computation graph, and query everything later.

Each **experiment** is a single `.db` file under your tracker directory; each experiment can hold many **runs**.

```
all_exp_dir/
├── mnist.db                     <- one experiment
│   ├── run 1 "baseline"  config={lr: 1e-3}    metrics: loss, acc
│   ├── run 2 "warmup"    config={lr: 5e-4}    metrics: loss, acc
│   └── ...
└── cifar.db
```

## Basic usage

```python
from simplegrad.track import Tracker

tracker = Tracker("./experiments")
tracker.set_experiment("mnist")
run_id = tracker.start_run(name="baseline", config={"lr": 1e-3, "bs": 64})

for step in range(num_steps):
    # ... train_step ...
    tracker.record("loss", loss.values.item(), step=step)
    tracker.record("acc",  current_acc,        step=step)

tracker.end_run()
```

## Plot afterwards

```python
from simplegrad.visual import plot

results = tracker.get_results(run_id)    # dict[metric] -> list[RecordInfo]
plot(results, selected=["loss", "acc"])
```

## Save the computation graph

```python
loss = loss_fn(model(x_batch), y_batch)
tracker.save_comp_graph(loss, run_id=run_id)
```

The graph is serialized to JSON next to the metrics, ready for browsing in SimpleBoard.

## Query historical runs

```python
runs = tracker.get_all_runs()                 # list[RunInfo]
records = tracker.get_records(runs[0].id, "loss")
```

---

::: simplegrad.track.tracker.Tracker

::: simplegrad.track.exp_db_manager.RunInfo

::: simplegrad.track.exp_db_manager.RecordInfo

::: simplegrad.track.exp_db_manager.ExperimentDBManager
