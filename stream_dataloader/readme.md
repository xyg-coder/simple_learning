# Stream Dataloader

Use resnet to benchmark that the help of pre-transfering one batch to gpu.

```
python -m stream_dataloader.benchmark

Iteration 50 completed
Iteration 100 completed
Iteration 150 completed
Iteration 200 completed
Iteration 250 completed
Iteration 300 completed
Iteration 350 completed
Iteration 400 completed
Iteration 450 completed
=========================
Benchmarking resnet50
average for mlenv.trainer.DATA_LOAD_time_ms: 0.01049361753463745 ms
average for mlenv.trainer.FORWARD_time_ms: 0.2882258996963501 ms
average for mlenv.trainer.BACKWARD_time_ms: 0.14624647045135497 ms
Throughput: 2.247311613383155 samples/s
=========================
Iteration 50 completed
Iteration 100 completed
Iteration 150 completed
Iteration 200 completed
Iteration 250 completed
Iteration 300 completed
Iteration 350 completed
Iteration 400 completed
Iteration 450 completed
=========================
Benchmarking resnet50_prefetch
average for mlenv.trainer.DATA_LOAD_time_ms: 0.008268608570098876 ms
average for mlenv.trainer.FORWARD_time_ms: 0.2616351842880249 ms
average for mlenv.trainer.BACKWARD_time_ms: 0.14252669525146483 ms
Throughput: 2.42459836077699 samples/s
=========================
```

Seems it can help reduce 20% dataloading time. It should help a lot if the dataloading is the main bottleneck.
