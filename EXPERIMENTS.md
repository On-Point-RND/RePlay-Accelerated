Для переиспользования результатов CCE и CCE- для моделей SasRec и Bert4Rec необходимо:
1. В файле `replay_benchmarks/configs/config.yaml` указать соответствующие конфиги, отвечающие за датасет и модель
```yaml
defaults:
  - dataset: movielens_1m # Например movielens_1m
  - model: sasrec_movielens_1m_cce_minus # Например sasrec_movielens_1m_cce_minus
  - mode: train
  - acceleration: null

env:
  SEED: 42
  CUDA_DEVICE_ORDER: "PCI_BUS_ID" 
  OMP_NUM_THREADS: "4"
  CUDA_VISIBLE_DEVICES: "0"

paths:
  data_dir: "replay_benchmarks/data/"
  log_dir: "replay_benchmarks/artifacts/logs/"
  checkpoint_dir: "replay_benchmarks/artifacts/checkpoints/"
  results_dir: "replay_benchmarks/artifacts/results/"
  main_csv_res_dir: "replay_benchmarks/artifacts"

metrics:
  types:
    - ndcg
    - recall
    - hitrate
    - precision
    - map
    - mrr
  ks:
    - 1
    - 5
    - 10
    - 20
    - 100

```

2. Выбрать соответствующий конфиг папке `RePlay-Accelerated/replay_benchmarks/configs/model`. Для примера представлены конфиги:
    - `bert4rec_movielens_1m_cce_minus.yaml`
    - `bert4rec_movielens_1m_cce.yaml`
    - `sasrec_movielens_1m_cce_minus.yaml`
    - `sasrec_movielens_1m_cce.yaml`

3. Если необходим конфиг для другого датасета, то нужно написать аналогичный файл для данного датасета. Рекомендации по формированию:
    - batch_size, max_seq_len, loss_sample_count и hidden_size выбирать степенями двойки либо числами, кратными 32.
    - При использовании CCE- loss_sample_count выбирать нечетным числом, на 1 меньше желаемого (например хотим 128 -> loss_sample_count=127)
    - Если вы хотите сделать запуск с ССЕ, то loss_sample_count указать null. В противном случае будет использован ССЕ-