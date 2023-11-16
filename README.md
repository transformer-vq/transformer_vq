# Transformer-VQ

Optimized implementation of 'Transformer-VQ: Linear-Time Transformers via Vector Quantization'.

Compared to the legacy implementation, this branch is optimized for *high throughput* and *fast compile times* rather than full flexibility. 
It does not yet support sampling, custom update lengths, or decoupling the k/v cache lengths from the block length. 

Support for model parallelism using pjit will be added in the next few weeks!

## Single-Host Launch

The scripts use [W&B](https://wandb.ai/) for cloud-based logging. It is free for personal and academic use.

Clone the repo and install the dependencies; we recommend using [venv](https://docs.python.org/3/library/venv.html) or similar to avoid overwriting the system python dependencies.
```
git clone https://github.com/transformer-vq/transformer_vq/;
cd transformer_vq;
##### CPU or GPU
pip3 install -e '.[no_tpu]';
##### TPU
pip3 install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```

To launch an experiment, run
```
export WANDB_API_KEY=WB_KEY;  # if logging to W&B
python3 ./scripts/launch.py \
    --config=CONFIG_FILE \
    --multihost=MULTIHOST \
    --workdir=WORKDIR \
    --mode=MODE \
    [--wb_enabled=WB_ENABLED] \
    [--wb_run=WB_RUN]
```
where
- ```CONFIG_FILE``` is a path to a configuration file,
- ```MULTIHOST``` is ```False``` single-host experiments,
- ```WORKDIR``` is a path to a local or cloud directory for experiment data. 
- ```MODE``` is one of ```train_vocab```, ```train```, ```validation```, or ```test```. 
- ```WB_ENABLED``` is optional and allows logging to W&B if ```True```. 
- ```WB_RUN``` is optional, and allows resuming logging of an existing W&B run. 

Settings in the specified configuration file can be overridden by appending a new setting.
For example, to change the batch size: 
```
python3 ./scripts/launch.py \
    --config=CONFIG_FILE \
    --multihost=MULTIHOST \
    --workdir=WORKDIR \
    --mode=MODE \
    --config.global_batch_size=8 \
```

## Multi-Host Launch - TPU Pod Slice

To launch on a TPU pod or pod slice, all commands can be run remotely as follows: 
```
##### switch to the correct project
##### 
gcloud config set project PROJECT_ID

##### set up bucket for transformer vq checkpoints, datasets, vocabs
##### LOCATION should contain ZONE used later 
##### e.g., use location EU-WEST-4 if zone will be europe-west4-a.
#####
gcloud storage buckets create gs://BUCKET_NAME \
    --location=LOCATION \
    --uniform-bucket-level-access \
    --public-access-prevention 

##### spin up tpu pod slice
##### preemptable flag can be omitted if you have an on-demand pod slice
#####
gcloud compute tpus tpu-vm create TPU_POD_SLICE_NAME \
  --zone ZONE \
  --accelerator-type SLICE_TYPE \
  --version TPU_SOFTWARE_VERSION \
  [--preemptable]

##### clone this repo and install the dependencies on each host vm
##### 
ssh-add ~/.ssh/google_compute_engine;
gcloud compute tpus tpu-vm ssh TPU_POD_SLICE_NAME \
  --zone ZONE \
  --worker=all \
  --command="git clone https://github.com/transformer-vq/transformer_vq/; cd transformer_vq; pip3 install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

##### launch the script inside a tmux session on each host vm
##### 
gcloud compute tpus tpu-vm ssh TPU_POD_SLICE_NAME \
  --zone ZONE \
  --worker=all \
  --command="tmux new-session -d -s transformer_vq_session 'cd transformer_vq; export WANDB_API_KEY=WB_KEY; python3 ./scripts/launch.py ...'"
```

On the last line, we have elided the flags required for ```./scripts/launch.py```. These must be provided similar to the single-host instructions, but with ```--multihost=True```.

### Attaching and detaching tmux sessions
To view the script output, you can SSH into any of the TPU hosts by running
```
gcloud compute tpus tpu-vm ssh TPU_POD_SLICE_NAME --zone ZONE --worker=WORKER_ID
```
Then attach the tmux session with
```
tmux attach -t transformer_vq_session
```
To return the session to detached mode, allowing it to continue running after you leave the ssh session, type Ctrl+b, then type d. 

### Killing scripts and tmux sessions

To kill the script running in the tmux session on all hosts, you can run the following on your local machine: it will SSH to each host and kill the script.

```
./scripts/kill.sh -n TPU_POD_SLICE_NAME -z ZONE -c NUM_HOSTS
```

To kill the tmux session you can run
```
gcloud compute tpus tpu-vm ssh TPU_POD_SLICE_NAME \
  --zone ZONE \
  --worker=all \
  --command="tmux kill-session -t transformer_vq_session" 
``` 

### Deleting the instance
You can delete the pod/slice instance as follows:
```
gcloud compute tpus tpu-vm delete TPU_POD_SLICE_NAME --zone ZONE
```

## Multi-Host Launch - GPU Clusters

Multi-host mode for GPUs is not currently supported by our scripts, as it requires rendezvous information to be provided to Jax. Support will be added soon.

## Training speed tests

Commands to speed test are as follows. 
For convenience, set some environment variables ```WORKDIR``` and ```SEQ_LEN``` before running. 
For example:
```
export WORKDIR=workdir;
export SEQ_LEN=2048;
```

### VQ Attention, single-head gated (SHGA, aka GAU):
```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=vq \
    --config.head_type=shga \
    --config.sequence_len="$SEQ_LEN";
```

### VQ Attention, multi-query (MQA):

```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=vq \
    --config.head_type=mqa \
    --config.sequence_len="$SEQ_LEN";
```

### VQ Attention, multi-head (MHA):

```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=vq \
    --config.head_type=mha \
    --config.sequence_len="$SEQ_LEN";
```

### Quadratic-time attention, single-head gated (SHGA, aka GAU):
```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=full \
    --config.head_type=shga \
    --config.sequence_len="$SEQ_LEN";
```

### Quadratic-time attention, multi-query (MQA):

```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=full \
    --config.head_type=mqa \
    --config.sequence_len="$SEQ_LEN";
```

### Quadratic-time attention, multi-head (MHA):

```
python3 scripts/launch.py \
    --config=scripts/config_throughput.py \
    --workdir="$WORKDIR" \
    --multihost=False \
    --mode=train \
    --config.attn_type=full \
    --config.head_type=mha \
    --config.sequence_len="$SEQ_LEN";
```

## FLOP/s estimation

[FLOP/s](https://en.wikipedia.org/wiki/FLOPS) can be estimated in our scripts using Jax's [AOT compilation](https://jax.readthedocs.io/en/latest/aot.html) functionality to produce the FLOP count first. 
However, the estimate is not currently available on TPU, and on CPU Jax's calculation appears to be *incorrect*: the flop count is even less than product of the parameter count and the local batch size in tokens. 

As a result, we discourage users from estimating the flop count using ```--config.mode=flop_count``` until this issue is resolved.
