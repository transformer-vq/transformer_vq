# Transformer-VQ

Tensor-parallel implementation of 'Transformer-VQ: Linear-Time Transformers via Vector Quantization'.

This branch is intended for training very large models on accelerator clusters.   
In addition, the model initialization and checkpointing occur in a distributed fashion.

### Environments tested on so far:
- CPU with 32 cores
- TPU v3 with 8 cores
- TPU v3 with 32 cores
- TPU v3 with 128 cores

### The current limitations:
- Global batch size >= num devices.
- Update length == sequence length.
- No dropout. 
- No sampling. 

### Implementation differences from the original paper:
- Weight decay is multiplied by peak learning rate and schedule factor.
- Weight decay is applied to all non-codebook parameters. 
- No biases in linear layers. 
- No gain or bias in RMSNorm.
- SHGA is the only supported head type (no MHA/MQA).
- Serial is the only supported cross-block reduction type.

## Single-Host Launch (CPU, GPU, and TPU)

The scripts use [W&B](https://wandb.ai/) for cloud-based logging. It is free for personal and academic use.

Clone the repo and install the dependencies. 
When not using TPUs, we strongly recommend installing all dependencies in a [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html) environment. 
```
git clone https://github.com/transformer-vq/transformer_vq/;
cd transformer_vq;
##### CPU
pip3 install -e '.[cpu]';
##### Nvidia GPU
pip3 install -e '.[gpu]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;
##### TPU VM
pip3 install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```

To launch an experiment, run
```
export WANDB_API_KEY=WB_KEY;  # if logging to W&B
python3 ./scripts/launch.py \
    --config=CONFIG_FILE \
    --multihost=MULTIHOST \
    --n_data_shard=N_DATA_SHARD \
    --n_model_shard=N_MODEL_SHARD \
    --workdir=WORKDIR \
    --mode=MODE \
    [--wb_enabled=WB_ENABLED] \
    [--wb_run=WB_RUN]
```
where
- ```CONFIG_FILE``` is a path to a configuration file,
- ```MULTIHOST``` is ```False``` single-host experiments,
- ```N_DATA_SHARD``` is the number of ways to shard the data.
- ```N_MODEL_SHARD``` is the number of ways to shard the model. 
- ```WORKDIR``` is a path to a local or cloud directory for experiment data. 
- ```MODE``` is one of ```train_vocab```, ```train```, ```validation```, or ```test```. 
- ```WB_ENABLED``` is optional and allows logging to W&B if ```True```. 
- ```WB_RUN``` is optional, and allows resuming logging of an existing W&B run. 

Settings in the specified configuration file can be overridden by appending a new setting.
For example, to change the batch size, append ```--config.global_batch_size=8```.

## Multi-Host Launch on GPU Clusters

For GPU clusters using Slurm or OpenMPI, our scripts should work without additional configuration. 
For other GPU clusters, you need to provide values for the coordinator IP and port, the number of processes, and a process ID for each process starting from 0.  
For example:
```
--gpu_coord_addr="192.168.0.1:1234" \
--gpu_n_process=2 \
--gpu_process_id=0 \
```
This information is passed to ```jax.distributed.initialize```. For more information, please refer to the [Jax docs](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html#jax.distributed.initialize).

## Multi-Host Launch on TPU Pod Slices

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
