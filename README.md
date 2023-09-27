# Transformer-VQ

Official implementation of 'Transformer-VQ: Linear-Time Transformers via Vector Quantization'. 

## Single-Host Launch

The scripts use [W&B](https://wandb.ai/) for logging; it is free for personal and academic use.

Clone the repo and install the dependencies:
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
export WANDB_API_KEY=WB_KEY;
chmod +x ./scripts/lm_DATASET.sh;
./scripts/lm_DATASET.sh \
    -c COMMAND \
    -r RNG_SEED \
    -p PRECISION \
    -t SEQUENCE_LEN \
    -l BLOCK_LEN \
    -m MEM_LEN \
    -s CODEBOOK_SIZE \
    -a AGG_CACHE \
    -g GRAD_THRU_CACHE \
    -e EXTRA_STEPS \
    -i IN_CHECKPOINT_DIR \
    -o OUT_CHECKPOINT_DIR \
    -d DATASETS_DIR \
    -v VOCAB_PATH \
    [-w WB_RUN_ID]
```
where 
- ```DATASET``` is one of ```{enwik8,pg19,imagenet64}```
- ```COMMAND``` is one of ```{train_vocab,train,val,test,sample}```
- ```RNG_SEED``` is the experiment seed or sampling seed
- ```PRECISION``` is one of ```{float32,bfloat16}```
- ```SEQUENCE_LEN``` is the sequence length
- ```BLOCK_LEN``` is an integer divisor of the sequence length and update length
- ```MEM_LEN``` is the uncompressed key/value cache length (set to ```BLOCK_LEN``` in our experiments) 
- ```CODEBOOK_SIZE``` is the number of codebook rows/compressed cache length
- ```AGG_CACHE``` is 0/1 to exclude/include the compressive cache
- ```GRAD_THRU_CACHE``` is 0/1 to stop/allow gradients through the caches
- ```EXTRA_STEPS``` is the number of constant learning rate steps to run after the cosine learning rate schedule (set to 0 in our experiments)
- ```INPUT_CHECKPOINT_DIR``` is a folder name for loading checkpoints
- ```OUTPUT_CHECKPOINT_DIR``` is a folder name for saving checkpoints
- ```DATASETS_DIR``` is a path for saving downloaded datasets locally or in Google Cloud Storage
- ```VOCAB_PATH``` is a path for a sentencepiece vocabulary, used for the PG-19 model
- ```WB_KEY``` can be obtained from ```https://wandb.ai/authorize```
- ```WB_RUN_ID``` should be the run ID from the W&B run URL, if resuming a run

### Training a vocabulary

To use the PG-19 model, you need a [SentencePiece](https://github.com/google/sentencepiece) vocabulary.

- An external SentencePiece vocabulary hosted on Google Cloud Storage can be used. For example, you can use the [T5 vocabulary](```gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model```).

- Alternately, a SentencePiece vocabulary can be trained on the PG-19 training corpus. To do so, launch ```lm_pg19.sh``` with ```COMMAND``` set to ```train_vocab```. The trained SentencePiece model will be written to ```VOCAB_PATH```, and this path can be supplied for training, evaluation, and sampling. 

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
  --command="tmux new-session -d -s transformer_vq_session 'cd transformer_vq; export WANDB_API_KEY=WB_KEY; chmod +x ./scripts/lm_DATASET.sh; ./scripts/lm_DATASET.sh -x ...; bash;'"
```

The last line uses ```-x``` as a flag for the control script ```lm_DATASET.sh``` to enable multi-host mode, and uses ```...``` as a placeholder for other arguments, which follow single-host launch.

### Attaching and detaching tmux sessions
To view the script output, you can SSH into any of the TPU hosts
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

Multi-host mode for GPUs is not currently supported by our scripts, as it requires rendezvous information to be provided to Jax. Support will be added in the future. 