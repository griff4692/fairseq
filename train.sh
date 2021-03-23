BART_RESTORE='/home/griffin/weights/bart.large/model.pt'
BART_OUT_DIR='/home/griffin/weights/bart.finetune/'
ENCODER_JSON='/home/griffin/fairseq/data/gpt2/encoder_special_toks.json'
VOCAB_BPE='/home/griffin/fairseq/data/gpt2/vocab.bpe'
IN_DIR=/home/griffin/nlp/projects/kabupra/mimic/clinbart/bin
LR=3e-5
TOTAL_STEPS=200000
WARMUP=10000


python fairseq_cli/train.py /home/griffin/bin \
--restore-file $BART_RESTORE \
--save-dir $BART_OUT_DIR \
--gpt2-encoder-json $ENCODER_JSON \
--gpt2-vocab-bpe $VOCAB_BPE \
--wandb-project clin-lm	\
--reset-optimizer --reset-dataloader --reset-meters \
--lr [$LR] \
--warmup-updates $WARMUP \
--total-num-updates $TOTAL_STEPS \
--update-freq=[8] \
--max-tokens=1024 \
--max-tokens-valid 1024 \
--tokens-per-sample=512 \
--max-source-positions=1024 \
--max-target-positions=1024 \
--save-interval 1 \
--save-interval-updates=5000 \
--keep-interval-updates 10 \
--no-epoch-checkpoints \
--adam-betas="(0.9, 0.98)" \
--num-workers 8 \
--log-format json \
--arch bart_large \
--train-subset train \
--valid-subset valid \
--best-checkpoint-metric loss \
--task denoising \
--sample-break-mode=eos \
--seed=1992 \
--log-format=json \
--fp16 \
--bpe gpt2 \
--pooler-activation-fn tanh \
--share-all-embeddings \
--layernorm-embedding \
--share-decoder-input-output-embed \
--optimizer adam \
--criterion cross_entropy \
--encoder-learned-pos \
--decoder-learned-pos \
--skip-invalid-size-inputs-valid-test \
--lr-scheduler polynomial_decay \
--weight-decay=0.01 \
--power=1.0 \
--mask=0.3 \
--mask-random=0.1 \
--poisson-lambda=3.0 \
--permute-sentences=1.0 \
--mask-length=span-poisson \
--replace-length=1 \
--insert=0.0 \
--rotate=0.0 \
--activation-fn=gelu \
--adam-eps=1e-06 \
