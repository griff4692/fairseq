BART_RESTORE='/home/griffin/weights/bart.large/model.pt'
BART_OUT_DIR='/home/griffin/weights/debug/'
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
--total-num-update $TOTAL_STEPS \
--update-freq=[4] \
--max-tokens=2048 \
--max-tokens-valid=2048 \
--tokens-per-sample=512 \
--save-interval 1 \
--save-interval-updates=5000 \
--keep-interval-updates 5 \
--no-epoch-checkpoints \
--adam-betas="(0.9, 0.98)" \
--num-workers 8 \
--log-format tqdm \
--arch bart_large \
--train-subset train \
--valid-subset valid \
--task denoising \
--sample-break-mode=eos \
--seed=1992 \
--fp16 \
--bpe gpt2 \
--optimizer adam \
--skip-invalid-size-inputs-valid-test \
--lr-scheduler polynomial_decay \
--weight-decay=0.01 \
--mask=0.3 \
--mask-random=0.1 \
--permute-sentences=1.0 \
--mask-length=span-poisson \
--replace-length=1 \
--insert=0.0 \
--rotate=0.0 \
--adam-eps=1e-06;
#--clip-norm=0.1 \
#--attention-dropout 0.1 \
#--criterion label_smoothed_cross_entropy \
#--label-smoothing=0.1 \
#--find-unused-parameters;
