MODEL=${MODEL:-Qwen/Qwen2.5-32B}   # TASK=DROP GPU=4 LR=0.001 EPS=0.05 RANK=8 PKEEP=4 STEP_INTERVAL=100 bash svdlozo.sh
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
EPS=${EPS:-5e-2}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-10000}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LR=${LR:-1e-3}
TASK=${TASK:-SST2} # SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
SEED=${SEED:-0}
RANK=${RANK:-8}
P=${P:-${PKEEP:-4}}
LOAD_FLOAT16=${LOAD_FLOAT16:-1}
# STEP_INTERVAL_v=1
STEP_INTERVAL_u=${STEP_INTERVAL_u:-${STEP_INTERVAL:-100}}
Tainer=${Tainer:-${TRAINER:-SVDLOZO}}
Projection=${Projection:-${PROJECTION:-Gaussian}}    # Rademacher, Gaussian, Uniform
# export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
gpu_no="${GPU:-4}"
if [[ -z "$gpu_no" ]]; then
    if [[ "${1:-}" == "--gpus" || "${1:-}" == "-g" ]]; then
        gpu_no="${2:-3}"
        shift 2
    elif [[ -n "${1:-}" && "${1:0:1}" != "-" ]]; then
        gpu_no="$1"
        shift
    else
        gpu_no="3"
    fi
fi
de=${de:-""}
cmd="python"

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL_u-$RANK-$P-$Projection


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: $RANK"
echo "PKEEP: $P"
echo "STEP INTERVAL: $STEP_INTERVAL_u"
echo "GPUS: $gpu_no"
# echo "STEP INTERVAL: $STEP_INTERVAL_v"

if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 2098 --wait-for-client"
fi
PRECISION_ARGS=""
if [[ "$LOAD_FLOAT16" == "1" || "$LOAD_FLOAT16" == "true" || "$LOAD_FLOAT16" == "True" ]]; then
    PRECISION_ARGS="--load_float16"
fi
CUDA_VISIBLE_DEVICES="$gpu_no" $cmd run_svdlozo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result1/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS \
    --train_as_classification \
    --step_u_interval $STEP_INTERVAL_u \
    --rank_r $RANK \
    --p_keep $P \
    --projection $Projection \
    $PRECISION_ARGS \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 


