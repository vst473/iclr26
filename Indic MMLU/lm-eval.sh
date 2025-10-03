# !/bin/bash

model_path="$1"
output_path="$2"
mode="$3"

echo "running for $model_path saving it to $output_path in $mode with "

export HF_DATASETS_CACHE="$HOME/.cache/huggingface/"
export OPENAI_API_KEY=""
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASKS=("mmlu_custom")

MODEL_PATHS=(
$model_path
)

OUTPUT_DIRS=(
$output_path
)

if [ ${#MODEL_PATHS[@]} -ne ${#OUTPUT_DIRS[@]} ]; then
    echo "❌ ERROR: MODEL_PATHS and OUTPUT_DIRS must have same length."
    exit 1
fi

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    LOG_FILE="$OUTPUT_DIR/task_status.log"

    mkdir -p "$OUTPUT_DIR"
    echo "================ Evaluating model: $MODEL_PATH ================" | tee -a "$LOG_FILE"

    for task in "${TASKS[@]}"; do
        # Check if the task is already marked as success
        if grep -Fxq "Task $task: SUCCESS" "$LOG_FILE"; then
            echo "Skipping already completed task: $task" | tee -a "$LOG_FILE"
            continue
        fi

        echo "Running task: $task" | tee -a "$LOG_FILE"
        lm_eval \
          --model vllm \
          --model_args pretrained="$MODEL_PATH,max_length=8192,tensor_parallel_size=8" \
          --include_path "/workspace/kundeshwar/vijay/benchmark/custom_tasks" \
          --tasks "$task" \
          --batch_size auto:40 \
          --output_path "$OUTPUT_DIR" \
          --log_samples \
          --trust_remote_code \
          --gen_kwargs "{'reasoning_effort':'$mode'}" 
        #   --apply_chat_template


        if [ $? -eq 0 ]; then
            echo "Task $task: SUCCESS" | tee -a "$LOG_FILE"
        else
            echo "Task $task: FAILURE" | tee -a "$LOG_FILE"
        fi
        echo "" | tee -a "$LOG_FILE"
    done

    echo "Finished evaluating model: $MODEL_PATH"
    echo "Log saved at: $LOG_FILE"
    echo ""

    echo "Running post-evaluation processing..." | tee -a "$LOG_FILE"

    python3 -c "
from convert_path import merge_paths
from accumulate_results import automatic_accumulate_results
merged_path = merge_paths('$OUTPUT_DIR', '$MODEL_PATH')
print('Merged Path:', merged_path)
automatic_accumulate_results(merged_path)
" || echo "Post-processing failed for $MODEL_PATH" | tee -a "$LOG_FILE"
done

echo "🎯 All model evaluations completed."

