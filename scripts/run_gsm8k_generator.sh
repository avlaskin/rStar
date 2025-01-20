python run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --num_subquestions 5 \
    --num_a1_steps 5 \
    --disable_a5 \
    --start_idx 0 \
    --end_idx 4 \
    --model_ckpt meta-llama/Llama-3.2-3B-Instruct \
    --note default \
    --num_rollouts 16
