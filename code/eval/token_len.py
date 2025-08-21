import logging
import random
import sys
import os
import json

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DiskDataArguments,
    # DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    #decontaminate_humaneval,
    get_checkpoint,
    get_disk_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from eval_config import EvalConfig

logger = logging.getLogger(__name__)

def get_mt_bench_data(file_path):
    ret_list = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line.strip())
            for choice in instance["choices"]:
                for text in choice["turns"]:
                    ret_list.append(text)
    return ret_list

def get_json_data(eval_args):
    ret_list = []

    file_path = os.path.join(eval_args.eval_root,
                             eval_args.eval_file_path)

    # print("file_path: ", file_path)

    if eval_args.dataset == "mt-bench":
        return get_mt_bench_data(file_path)
    else:
        pass


def main():
    parser = H4ArgumentParser((ModelArguments, DiskDataArguments, EvalConfig))
    model_args, data_args, eval_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = eval_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    ret_list = get_json_data(eval_args)


    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    cnt = 0
    total_ll = 0
    for s in ret_list:
        out = tokenizer(s)
        input_ids, mask = out["input_ids"], out["attention_mask"]
        ll = sum(mask)
        total_ll += ll
        cnt += 1
    print('avg: ', total_ll/cnt)    

    logger.info("*** Eval complete! ***")


if __name__ == "__main__":
    main()
