"""Given a data file with KV records, get LM retrieval results.

The KV records are used in the exact order that they're given.
"""
import argparse
import json
import logging
import math
import openai
import pathlib
import random
import sys
from copy import deepcopy

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.prompting import get_kv_retrieval_prompt

load_dotenv()

logger = logging.getLogger(__name__)
random.seed(0)

client = OpenAI()


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    batch_size,
    gold_index,
    max_memory_per_gpu,
    query_aware_contextualization,
    max_new_tokens,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_ordered_kv_records = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
            key = input_example["key"]
            value = input_example["value"]

            original_kv_index = ordered_kv_records.index([key, value])
            # Remove the kv from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            ordered_kv_records.insert(gold_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )

            if "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                kv_prompt = format_chat_prompt(kv_prompt)
            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

    # Get responses for all of the prompts
    responses = []

    for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
        inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            # Disable use_cache if using longchat models with flash attention
            use_cache=not ("longchat" in model_name and longchat_flash_attn),
        )
        for i, generated_sequence in enumerate(outputs):
            input_ids = inputs["input_ids"][i]
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text)

    with xopen(output_path, "w") as f:
        for example, ordered_kv_records, prompt, response in zip(
            examples, all_model_ordered_kv_records, prompts, responses
        ):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_ordered_kv_records"] = ordered_kv_records
            f.write(json.dumps(output_example) + "\n")


def getCompletion(prompt):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    )
    return completion.choices[0].message
