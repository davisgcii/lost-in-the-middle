"""Given a data file with KV records, get LM retrieval results.

The KV records are used in the exact order that they're given.
"""
import argparse
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

from dotenv import load_dotenv
from fastchat.model import get_conversation_template
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
    gold_index,
    query_aware_contextualization,
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

            if "chat" in model_name:  # TODO: check and see if this is the right way to do this
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                kv_prompt = format_chat_prompt(kv_prompt)
            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

    # Get responses for all of the prompts
    responses = []

    for prompt in tqdm(prompts):
        completion = getCompletion(prompt=prompt, temperature=temperature, top_p=top_p)
        print(f"Prompt: {prompt}\nCompletion: {completion}\n\n")  # TODO: remove this
        responses.append(completion)

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


def getCompletion(prompt, temperature=1.0, top_p=1.0):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def format_chat_prompt(input):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def format_chat_prompt(input):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--model", help="Model to use in generating responses", required=True)
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=1.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--gold-index", help="Move the key to retrieve to this index", type=int, required=True)
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.gold_index,
        args.query_aware_contextualization,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
