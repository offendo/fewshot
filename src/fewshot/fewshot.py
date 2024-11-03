#!/usr/bin/env python3

import random
import os
import sys
import pandas as pd
import logging
from sacrebleu import BLEU
from tqdm import tqdm
from typing import Callable, Iterable, Mapping, Sequence
from openai import OpenAI
from argparse import ArgumentParser
import heapq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))


def retrieve_random(x: dict, examples: list, n_examples: int, template: str):
    return random.choices(examples, k=n_examples)


def retrieve_fixed(x: dict, examples: list, n_examples: int, template: str):
    return examples[:n_examples]


def retrieve_bleu(x: dict, examples: list, n_examples: int, template: str):
    bleu = BLEU(effective_order=True)
    inputs = template.format(**x)
    topk = heapq.nlargest(
        n_examples,
        [(y, bleu.sentence_score(inputs, [template.format(**y)]).score) for y in examples],
        key=lambda d: d[1],
    )
    return [p[0] for p in topk]


RETRIEVE_METHODS = {
    "fixed": retrieve_fixed,
    "random": retrieve_random,
    "bleu": retrieve_bleu,
}


def format_chatgpt_prompt(
    x: dict,
    instruction: str,
    system: str,
    template: str,
    examples: Sequence[dict],
    n_examples: int,
    retrieve_fn: Callable,
):
    selected = retrieve_fn(x, examples, n_examples, template)

    # Add the system info & prompt instructions.
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": instruction})
    for example in selected:
        # Format prompt with all but 'completion'
        inputs = {k: v for k, v in example.items() if k != "completion"}
        prompt = template.format(**inputs)

        # Add input/output to messages
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": example["completion"] + "<|endoftext|>"})

    # Add the final input to the messages
    messages.append({"role": "user", "content": template.format(**x)})

    return messages


def complete(model: str, messages: list, temperature: float = 0.0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = ArgumentParser(prog="fewshot", description="prompts LLM with examples")

    # fmt:off
    # IO stuff
    parser.add_argument("--file", "-f", type=str, required=True, help="JSONL file with inputs to process")
    parser.add_argument("--output", "-o", type=str, required=False, default=sys.stdout, help="path to output JSONL file")

    # Examples/prompt stuff
    parser.add_argument("--examples", "-e", type=str, required=False, help="JSONL file with examples to process. Must have 'completion' field for outputs.")
    parser.add_argument("--instruction", "-i", type=str, required=True, help="instruction to give to LLM")
    parser.add_argument("--template", "-t", type=str, required=True, help="prompt template to format inputs, like `template.format(**ex)`")
    parser.add_argument("--n_examples", "-n", type=int, required=False, default=1, help="number of examples to add to prompt")
    parser.add_argument("--retrieve", "-r", type=str, required=False, default="fixed", choices=RETRIEVE_METHODS.keys(), help="example retrieval method to use")

    # Model stuff
    parser.add_argument("--system", "-s", type=str, required=False, default="You are a helpful assistant.", help="system instruction")
    parser.add_argument("--llm", "-m", type=str, required=False, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"], help="name of OpenAI model to use")
    # fmt:on

    args = parser.parse_args()

    # Read in the input file
    # ======================
    df = pd.read_json(args.file, lines=True)
    logger.info(f"Input file contains {len(df)} examples.")

    if args.examples is not None:
        examples = pd.read_json(args.examples, lines=True).to_dict(orient="records")
        logger.info(f"Example file contains {len(examples)} examples.")
    else:
        examples = []
        logger.info(f"No examples provided - defaulting to zero-shot.")

    predictions = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        messages = format_chatgpt_prompt(
            x=row.to_dict(),
            instruction=args.instruction,
            system=args.system,
            template=args.template.replace("\\n", "\n"),
            examples=examples,
            n_examples=args.n_examples,
            retrieve_fn=RETRIEVE_METHODS[args.retrieve],
        )

        predictions.append(complete(args.llm, messages, 0.0))

    df["predictions"] = predictions
    df.to_json(args.output, orient="records", lines=True)
    logger.info(f"Saved predictions to {getattr(args.output, 'name', args.output)}")
