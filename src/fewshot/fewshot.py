#!/usr/bin/env python3

import heapq
import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Sequence

from openai import NOT_GIVEN, OpenAI
from sacrebleu import BLEU

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

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
    system: str | None,
    template: str,
    examples: Sequence[dict],
    n_examples: int,
    retrieve_fn: Callable,
):
    selected = retrieve_fn(x, examples, n_examples, template)

    # Add the system info & prompt instructions.
    if system is not None:
        messages = [{"role": "system", "content": system}]
    else:
        messages = []
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


def is_temp_supported(model):
    return False


def is_response_format_supported(model):
    return True


def parse_json_schema(path):
    with open(path, "r") as f:
        schema = json.load(f)
    return schema


def complete(
    model: str, messages: list, temperature: float = 0.0, response_schema: Path | None = None, json_mode: bool = False
):
    temp = temperature if is_temp_supported(model) else NOT_GIVEN
    fmt = (
        parse_json_schema(response_schema)
        if is_response_format_supported(model) and response_schema is not None
        else {"type": "json_object"} if json_mode else NOT_GIVEN
    )
    response = client.chat.completions.create(model=model, messages=messages, temperature=temp, response_format=fmt)
    return response.choices[0].message.content
