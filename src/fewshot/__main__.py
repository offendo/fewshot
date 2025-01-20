from fewshot.fewshot import *
from argparse import ArgumentParser
import sys
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser(prog="fewshot", description="prompts LLM with examples")

# fmt:off
# IO stuff
parser.add_argument("--file", "-f", type=str, required=True, help="JSONL file with inputs to process")
parser.add_argument("--output", "-o", type=str, required=False, default=sys.stdout, help="path to output JSONL file")

# Examples/prompt stuff
parser.add_argument("--examples", "-e", type=str, required=False, help="JSON file with examples to process. Must have 'completion' field for outputs.")
parser.add_argument("--instruction", "-i", type=str, required=True, help="instruction to give to LLM")
parser.add_argument("--template", "-t", type=str, required=True, help="prompt template to format inputs, like `template.format(**ex)`")
parser.add_argument("--n_examples", "-n", type=int, required=False, default=1, help="number of examples to add to prompt")
parser.add_argument("--retrieve", "-r", type=str, required=False, default="fixed", choices=RETRIEVE_METHODS.keys(), help="example retrieval method to use")

# Model stuff
parser.add_argument("--system", "-s", type=str, required=False, default=None, help="system instruction")
parser.add_argument("--llm", "-m", type=str, required=False, default="gpt-3.5-turbo", help="name of OpenAI model to use")
parser.add_argument("--response_schema", type=str, required=False, default=None, help="path to JSON schema")
parser.add_argument("--json_mode", action='store_true', help="enable JSON mode or not")
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

    predictions.append(
        complete(
            args.llm, messages=messages, temperature=0.0, response_schema=args.response_schema, json_mode=args.json_mode
        )
    )

df["predictions"] = predictions
df.to_json(args.output, orient="records", lines=True)
logger.info(f"Saved predictions to {getattr(args.output, 'name', args.output)}")
