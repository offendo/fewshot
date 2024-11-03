# fewshot

Simple fewshot learning with ChatGPT. 

## Usage

``` sh
❯ python -m fewshot --help
usage: fewshot [-h] --file FILE [--output OUTPUT] [--examples EXAMPLES] --instruction INSTRUCTION --template TEMPLATE [--n_examples N_EXAMPLES] [--retrieve {fixed,random,bleu}] [--system SYSTEM]
               [--llm {gpt-3.5-turbo,gpt-4-turbo,gpt-4o}]

prompts LLM with examples

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  JSONL file with inputs to process
  --output OUTPUT, -o OUTPUT
                        path to output JSONL file
  --examples EXAMPLES, -e EXAMPLES
                        JSON file with examples to process. Must have 'completion' field for outputs.
  --instruction INSTRUCTION, -i INSTRUCTION
                        instruction to give to LLM
  --template TEMPLATE, -t TEMPLATE
                        prompt template to format inputs, like `template.format(**ex)`
  --n_examples N_EXAMPLES, -n N_EXAMPLES
                        number of examples to add to prompt
  --retrieve {fixed,random,bleu}, -r {fixed,random,bleu}
                        example retrieval method to use
  --system SYSTEM, -s SYSTEM
                        system instruction
  --llm {gpt-3.5-turbo,gpt-4-turbo,gpt-4o}, -m {gpt-3.5-turbo,gpt-4-turbo,gpt-4o}
                        name of OpenAI model to use

```

Here we have a JSONL file of examples to use in our prompt:
``` python
# examples.json 
{"prompt": "test1", "completion": "completion1"}
{"prompt": "test2", "completion": "completion2"}
{"prompt": "test3", "completion": "completion3"}
{"prompt": "test4", "completion": "completion4"}
{"prompt": "test5", "completion": "completion5"}

# inputs.json
{"prompt":"test6"}
```

Now, we can process a file of prompts called `inputs.json` with the instruction `"Solve the following task."`, fetching `n = 1` examples by nearest BLEU score. Outputs will be written to `stdout` by default.

``` sh
❯ python src/fewshot/fewshot.py              \
  --file inputs.json                         \
  --instruction "Solve the following task."  \
  --template "Inputs: {prompt}\n\nOutputs: " \
  --examples ./examples.json                 \
  --num_examples 1                           \
  --retrieve 'bleu'
  
# Output
{"prompt":"test6","predictions":"completion6"}
```
