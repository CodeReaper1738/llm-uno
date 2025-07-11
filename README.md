# Using RLCard to implement an LLM driven NPC

Requirements to install/setup rlcard

make sure that you have **Python 3.10+** and **pip** installed.


## Install Required Dependencies
pip install transformers
pip install accelerate

**Authenticate with Hugging Face to access their models**
huggingface-cli login

**RLcard setup**

Go to the rlcard website and setup rlcard
## RLCard Documents
For more documentation, please refer to the [Documents](docs/README.md) for general introductions. API documents are available at our [website](http://www.rlcard.org).

**LLM Interface Setup** 

clone my repository inside of this rlcard path cd /path-to-folder/rlcard/examples/human

## Running the LLM-Driven Uno Agent
 

1. Navigate to the LLM Files

cd /path-to-folder/rlcard/examples/human

2. Adjust the template you want to send into the LLM

Update the prompt_template.txt file, which is passed into the LLM to guide its behavior.
We've provided examples of different types of templates in the prompt_examples.txt

2. LLM agent assisting one of two reinforcement learning agents (2rl, 1LLM)

Run:

python uno_rllogging.py --model <model_name_or_path>
Replace <model_name_or_path> with the name or path of the model you want to use.

API, specify the API key:
python openrouter_logging.py --apikey <your_api_key> 

3. LLM agent assisting either human or reinforcement learning agent (1 human, 1rl, 1LLM)

Run: 
python uno_human.py --model <model_name_or_path>


