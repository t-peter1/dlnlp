# Repository for Deep Learning for Natural Language Processing course Project

## Running our implementation the fine-tuning process:
Install the requirements
```bash
pip install -r requirements.txt
```
Run train.py with
```bash
python train.py
```
Depending on the GPU, in 20 mins/ 1-2 hours you should have the weights output in the main project folder.
Fine-tuning process on GPT-2 medium is done with LoRA. 

## Running the evaluation
```bash
python eval_metrics.py
```
This injects the lora weights (weight-file can be changed in the configuration part, does not accept args so lora_weights_r4_preproccess.pt is hardcoded), then using inference generates the sentences needed for the E2E NLG challenge. After the evaluation process is done, it prints the evaluation scores BLEU, ROUGE-L and METEOR.

## Using inference for example text generation
```bash
python inference.py
```
The weights are injected again to GPT-2 medium with an example prompt, the output is printed.

## The text generation with the same prompt on the non fine-tuned GPT2
Run with 
```bash
python baseline.py
```