import os, torch, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

HFGC_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
HGFC_MODEL="meta-llama/Llama-3.1-8b-instruct"
LORA_PATH="finetuned_llama3_adapter"
VALID_DATASET="valid_ans.json"
MAX_SEQ_LENGTH=2048
FP16=True
BF16=False
DEVICE_MAP="auto"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model=AutoModelForCausalLM.from_pretrained(
    HGFC_MODEL,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP,
    token=HFGC_API_TOKEN,
)
tokenizer=AutoTokenizer.from_pretrained(HGFC_MODEL, token=HFGC_API_TOKEN)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

model=PeftModel.from_pretrained(base_model,LORA_PATH,device_map=DEVICE_MAP)
model=model.merge_and_unload()
model.eval()

eval_dataset_raw=load_dataset('json', data_files=VALID_DATASET)
eval_dataset=eval_dataset_raw["train"]

def format_validation_data(example):
    messages=[
        {"role": "system", "content": "You are an AI assistant specialized in answering questions about academic research papers. Provide concise and accurate answers based *only* on the provided context."},
        {"role": "user", "content": f"Context: {example['context']}\nQuestion: {example['question']}"},
    ]
    return {"prompt": tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)}

evaluated_dataset=eval_dataset.map(format_validation_data,batched=False)
generated_ans=[]
reference_ans=[]

for example in tqdm(evaluated_dataset,desc="Generating answers"):
    prompt=example["prompt"]
    reference_ans.append(example["answer"])
    inputs=tokenizer(prompt,return_tensors="pt",padding=True,truncation=True,max_length=MAX_SEQ_LENGTH)
    inputs={k:v.to(model.device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs=model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=0.7,top_p=0.9,pad_token_id=tokenizer.pad_token_id)

    generated_id=outputs[0,inputs["input_ids"].shape[1]:]
    generated_text=tokenizer.decode(generated_id,skip_special_tokens=True)
    generated_ans.append(generated_text)

rouge_eval=evaluate.load("rouge")
results=rouge_eval.compute(predictions=generated_ans,references=reference_ans,use_stemmer=True)
for key,vals in results.items():
    print(f"{key}: {vals.mid.fmeasure:.4f} (precision: {vals.mid.precision:.4f}, recall: {vals.mid.recall:.4f})")

bleu_eval=evaluate.load("sacrebleu")
bleu_results=bleu_eval.compute(predictions=generated_ans,references=[[ans] for ans in reference_ans])
print(f"BLEU: {bleu_results['score']:.4f} (ppl: {bleu_results['ppl']:.4f})")