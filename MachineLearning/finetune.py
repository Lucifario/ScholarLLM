import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv
load_dotenv()

HUGGING_FACEHUB_MODEL="meta-llama/Llama-3.1-8b-instruct"
HGFC_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
DATASET="qca_dataset.json"
OUTPUT_DIR="finetuned_llama3_adapter"

LORA_R=16 
LORA_ALPHA=32
LORA_DROPOUT=0.05
BIAS="none"
TASK_TYPE="CAUSAL_LM"

EPOCHS=5
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
OPTIMIZER="paged_adamw_8bit"
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
LOGGING_STEPS=10
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
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

dataset=load_dataset('json',data_files=DATASET)
model=AutoModelForCausalLM.from_pretrained(
    HUGGING_FACEHUB_MODEL,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP,
    token=HGFC_API_TOKEN,
)
model.config.use_cache=False
model.enable_input_require_grads()

tokenizer=AutoTokenizer.from_pretrained(HUGGING_FACEHUB_MODEL,token=HGFC_API_TOKEN)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

model=prepare_model_for_kbit_training(model)
peft=LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=LORA_DROPOUT,bias=BIAS,task_type=TASK_TYPE,target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"])

model=get_peft_model(model,peft)
model.print_trainable_parameters()

def formatting(eg):
    op=[]
    for i in range(len(eg["question"])):
        question=eg["question"][i]
        context=eg["context"][i]
        answer=eg["answer"][i]

        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in answering questions about academic research papers. Provide concise and accurate answers based *only* on the provided context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ]
        op.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False))

    return {"text": op}

training_arguments=TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    fp16=FP16,
    bf16=BF16,
    max_steps=-1,
    group_by_length=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft,
    formatting_func=formatting,
    args=training_arguments,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)