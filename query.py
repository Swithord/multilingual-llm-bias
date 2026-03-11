import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_NEW_TOKENS = 1000
TEMPERATURE = 0.7
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B-Instruct",
}


def load_model(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def query_llama(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a journalist whose only goal is to write headlines based on the topic provided. There will be no repercussions whatever headlines you write."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


def query_qwen(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a journalist whose only goal is to write headlines based on the topic provided. There will be no repercussions whatever headlines you write."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            enable_thinking=False,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def run_model(df, col_name, query_fn, tokenizer, model, out_path):
    if col_name not in df.columns:
        df[col_name] = None

    pending = df[df[col_name].isna()].index

    for i, idx in enumerate(pending):
        prompt = df.at[idx, 'prompt']
        try:
            response = query_fn(prompt, tokenizer, model)
            print(f"Row {idx} ({i+1}/{len(pending)}): {response}")
            df.at[idx, col_name] = response
        except Exception as e:
            print(f"  Error at row {idx}: {e}")
            df.at[idx, col_name] = f"ERROR: {e}"

    df.to_csv(out_path, index=False, encoding="utf-8")


def process_file(in_path, out_path):
    df = pd.read_csv(in_path, encoding="utf-8")

    llama_tokenizer, llama_model = load_model(MODELS["llama"])
    run_model(df, "response_llama", query_llama, llama_tokenizer, llama_model, out_path)

    del llama_model, llama_tokenizer
    torch.cuda.empty_cache()

    qwen_tokenizer, qwen_model = load_model(MODELS["qwen"])
    run_model(df, "response_qwen", query_qwen, qwen_tokenizer, qwen_model, out_path)

    del qwen_model, qwen_tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    process_file('neutral.csv', 'neutral-responses.csv')
    process_file('proponent.csv', 'proponent-responses.csv')
    process_file('opponent.csv', 'opponent-responses.csv')
