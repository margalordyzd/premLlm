from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-3-1b-it-qat-6bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
