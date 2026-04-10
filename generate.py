from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-finetuned")

input_text = "Success is"

inputs = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=3,
    no_repeat_ngram_size=2,
    do_sample=True,          
    temperature=0.7,         
)

for i, output in enumerate(outputs):
    print(f"\nGenerated {i+1}:")
    print(tokenizer.decode(output, skip_special_tokens=True))