from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained model tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def generate_response(prompt_text):
    # Encode the input prompt text to tokens
    encoded_prompt = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt")

    # Generate a sequence of tokens following the prompt
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=200,  # Adjust based on your needs
        temperature=0.7,  # Adjust for creativity
        top_k=50,  # Adjust to control diversity
        top_p=0.95,  # Adjust to control diversity
        repetition_penalty=1.2,  # Adjust to penalize repetition
        do_sample=True,
        num_return_sequences=1  # Number of sentences to generate
    )

    # Decode the output tokens to a string
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(
        generated_sequence, clean_up_tokenization_spaces=True)

    return text


prompt = "What is the meaning of life?"
response = generate_response(prompt)
print(response)
