from transformers import GPT2LMHeadModel, GPT2Tokenizer

# don't touch it please
model_name = 'sberbank-ai/rugpt3medium_based_on_gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
stop_token = '</s>'
######################


def generate_text(prefix, length=90, k=50, p=0.9, temperature=0.89, append_prefix=True):
    '''
    :param prefix: prefix for generating
    :param length: max lenght of sequence
    :param top_k: select top-k predicted words and sample from it
    :param top_p: nucleuos sampling (find minimal subset of words whose probabilities' sum = p)
    :param temperature: softmax temperature
    :param append_prefix: append prefix to generated text or not
    :return: generated string
    '''
    encoded_input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    total_len = length + len(encoded_input[0]) if append_prefix else length
    output = model.generate(
                input_ids=encoded_input,
                max_length=total_len,
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=1,
                do_sample=True,
                num_return_sequences=1,
            ).squeeze()
    
    text = tokenizer.decode(output, clean_up_tokenization_spaces=True)
    text = text[:text.find(stop_token)]
    text = text[len(tokenizer.decode(encoded_input[0], clean_up_tokenization_spaces=True)):]
    
    if append_prefix:
        return prefix + text
    
    return text
    
if __name__ == '__main__':
    prefix = ''
    
    while prefix != 'exit':
        print('Enter prefix>')
        prefix = input()
        print('GPT3>', generate_text(prefix))
