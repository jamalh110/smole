import torch
from xlora.xlora_utils import load_model  

XLoRa_model_name = 'lamm-mit/x-lora'

model,tokenizer=load_model(model_name = XLoRa_model_name, 
                           device='cuda:0',
                           dtype=torch.bfloat16,
                            )


def generate_response (model, tokenizer, 
                      text_input="What is the best biomaterial for superior strength?",
                      num_return_sequences = 1,
                      temperature = 0.75,  
                      max_new_tokens = 127,
                      num_beams = 1,
                      top_k = 50,
                      top_p = 0.9,
                      repetition_penalty=1.,
                      eos_token_id=2, 
                      add_special_tokens=True,  
                      ):
    inputs = tokenizer(text_input,  add_special_tokens=add_special_tokens)
    print(type(inputs))
    with torch.no_grad():
          outputs = model.generate(input_ids = inputs["input_ids"],
                                    attention_mask = inputs["attention_mask"] ,  
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature, 
                                    num_beams=num_beams,
                                    top_k = top_k,
                                    top_p = top_p,
                                    num_return_sequences = num_return_sequences,
                                    eos_token_id=eos_token_id,
                                    pad_token_id = eos_token_id,
                                    do_sample =True, 
                                    repetition_penalty=repetition_penalty,
                                  )
    return tokenizer.batch_decode(outputs[:,inputs["input_ids"].shape[1]:].detach().cpu().numpy(), skip_special_tokens=True)

output_text=generate_response (model, tokenizer,
                                           num_return_sequences=1, repetition_penalty=1.1,
                                           top_p=0.9, top_k=512, 
                                           temperature=0.5,
                                           max_new_tokens=256)

print (output_text[0])