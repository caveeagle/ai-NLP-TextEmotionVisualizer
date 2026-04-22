from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
tokenizer.save_pretrained('./model')

print('Job finished')
