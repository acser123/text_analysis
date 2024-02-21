# !pip3 install torch torchvision torchaudio
# !pip install transformers
# !pip install sentencepiece

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import bs4 as bs
import urllib.request
import re
import nltk

impfile = ""



import sys, getopt

## From https://stackabuse.com/text-summarization-with-nltk-in-python/
## NLTK install instructions are at https://www.nltk.org/


scraped_data = urllib.request.urlopen(sys.argv[1])
# scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text


# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)


# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)




model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# src_text = [
#    """this happened 5/6 years ago so my whole family every xmas day goes around to my aunties for celebrations. my cousin (of course) was there and he
# asked if i wanted to play cops and robbers. i accepted of course. now, next to the side of my aunts house is a little area with a small fence, a covered
# water tank and super duper sharp stones. my cousin (who was the cop) was gaining on me. i (tried) to jump over the fence, aaand i failed the jump
# and went crashing onto the gravel, my leg hitting the sharpest bit and, then the next thing i knew it had a nasty gash."""

# ]

src_text = formatted_article_text

batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest',return_tensors='pt')
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

print(tgt_text)
