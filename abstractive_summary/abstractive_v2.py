from transformers import T5ForConditionalGeneration, T5Tokenizer
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





class AbstractiveTopicModeler:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def generate_summary(self, text, max_length=200, num_beams=4):
        """
        Generates a summary for the input text.
        
        :param text: The text to summarize
        :param max_length: The maximum length of the summary
        :param num_beams: The number of beams for beam search
        :return: The generated summary
        """
        # Prepend the prefix "summarize: " to the text as T5 expects it
        input_text = "summarize: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate the summary
        summary_ids = self.model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

if __name__ == "__main__":
    # Example text to summarize
    text = """The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. 
    In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. 
    The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. 
    The authors claimed that within three or five years, machine translation would be a solved problem. 
    However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, 
    funding for machine translation was dramatically reduced. 
    Long after, the field has benefitted from advances in deep learning and big data. 
    The deep learning wave has led to significant improvements in the field by leveraging models like BERT and GPT for various NLP tasks."""
    
    text = formatted_article_text
    # Initialize the modeler
    modeler = AbstractiveTopicModeler('t5-small')
    
    # Generate and print the summary
    summary = modeler.generate_summary(text)
    print("Generated Summary:", summary)
