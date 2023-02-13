from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv

#summarization model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokanizer=PegasusTokenizer.from_pretrained(model_name)
model=PegasusForConditionalGeneration.from_pretrained(model_name)
sentiment = pipeline("sentiment-analysis")

#news and sentiment pipeline
# monitored_tickers=input('Inport your monitored tickers: ').split(', ')
monitored_tickers=['ETH']

print(f'Searching news for {monitored_tickers}...')
def news_urls(ticker):
    search_url=f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
    req=requests.get(search_url)
    soup=BeautifulSoup(req.text, 'html.parser')
    atags=soup.find_all('a')
    hrefs=[link['href'] for link in atags]
    return hrefs

raw_urls={ticker:news_urls(ticker) for ticker in monitored_tickers}
print(raw_urls)
print('Done')

#strip out unwanted urls
print('Stripping unwanted urls...')
exclude=['maps', 'policies', 'preferences', 'accounts', 'support']
def strip_urls(urls, exclude):
    val=[]
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude):
            print('True')
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls={ticker:strip_urls(raw_urls[ticker], exclude) for ticker in monitored_tickers}
print(cleaned_urls)
print('Done')

#serch and scrape
print('Scraping news linkes...')
def scrape(urls):
    articles=[]
    for url in urls:
        req=requests.get(url)
        soup=BeautifulSoup(req.text, 'html.parser')
        paragraphs=soup.find_all('p')
        text=[paragraph.text for paragraph in paragraphs]
        article=' '.join(' '.join(text).split(' ')[:350])
        articles.append(article)
    return articles

articles={ticker:scrape(cleaned_urls[ticker]) for ticker in monitored_tickers}
print(articles)
print('Done')

#summarize articles
print('Summarizing articles...')
def summarize_articles(articles):
    summaries=[]
    for article in articles:
        input_ids=tokanizer.encode(article, return_tensors='pt')
        output=model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary=tokanizer.decode(output[0], skip_special_tockens=True)
        summaries.append(summary)
    return summaries

summaries={ticker:summarize_articles(articles[ticker]) for ticker in monitored_tickers}
print(summaries)
print('Done')

#sentiment analysis
print('Calculating sentiments...')
scores={ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
print(scores)
print('Done')

#exporting result
print('Exporting results...')
def output(urls, summaries, scores):
    output=[]
    for ticker in monitored_tickers:
        for i in range(len(summaries[ticker])):
            ticker_output=[
                ticker, 
                summaries[ticker][i],
                scores[ticker][i]['label'],
                scores[ticker][i]['score'],
                urls[ticker][i]]
            output.append(ticker_output)
    return(output)

final_output=output(cleaned_urls, summaries, scores)
final_output.insert(0, ['Ticker', 'Summary', 'Sentiment Score', 'Url'])

with open('financial_summaries.csv', mode='w', newline='') as f:
    csv_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

print('Done')
