from bs4 import BeautifulSoup
import requests
import re

def get_news_content(url):
    resp = requests.get(url)
    
    soup = BeautifulSoup(resp.text, 'lxml')
    
    content = ''
    
    
    if re.search(r'daum', url) != None:
        tag = 'div#harmonyContainer p'
    elif re.search(r'yna', url) != None:
        tag = 'div[class=scroller01] p'
    elif re.search(r'joins', url) != None:
        tag = 'div#article_body'
        
    for p in soup.select(tag):
        content += p.get_text()
    return content

def create_news_txt(url):
    content = get_news_content(url)
    
    with open('input.txt', 'w') as f:
        f.write(content)