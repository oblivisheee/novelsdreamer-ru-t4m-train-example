import httpx
from bs4 import BeautifulSoup
import os
url = 'https://ranobelib.me/watashi-no-shiawase-na-kekkon-novel/v1/c24'

def receive_list(url, file_num: int):
    if  not os.path.isfile(str(file_num) + '.txt'):
        file_name = str(file_num + 1) + '.txt'
    else:
        file_name = str(file_num) + '.txt'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    lib = httpx.get(url, headers=headers)
    soup = BeautifulSoup(lib.content, 'html.parser')
    divs = soup.find_all('div')
    text = ''
    for div in divs:
        if div.find_next_sibling('div'):
            p_tags = div.find_all('p')
            if p_tags:
                for tag in p_tags:
                    if 'Больше не показывать' not in tag.text:
                        text += tag.text + '\n'  # Changed ' ' to '\n' to make each </p> a new line
    with open(file_name, 'w') as f:
        f.write(text)



receive_list(url, file_num=1)