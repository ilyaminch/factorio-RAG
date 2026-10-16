"""Сбор Factorio Wiki. Установка: pip install requests beautifulsoup4

Запуск: python parser.py
Результат: data/raw/<page_id>.json рядом со скриптом.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


BASE_URL = 'https://wiki.factorio.com'
OUTPUT = Path(__file__).parent / 'data' / 'raw'


def get_json(session, **params):
    time.sleep(0.75)
    response = session.get(
        BASE_URL + '/api.php',
        params={'format': 'json', 'formatversion': 2, **params},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data


def get_pages(session):
    pages = []
    # 0 — статьи, 3000 — руководства Tutorial с описаниями механик.
    for namespace in (0, 3000):
        params = {
            'action': 'query', 'list': 'allpages', 'apnamespace': namespace,
            'apfilterredir': 'nonredirects', 'aplimit': 500,
        }
        while True:
            data = get_json(session, **params)
            for page in data['query']['allpages']:
                title = page['title']
                if re.search(r'/[a-z]{2,3}(?:-[a-z]+)?$', title, re.I):
                    continue
                if title.startswith(('Prototype/', 'Types/', 'Version history/', 'News/', 'Main Page/sandbox')):
                    continue
                pages.append(page)
            if 'continue' not in data:
                break
            params.update(data['continue'])
    return pages


def get_text(element):
    # В рецептах названия ресурсов находятся в ссылках у иконок.
    soup = BeautifulSoup(str(element), 'html.parser')
    for icon in soup.select('.factorio-icon'):
        link = icon.select_one('a[title]')
        image = icon.select_one('img')
        name = link['title'] if link else (image.get('alt', '') if image else '')
        amount = icon.select_one('.factorio-icon-text')
        amount = amount.get_text(strip=True) if amount else ''
        icon.replace_with(f' {amount} {name} ')
    for image in soup.select('img'):
        link = image.find_parent('a')
        name = image.get('alt') or (link.get('title', '') if link else '')
        image.replace_with(f' {name} ')
    return soup.get_text(' ', strip=True)


def parse_article(page):
    soup = BeautifulSoup(page['text'], 'html.parser')
    article = soup.select_one('.mw-parser-output') or soup
    html = str(article)
    infoboxes = [
        {'html': str(box), 'text': get_text(box)}
        for box in article.select('.infobox')
    ]
    for element in article.select('.infobox, .languages-container, .navbox, .toc, #toc, .mw-editsection, script, style'):
        element.decompose()

    description = []
    for element in article.descendants:
        if element.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            break
        if element.name == 'p':
            description.append(element.get_text(' ', strip=True))

    return {
        'page_id': page['pageid'],
        'title': page['title'],
        'url': BASE_URL + '/' + quote(page['title'].replace(' ', '_'), safe='/:()'),
        'revision_id': page['revid'],
        'description': '\n\n'.join(description),
        'infoboxes': infoboxes,
        'text': get_text(article),
        'html': html,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        pages = get_pages(session)
        print(f'Найдено страниц: {len(pages)}')
        saved = skipped = errors = 0
        for index, page in enumerate(pages, 1):
            # ID страницы подходит для имени файла в Windows и не меняется при переименовании статьи.
            path = OUTPUT / f"{page['pageid']}.json"
            if path.exists():
                skipped += 1
                continue
            try:
                data = get_json(session, action='parse', pageid=page['pageid'], prop='text|revid', disableeditsection=1)
                article = parse_article(data['parse'])
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(path)
                saved += 1
                print(f"{index}/{len(pages)}: {page['title']}")
            except (requests.RequestException, ValueError, RuntimeError) as error:
                errors += 1
                print(f"Ошибка: {page['title']}: {error}")
    print(f'Сохранено: {saved}, уже скачано: {skipped}, ошибок: {errors}')
    print(f'Данные сохранены в {OUTPUT}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
