#!/usr/bin/env python3
"""Busca imagens públicas no Wikimedia Commons e gera um arquivo urls (classe url).

Exemplo:
    python fetch_wikimedia_urls.py --out urls_wikimedia.txt --count 50

Padrão de classes (pt -> en): pessoa, cadeira, mesa, porta, parede
"""
import argparse
import requests
import time
from typing import List

SEARCH_MAP = {
    'pessoa': 'person',
    'cadeira': 'chair',
    'mesa': 'table',
    'porta': 'door',
    'parede': 'wall'
}


def fetch_images_for_term(term: str, target: int, session: requests.Session) -> List[str]:
    urls = []
    # try a single request with generous limit
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': term,
        'gsrnamespace': 6,  # file namespace
        'gsrlimit': min(100, target * 2),
        'prop': 'imageinfo',
        'iiprop': 'url',
        'iiurlwidth': 640,  # request thumbnail URL when available
        'format': 'json'
    }

    headers = {'User-Agent': 'assistiva-ia/1.0 (contact: none)'}
    r = session.get('https://commons.wikimedia.org/w/api.php', params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    pages = data.get('query', {}).get('pages', {})
    for pid, page in pages.items():
        iinfo = page.get('imageinfo')
        if not iinfo:
            continue
        info = iinfo[0]
        # prefer thumbnail URL (thumburl) if API returned it
        url = info.get('thumburl') or info.get('url')
        if url:
            urls.append(url)
            if len(urls) >= target:
                break

    # If not enough results, return what we have
    return urls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='urls_wikimedia.txt')
    p.add_argument('--count', type=int, default=50, help='Quantidade por classe')
    p.add_argument('--classes', nargs='*', default=list(SEARCH_MAP.keys()))
    args = p.parse_args()

    session = requests.Session()
    out_lines = []

    for cls in args.classes:
        en = SEARCH_MAP.get(cls, cls)
        print(f'Buscando ~{args.count} imagens para "{cls}" (termo: "{en}")...')
        try:
            urls = fetch_images_for_term(en, args.count, session)
        except Exception as e:
            print('Erro ao buscar:', e)
            urls = []

        print(f'Encontradas {len(urls)} para {cls}')
        for u in urls:
            out_lines.append(f"{cls} {u}")

        # respeitar limites
        time.sleep(1.0)

    with open(args.out, 'w', encoding='utf-8') as f:
        for line in out_lines:
            f.write(line + '\n')

    print(f'Arquivo gerado: {args.out} ({len(out_lines)} linhas)')


if __name__ == '__main__':
    main()
