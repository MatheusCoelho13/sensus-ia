"""Baixa imagens a partir de uma lista de URLs e salva em pastas para treino/val.

Formato de entrada: um arquivo `urls.txt` com uma URL por linha.
Uso:
    python download_images.py data/urls/urls.txt ../datasets --max 100 --split 0.8
"""
import sys
import os
import argparse
import requests
from pathlib import Path
import mimetypes
import time


DEFAULT_BROWSER_UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'


def build_headers(url, contact=''):
    if contact:
        user_agent = f'assistiva-ia/1.0 (mailto:{contact}); {DEFAULT_BROWSER_UA}'
    else:
        user_agent = DEFAULT_BROWSER_UA

    headers = {
        'User-Agent': user_agent,
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    }

    if 'wikimedia.org' in url:
        headers['Referer'] = 'https://commons.wikimedia.org/'

    return headers


def download(url, dest, headers=None):
    if headers is None:
        headers = build_headers(url)
    try:
        with requests.get(url, headers=headers, timeout=20, stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get('content-type', '')
            if not Path(dest).suffix and content_type:
                ext = mimetypes.guess_extension(content_type.split(';')[0].strip()) or '.jpg'
                dest = str(dest) + ext

            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True, dest, None
    except requests.exceptions.HTTPError as e:
        return False, None, f"HTTP {e.response.status_code}: {e.response.reason}"
    except Exception as e:
        return False, None, str(e)


def download_with_retries(url, dest, headers=None, max_retries=3):
    # If Wikimedia full URL and not thumb, convert to thumb pattern as fallback
    def make_wikimedia_thumb(u):
        if 'upload.wikimedia.org/wikipedia/commons/' in u and '/thumb/' not in u:
            parts = u.split('/')
            filename = parts[-1]
            path_parts = parts[-3:-1]
            # construct thumb url: /wikipedia/commons/thumb/<a>/<b>/<filename>/<width>px-<filename>
            thumb = '/'.join(parts[:-1])
            # parts[-3] and [-2] may be the two-letter path components
            thumb = u.replace('/wikipedia/commons/', '/wikipedia/commons/thumb/')
            thumb = thumb + '/' + '640px-' + filename
            return thumb
        return u

    attempt = 0
    cur_url = url
    while attempt < max_retries:
        if headers is None:
            current_headers = build_headers(cur_url)
        else:
            current_headers = headers
        ok, saved, err = download(cur_url, dest, headers=current_headers)
        if ok:
            return True, saved, None
        # on 429 or HTTP 5xx, retry with backoff; if Wikimedia, try thumb url
        err_lower = (err or '').lower()
        if '429' in err_lower or 'too many requests' in err_lower or 'rate' in err_lower or err is None:
            # try converting to thumb if possible on first retry
            if attempt == 0:
                new_url = make_wikimedia_thumb(url)
                if new_url != url:
                    cur_url = new_url
                    attempt += 1
                    time.sleep(1 * attempt)
                    continue
            # exponential backoff
            attempt += 1
            time.sleep(1 * attempt)
            continue
        else:
            return False, None, err

    return False, None, f"failed after {max_retries} retries: {err}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('urls_file')
    p.add_argument('out_dir')
    p.add_argument('--max', type=int, default=500)
    p.add_argument('--split', type=float, default=0.8)
    p.add_argument('--delay', type=float, default=1.0, help='delay (seconds) between downloads')
    p.add_argument('--wikimedia-delay', type=float, default=2.0, help='min delay for Wikimedia URLs')
    p.add_argument('--contact', type=str, default='', help='contact email to include in User-Agent')
    p.add_argument('--per-class', type=int, default=0, help='target number of images per class (0 = unlimited)')
    args = p.parse_args()

    # prepare headers with contact info if provided
    contact = args.contact.strip()
    if contact:
        user_agent = f"assistiva-ia/1.0 (mailto:{contact})"
    else:
        user_agent = "assistiva-ia/1.0"
    headers = {'User-Agent': user_agent, 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'}

    urls = [l.strip() for l in open(args.urls_file, 'r', encoding='utf-8') if l.strip()]
    urls = urls[:args.max]

    out_dir = Path(args.out_dir)
    train_dir = out_dir / 'images' / 'train'
    val_dir = out_dir / 'images' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    split_index = int(len(urls) * args.split)

    # downloaded URLs log to avoid duplicates
    downloaded_log = out_dir / 'downloaded_urls.txt'
    downloaded_urls = set()
    if downloaded_log.exists():
        with open(downloaded_log, 'r', encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                if l:
                    downloaded_urls.add(l)

    p_per = getattr(args, 'per_class', 0)
    # build existing counts per class (train+val)
    existing_counts = {}
    if p_per:
        for sub in ('train', 'val'):
            base = out_dir / 'images' / sub
            if base.exists():
                for cls_dir in base.iterdir():
                    if cls_dir.is_dir():
                        cnt = sum(1 for _ in cls_dir.iterdir() if _.is_file())
                        existing_counts[cls_dir.name] = existing_counts.get(cls_dir.name, 0) + cnt

    for i, line in enumerate(urls):
        parts = line.split()
        if len(parts) == 1:
            cls = 'common'
            url = parts[0]
        else:
            cls = parts[0]
            url = parts[-1]

        # skip if URL was already downloaded
        if url in downloaded_urls:
            print(f"[{i+1}/{len(urls)}] SKIP already downloaded: {cls} {url}")
            continue

        # skip if class already has enough images
        if p_per:
            have = existing_counts.get(cls, 0)
            if have >= p_per:
                print(f"[{i+1}/{len(urls)}] SKIP class full ({have}/{p_per}): {cls}")
                continue

        which = 'train' if i < split_index else 'val'
        fname = f"img_{i:06d}"
        class_dir = (train_dir if which == 'train' else val_dir) / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / fname
        ok, saved_path, err = download_with_retries(url, dest, headers=headers)
        if not ok:
            print(f"[{i+1}/{len(urls)}] FAIL download: {which} {url} -> {err}")
            continue

        print(f"[{i+1}/{len(urls)}] {which} -> {saved_path} OK")

        # record downloaded URL and increment class count
        try:
            with open(downloaded_log, 'a', encoding='utf-8') as f:
                f.write(url + '\n')
        except Exception:
            pass
        downloaded_urls.add(url)
        if p_per:
            existing_counts[cls] = existing_counts.get(cls, 0) + 1

        # polite delay to avoid rate limits
        delay = args.delay
        if 'wikimedia.org' in url:
            delay = max(delay, args.wikimedia_delay)
        time.sleep(delay)


if __name__ == '__main__':
    main()
