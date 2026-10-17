"""Очистка статей и создание чанков с метаданными.

Запуск из корня: python -m src.data_pipeline.preprocess.

Очищенные статьи: data/processed/pages/<page_id>.json.
Чанки для последующего вычисления эмбеддингов: data/processed/chunks.jsonl.
"""

import json
import re
from pathlib import Path
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup
from src.data_pipeline.parser import get_text


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
OUTPUT = ROOT / 'data' / 'processed'
CHUNK_SIZE = 1600  # Символы, включая название статьи и раздела, не токены.
OVERLAP = 200
SKIP_SECTIONS = {'history', 'changelog', 'see also', 'references', 'external links',
                 'gallery', 'achievements', 'relevant factorio friday facts'}
SKIP_PAGES = ('Main Page', 'News', 'Roadmap', 'Upcoming features', 'Version history',
              'Prototype', 'Data.raw', 'Data types', 'Modding', 'Tutorial:Mod',
              'Tutorial:Script', 'Tutorial:Inspecting a live mod', 'Tutorial:Localisation')


def clean_text(node):
    soup = BeautifulSoup(str(node), 'html.parser')
    # Соседние иконки без знаков между ними — перечисление, а не формула рецепта.
    for icon in soup.select('.factorio-icon'):
        previous = icon.previous_sibling
        while isinstance(previous, str) and not previous.strip():
            previous = previous.previous_sibling
        if getattr(previous, 'name', None) and 'factorio-icon' in previous.get('class', []):
            icon.insert_before('; ')
    for item in soup.select('li + li'):
        item.insert_before('; ')
    for line_break in soup.select('br'):
        line_break.replace_with('; ')
    value = re.sub(r'\s+', ' ', get_text(soup).replace('\u200b', '').replace('\xad', '')).strip()
    value = re.sub(r'\s+([,.;:!?%)\]])', r'\1', value)
    return re.sub(r'([(\[])\s+', r'\1', value)


def table_rows(table):
    """Повторяем подписи колонок в строках; учитываем объединённые ячейки."""
    headers, spans, result = [], {}, []
    for row in table.find_all('tr'):
        if row.find_parent('table') is not table:
            continue
        cells = row.find_all(['td', 'th'], recursive=False)
        values = {column: value for column, (value, count) in spans.items()}
        spans = {column: (value, count - 1) for column, (value, count) in spans.items() if count > 1}
        column = 0
        for cell in cells:
            while column in values:
                column += 1
            value = clean_text(cell)
            for _ in range(int(cell.get('colspan', 1))):
                values[column] = value
                if int(cell.get('rowspan', 1)) > 1:
                    spans[column] = (value, int(cell['rowspan']) - 1)
                column += 1
        if not values:
            continue
        values = [values.get(i, '') for i in range(max(values) + 1)]
        if cells and all(cell.name == 'th' for cell in cells):
            headers = values
        elif headers and len(headers) == len(values):
            result.append('; '.join(f'{label}: {value}' if label else value
                                    for label, value in zip(headers, values) if value))
        else:
            result.append(' | '.join(value for value in values if value))
    # Таблицы только с заголовками тоже могут содержать полезные значения.
    if not result and headers:
        return [' | '.join(headers)]
    return result


def blocks(node):
    """Обходим обёртки, сохраняя абзацы, списки и таблицы целиком."""
    for child in node.children:
        if child.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table',
                          'ul', 'ol', 'dl', 'pre', 'blockquote', 'figure'):
            yield child
        elif child.name:
            yield from blocks(child)


def clean_page(raw):
    """Очистить статью и разделить характеристики и основной текст на разделы.

    Args:
        raw: Исходная запись из data/raw с HTML и данными страницы.

    Returns:
        Словарь с метаданными и очищенными разделами.
    """
    soup = BeautifulSoup(raw['html'], 'html.parser')
    for node in soup.select('.languages-container, .navbox, .toc, #toc, .mw-editsection, '
                            '.reference, .mw-references-wrap, .reflist, .landmark, script, style'):
        node.decompose()
    # Для формул оставляем исходную запись вместо дублирующего скрытого MathML.
    for formula in soup.select('.mwe-math-element'):
        image = formula.select_one('img[alt]')
        if image:
            formula.replace_with(image['alt'])
    for color in soup.select('.template-color'):
        value = re.search(r'background-color:\s*([^;]+)', color.get('style', ''))
        if value:
            color.replace_with(value.group(1))

    sections = []
    for box in soup.select('.infobox'):
        tabs = [clean_text(tab) for tab in box.select('.tabber li')]
        for index, table in enumerate(box.find_all('table', recursive=False)):
            variant = tabs[index] if index < len(tabs) else None
            lines, label = [], ''
            for row in table.find_all('tr'):
                if row.find_parent('table') is not table:
                    continue
                cells = row.find_all(['td', 'th'], recursive=False)
                if len(cells) > 1:
                    lines.append(clean_text(cells[0]) + ': ' + ' | '.join(clean_text(c) for c in cells[1:]))
                elif cells and 'infobox-vrow-value' in cells[0].get('class', []):
                    lines.append(label + ': ' + clean_text(cells[0]))
                elif cells:
                    label = clean_text(cells[0])
            if lines:
                sections.append({'heading': 'Characteristics' + (f' / {variant}' if variant else ''),
                                 'anchor': None, 'game_variant': variant, 'blocks': lines})
        box.decompose()

    path = []
    section = {'heading': 'Introduction', 'anchor': None, 'game_variant': None, 'blocks': []}
    for node in blocks(soup):
        if re.fullmatch(r'h[1-6]', node.name):
            if section['blocks']:
                sections.append(section)
            level = int(node.name[1])
            path = [(depth, name) for depth, name in path if depth < level]
            path.append((level, clean_text(node)))
            anchor_node = node if node.get('id') else node.find(id=True)
            section = {'heading': ' / '.join(name for _, name in path),
                       'anchor': anchor_node.get('id') if anchor_node else None,
                       'game_variant': None, 'blocks': []}
        elif not any(name.casefold() in SKIP_SECTIONS for _, name in path):
            if node.name == 'table':
                section['blocks'].extend(line for line in table_rows(node) if line)
            elif node.name == 'pre':
                section['blocks'].append(node.get_text().strip())
            elif node.name in ('ul', 'ol'):
                section['blocks'].extend('- ' + clean_text(li) for li in node.find_all('li', recursive=False))
            else:
                value = clean_text(node)
                if value:
                    section['blocks'].append(value)
    if section['blocks']:
        sections.append(section)

    metadata = {key: raw[key] for key in ('page_id', 'title', 'url', 'revision_id')}
    metadata.update(language='en', source_file=f"data/raw/{raw['page_id']}.json")
    return {'metadata': metadata, 'sections': sections}


def split_text(text, size, overlap=OVERLAP):
    """Режем по абзацам, затем предложениям/словам; соседние части пересекаются."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for separator in ('\n\n', '. ', '; ', ' '):
                boundary = text.rfind(separator, start + overlap + 1, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break
        value = text[start:end].strip()
        if value:
            yield value
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Начинаем на границе слова, не отрезая его первую половину.
        while next_start < end and next_start > 0 and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start


def make_chunks(document):
    """Сформировать чанки одного документа с источниками и стабильными UUID.

    Args:
        document: Очищенная статья с метаданными и разделами.

    Yields:
        Словари с id, text и metadata для эмбеддингов.
    """
    metadata = document['metadata']
    for section_index, section in enumerate(document['sections']):
        prefix = f"{metadata['title']}\n{section['heading']}\n\n"
        size = CHUNK_SIZE - len(prefix)
        if size <= OVERLAP:
            raise ValueError('Слишком длинный заголовок для выбранного CHUNK_SIZE')
        for chunk_index, body in enumerate(split_text('\n\n'.join(section['blocks']), size)):
            text = prefix + body
            chunk_metadata = {**metadata, 'section': section['heading'],
                              'section_index': section_index, 'chunk_index': chunk_index,
                              'game_variant': section['game_variant'],
                              'section_url': metadata['url'] + ('#' + quote(section['anchor']) if section['anchor'] else '')}
            chunk_id = str(uuid5(NAMESPACE_URL, f"{metadata['url']}:{section_index}:{chunk_index}:{text}"))
            yield {'id': chunk_id, 'text': text, 'metadata': chunk_metadata}


def main():
    files = sorted(RAW.glob('*.json'))
    if not files:
        raise SystemExit(f'Нет исходных JSON в {RAW}')
    pages_dir = OUTPUT / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages_count = chunks_count = skipped = 0
    temporary = OUTPUT / 'chunks.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as output:
        for index, file in enumerate(files, 1):
            raw = json.loads(file.read_text(encoding='utf-8'))
            if raw['title'].startswith(SKIP_PAGES) or ' API' in raw['title']:
                skipped += 1
                continue
            document = clean_page(raw)
            if not document['sections']:
                skipped += 1
                continue
            (pages_dir / file.name).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
            for chunk in make_chunks(document):
                output.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                chunks_count += 1
            pages_count += 1
            if index % 100 == 0:
                print(f'Обработано исходных файлов: {index}/{len(files)}', flush=True)
    temporary.replace(OUTPUT / 'chunks.jsonl')
    print(f'Статей: {pages_count}, чанков: {chunks_count}, пропущено служебных/пустых страниц: {skipped}')
    print(f'Результат: {OUTPUT}')


if __name__ == '__main__':
    main()
