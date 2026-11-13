# Роль 5 — минимальный интерфейс

Чат Streamlit: сообщения пользователя и ответы идут последовательно, источники показаны под соответствующим ответом. На странице сохраняется весь разговор текущей сессии, а в RAG передаются только последние пять успешных пар (или значение `HISTORY_TURNS`). Ошибки видны в чате, но в контекст RAG не попадают. Постоянного хранения разговоров нет.

## Запуск локально

Из корня репозитория в PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
Set-Location streamlit-app
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Откройте `http://127.0.0.1:8501`. Для остановки нажмите Ctrl+C в терминале приложения.

## Подключение RAG

Чтобы отправить вопрос, должны существовать и быть доступны импорты из согласованного командного контракта:

```text
src.config.load_settings
src.database.retriever.create_retriever
src.rag.generation.create_llm
src.rag.service.answer
```

`answer` принимает `question`, список завершённых `history`, `retriever`, `llm` и `settings`; результат содержит `status`, `answer`, `sources`, `search_query`, `retrieved_chunks` и `error`.

Пока нужных модулей в проекте нет, приложение покажет сообщение об ошибке интеграции вместо ответа.

После передачи модуля Mistral владелец запуска должен добавить его зависимость в этот файл требований или в общий список команды. Зависимости поиска и базы уже перечислены в корневом `requirements.txt`.

## Переменные среды

Настоящая цепочка получает настройки из `src.config.load_settings`. В командном соглашении используются `MISTRAL_API_KEY`, `MISTRAL_MODEL`, `QDRANT_PATH` и связанные RAG-параметры. Задавайте ключ приватно в окружении PowerShell и не записывайте его в репозиторий.

Пример для текущего окна PowerShell; заменить оба маркера собственными значениями:

```powershell
$env:MISTRAL_API_KEY = '<секретный ключ>'
$env:MISTRAL_MODEL = '<доступный model ID>'
```

Отдельный `.env` не загружается автоматически кодом этого интерфейса.
