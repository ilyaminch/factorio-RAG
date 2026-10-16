@echo off
setlocal
set "FACTORIO_PROJECT_DIR=%~dp0"
python -I -c "import os, shutil, sys; from pathlib import Path; root = Path(os.environ['FACTORIO_PROJECT_DIR']).resolve(); expected = root / 'data' / 'processed'; target = expected.resolve(); target == expected or sys.exit('Refusing to clear a redirected data folder'); shutil.rmtree(target) if target.exists() else None; target.mkdir(parents=True, exist_ok=True); print('Cleared:', target)"
exit /b %errorlevel%
