# Сборка CryptoChat.exe и папки publish\ для передачи другим.
# Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1              обычная сборка (без поиска по смыслу)
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1 -Semantic    с поиском по смыслу (PyTorch + модель)
param([switch]$Semantic)
$ErrorActionPreference = "Stop"

Write-Host "1/6 Установка библиотек..."
python -m pip install --quiet -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    # Для очень новых версий Python у RapidFuzz может ещё не быть готовой сборки.
    # Тогда ставим остальное: бот будет учитывать опечатки встроенным difflib.
    Write-Host "Не все библиотеки установились. Пробую без RapidFuzz..."
    python -m pip install --quiet "pymorphy3>=2.0" "pyinstaller>=6.0"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить библиотеки. Проверьте интернет." }
}
if ($Semantic) {
    python -m pip install --quiet -r requirements-semantic.txt
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить sentence-transformers / PyTorch. Для новых версий Python попробуйте Python 3.13." }
    Write-Host "    Скачиваю модель..."
    python run.py --download-model
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скачать модель." }
}

Write-Host "2/6 Тесты..."
python -m unittest
if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли. Сборка остановлена." }

Write-Host "3/6 Сборка CryptoChat.exe..."
Stop-Process -Name CryptoChat -Force -ErrorAction SilentlyContinue
# --collect-all забирает в .exe словари pymorphy3, без них морфология не загрузится.
$pyiArgs = @("--onefile", "--console", "--name", "CryptoChat", "--clean", "--noconfirm",
          "--collect-all", "pymorphy3", "--collect-all", "pymorphy3_dicts_ru", "--collect-all", "rapidfuzz")
if ($Semantic) {
    # Модуль подключается через importlib, поэтому его нужно указать явно.
    $pyiArgs += @("--hidden-import", "sentence_transformers", "--collect-all", "sentence_transformers",
               "--collect-all", "transformers", "--collect-data", "torch")
} else {
    # Обычная сборка: PyTorch не берём, даже если он установлен — иначе .exe вырастет в разы.
    $pyiArgs += @("--exclude-module", "torch", "--exclude-module", "sentence_transformers", "--exclude-module", "transformers")
}
python -m PyInstaller @pyiArgs run.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой." }

Write-Host "4/6 Папка publish..."
if (Test-Path publish) { Remove-Item publish -Recurse -Force }
New-Item -ItemType Directory publish | Out-Null
Copy-Item dist\CryptoChat.exe publish\
Copy-Item content publish\content -Recurse
if ($Semantic) {
    Copy-Item models publish\models -Recurse
    # В собранной программе модель берётся только из папки models — интернет не нужен.
    python -c "import json,pathlib; p=pathlib.Path('publish/content/config.json'); c=json.loads(p.read_text(encoding='utf-8')); c.setdefault('semantic',{}).update(enabled=True, localFilesOnly=True); p.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')"
} else {
    # Без PyTorch поиск по смыслу недоступен — выключаем его, чтобы не было лишнего сообщения.
    python -c "import json,pathlib; p=pathlib.Path('publish/content/config.json'); c=json.loads(p.read_text(encoding='utf-8')); c.setdefault('semantic',{}).update(enabled=False); p.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')"
}

Write-Host "5/6 Проверка собранной программы..."
& .\publish\CryptoChat.exe --validate
if ($LASTEXITCODE -ne 0) { throw "Собранная программа не прошла проверку базы." }

$done = "6/6 Готово: publish\CryptoChat.exe и publish\content"
if ($Semantic) { $done += " и publish\models" }
Write-Host $done
# Пароль почты в сборку намеренно не копируется: папку publish передают другим людям.
if (Test-Path secrets.json) {
    Write-Host "Отправка журналов на почту: положите secrets.json рядом с CryptoChat.exe вручную, только на своём компьютере."
}
