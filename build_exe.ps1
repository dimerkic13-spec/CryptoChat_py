# Сборка CryptoChat.exe и папки publish\ для передачи другим.
# Запуск из корня проекта:  powershell -ExecutionPolicy Bypass -File build_exe.ps1
$ErrorActionPreference = "Stop"

Write-Host "1/5 Установка библиотек (pymorphy3, PyInstaller)..."
python -m pip install --quiet -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить библиотеки. Проверьте интернет." }

Write-Host "2/5 Тесты..."
python -m unittest
if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли. Сборка остановлена." }

Write-Host "3/5 Сборка CryptoChat.exe..."
Stop-Process -Name CryptoChat -Force -ErrorAction SilentlyContinue
# --collect-all забирает в .exe словари pymorphy3, без них морфология не загрузится.
python -m PyInstaller --onefile --console --name CryptoChat --clean --noconfirm `
    --collect-all pymorphy3 --collect-all pymorphy3_dicts_ru run.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой." }

Write-Host "4/5 Папка publish..."
if (Test-Path publish) { Remove-Item publish -Recurse -Force }
New-Item -ItemType Directory publish | Out-Null
Copy-Item dist\CryptoChat.exe publish\
Copy-Item content publish\content -Recurse

Write-Host "5/5 Проверка собранной программы..."
& .\publish\CryptoChat.exe --validate
if ($LASTEXITCODE -ne 0) { throw "Собранная программа не прошла проверку базы." }

Write-Host "Готово: publish\CryptoChat.exe и publish\content"
# Пароль почты в сборку намеренно не копируется: папку publish передают другим людям.
if (Test-Path secrets.json) {
    Write-Host "Отправка журналов на почту: положите secrets.json рядом с CryptoChat.exe вручную, только на своём компьютере."
}
