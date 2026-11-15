## Current version 0.7.0

[Amino-acid/GSH recycling and NADPH supply, 4503 states](docs/REDOX_STAGE_0.7.0.md). Explicit opt-in: `--redox-v15` for fresh smoke/pulse runs. Earlier 4425/4431-state scenarios remain available. Conditional prototype: the initial GSH pool is not yet stationary, no development-plan item is closed by this snapshot, and human calibration remains open.

## Previous stage 0.6.0

[Stimulus offsets, recovery and expanded kinetic search](docs/PHASE_STAGE_0.6.0.md). Explicit opt-in: `--acid-tau-s 120 --phase-v14`, with 4431 states. Legacy defaults use 4425 states; prior scenarios remain available. Conditional prototype; human calibration remains open.

## Previous stage 0.5.0

[Waveform and voltage recovery stage](docs/WAVEFORM_STAGE_0.5.0.md). Includes the 0.4.0 data requests and filename migration. Conditional research prototype; human calibration remains open.

# Мотонейрон — пакет для научной проверки

**[Help improve the model: experimental data requests (U01–U43)](DATA_REQUESTS.md)** — share measurements, dataset links or corrections for unresolved parameters.

[codex] 2026-11-13. [English README](README.md).

Это отдельная переносимая выборка проекта. Исходный рабочий проект и его Git-история сохранены. Публичный репозиторий: [osmirnoff02-cloud/motoneuron](https://github.com/osmirnoff02-cloud/motoneuron). Скачать можно через Code → Download ZIP или командой `git clone https://github.com/osmirnoff02-cloud/motoneuron.git`. Это фиксированный проверенный снимок, без более поздних изменений рабочего проекта.

В пакете: полная связанная модель на 4425 переменных для выбранных протоколов, необходимые исходные таблицы, реестр параметров, сохранённые ключевые результаты, проверки, описание ограничений и инструкции. Число переменных включает учётные счётчики и проекции. Это условный исследовательский прототип, а не откалиброванный человеческий мотонейрон.

## Миссия

Миссия «Мотонейрона» — сохранять модель открытой и доступной для исследования БАС и связанных неврологических заболеваний, поддерживать совместную разработку методов их предотвращения и лечения. Достижение этой цели не должно прекращать доступность опубликованных версий.

[Полный текст миссии RU/EN](MISSION.md). Миссия не добавляет ограничений к лицензии.

## Условия использования с версии 0.2.0

**Некоммерческие научные исследования — бесплатно; коммерческое использование — по отдельному письменному договору.** Правообладатели: **Максим Смирнов и Олег Смирнов**. Исследователи могут запускать, изучать, изменять и распространять модель в рамках [лицензии](LICENSE.txt). [Коммерческие запросы](COMMERCIAL_LICENSE.md). Это код с доступными исходниками, но с ограничением коммерческого использования, поэтому не open source в смысле OSI.

**Версия 0.1.2 остаётся AGPL-3.0-only.** Выпуск 0.2.0 менял документы и лицензирование. Код, ранее опубликованный в 0.1.2, по-прежнему доступен в той версии по AGPL, включая коммерческое применение при соблюдении её условий. Последующие оригинальные дополнения имеют явно указанные для них условия. Новые условия не отзывают прежних прав. [Подробности и граница версий](docs/LICENSE_SCOPE.md).

## Запуск в Windows

Установите Python 3.12, откройте PowerShell в этой папке и выполните:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -I -B -X utf8 verify_package.py
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo.py --protocol smoke --check-restart --output outputs/smoke
.\.venv\Scripts\python.exe -I -B -X utf8 tests/check_output.py outputs/smoke
```

Затем пример со ступенью тока 25 пА:

```powershell
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo.py --protocol pulse --profile Burley --candidate C08 --current-pa 25 --output outputs/pulse25
.\.venv\Scripts\python.exe -I -B -X utf8 tests/check_output.py outputs/pulse25
```

Результаты находятся в выбранной папке: PNG-график, CSV временных рядов, JSON с проверками/параметрами и полный checkpoint. Первый запуск может несколько минут компилировать численные функции. Для нового запуска используйте новое имя папки: существующие результаты не перезаписываются и не удаляются. `-I` исключает влияние пользовательского PYTHONPATH.

Smoke проверяет запуск всех состояний на 20 мс и согласованность продолжения до 40 мс. Он не доказывает здоровье или длительную устойчивость нейрона. В электрическом примере ожидаются четыре разряда во время ступени — прежний ориентир модели, не новый эксперимент.

## Что читать учёному

1. [Описание модели](docs/MODEL.md) и [границы пакета](docs/SCOPE.md).
2. [Допущения и требуемые измерения](docs/ASSUMPTIONS.md), [числовой реестр](parameters/README.md).
3. [Проверки именно этой сборки](docs/VALIDATION.md), [выбранные прежние результаты](results/README.md).
4. [Длительные протоколы](docs/RUNNING.md), [данные и воспроизводимость](docs/DATA.md).

Английский текст подготовлен для внешних коллег; часть исходных комментариев и параметрического реестра остаётся по-русски. Статусы реализации, подключения, численной проверки и человеческой калибровки разделены. Не все направления БАС и все исторические анализы включены в исполняемую часть.

Версия снимка — **0.7.0** (2026-11-15), [Motoneuron Noncommercial Research License 1.0](LICENSE.txt); правообладатели — Максим Смирнов и Олег Смирнов. Численная модель и её ограничения сохранены. [Проверка выпуска](PUBLICATION_CHECKLIST.md), [сторонние права и прежняя AGPL](docs/LICENSE_SCOPE.md).


## Acid-removal stage 0.3.0

[New optional 4431-state scenario, results and commands](docs/ACID_STAGE_0.3.0.md). Prior 4425-state demo remains available with acid disabled. Human calibration and waveform limitations remain.
