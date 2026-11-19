# n8n-воркфлоу: видео → 50 траекторий → самокоррекция → деплой на робота

![Схема воркфлоу](workflow_screenshot.png)

*(рендер схемы `workflow.json` — после импорта канвас в n8n будет выглядеть так же)*

Оркестрация пайплайна обучения: [pose_to_trajectories.py](pose_to_trajectories.py)
(видео → скелет → 50 траекторий + метрики) → батчевая симуляция в MuJoCo → проверка
порогов из `config.yaml` → дообучение (`/finetune`) с ретраями → деплой на робота через
ROS2-мост → Telegram-уведомление + журнал итераций в Postgres.

## Состав

| Файл | Что это |
|---|---|
| `workflow.json` | импортируемый воркфлоу (n8n 1.x, 22 ноды + 3 sticky-заметки) |
| `mock_worker.py` | локальный воркер для теста без MuJoCo/ROS2: реальный `pose_to_trajectories.py` + псевдо-физика + «дообучение», сходящееся по итерациям |
| `config.yaml` | единый источник порогов/лимитов (`thresholds`, `robot`, `augmentation`, `training`) |
| `build_workflow.py` | генератор `workflow.json` (правки нод → перегенерация) |
| `verify_workflow.py` | оффлайн-верификация: структура JSON + симуляция всего датафлоу (5 сценариев A–E) |
| `workflow_screenshot.png` | схема (для README) |

## Импорт (1 команда)

```bash
n8n import:workflow --input=workflow.json
```

В docker:

```bash
docker cp workflow.json <n8n-контейнер>:/tmp/workflow.json
docker exec <n8n-контейнер> n8n import:workflow --input=/tmp/workflow.json
docker restart <n8n-контейнер>   # подхватить импортированное
```

После импорта в UI (воркфлоу **не активен**):
1. назначьте креды: **Postgres** (нод `Log Iteration`) и **Telegram Bot API** (нод `Notify`);
2. Settings → **Error Workflow** → выберите этот же воркфлоу (включит ветку `On Workflow Error`);
3. Activate → webhook станет доступен на `POST /webhook/train`.

## Переменные окружения (n8n)

| Переменная | Значение | Куда смотрит |
|---|---|---|
| `TRAINING_WORKER_URL` | `http://<host>:8000` | `/config`, `/process_video`, `/finetune` |
| `MUJOCO_WORKER_URL` | `http://<host>:8000` | `/simulate` |
| `ROS2_BRIDGE_URL` | `http://<ros2-bridge>:9000` | `/deploy` |
| `TELEGRAM_CHAT_ID` | id чата/канала | нод `Notify (Telegram)` |

* Самостоятельный n8n: `$env` доступен по умолчанию. Если задан `N8N_BLOCK_ENV_ACCESS_IN_NODE=true` — выключите его.
* Для mock-воркера обе URL совпадают (`/config`, `/process_video`, `/finetune`, `/simulate` — один сервис).

## Быстрый старт (без MuJoCo и ROS2)

```bash
# 1. Журнал итераций
psql -c "$(sed -n '/CREATE TABLE/,/;/p' README.md | head -20)"   # или скопируйте SQL ниже

# 2. Mock-воркер (порт 8000; читает config.yaml)
python mock_worker.py

# 3. Импорт, креды, Activate — и запуск:
curl -X POST http://localhost:5678/webhook/train \
  -H "Content-Type: application/json" \
  -d '{"video_url": "demo://", "params": {"seed": 7}}'
```

`demo://` — синтетический скелет (не нужен ни файл, ни mediapipe). Реальное видео:
`{"video_url": "https://.../input.mp4"}` (воркер скачает и прогонит MediaPipe) или
`{"video_base64": "..."}`. Webhook отвечает сразу `{"message": "Workflow was started"}`;
итог уходит в Telegram и Postgres.

## Как работает цикл самокоррекции

**В n8n нет ни одной Code node** — вся логика (job_id, пороги, агрегация, тексты
уведомлений) выполняется в воркере по HTTP. Fan-out/fan-in, недоступные HTTP-нодам,
делают стандартные **Split Out** и **Aggregate**.

```
Webhook POST /train (video_url|video_base64 + params)
 → GET /init_job: job_id (UUID v4) генерирует воркер
 → GET /config: config.yaml воркера (пороги/лимиты/аугментация — НЕ в коде воркфлоу)
 → POST /process_video: pose_to_trajectories.py → 50 траекторий + готовые episodes
 → Split Out (episodes): 1 item → 50 эпизодов, episode_id = UUID5(job_id:iteration:trajectory)
   — идемпотентен при ретраях
 → Split In Batches (10): батчи по 10 эпизодов
     loop → POST /simulate (MuJoCo) → POST /check_episode (воркер: пороги из config,
            NaN → failed_retryable) → Retry episode? (IF) → ретрай ТОГО ЖЕ episode_id
            (до training.max_episode_retries) → обратно в цикл
     done → Aggregate (50 items → data) → POST /aggregate_iteration: агрегат итерации
            + готовый finetune_request
 → IF «Passed or max iterations?» (all_passed OR iteration ≥ training.max_iterations)
     false → POST /finetune (ответ содержит episodes следующей итерации)
             → Split Out → новая итерация цикла
     true  → IF «Actually passed?»
             true → POST /deploy (ROS2-мост) → POST /build_deploy_result
             false → POST /build_fail_result
 → POST /log_iteration: append в logs/training_iterations.jsonl у воркера
   (журнал стоит ДО уведомления: Telegram затирает item своим ответом)
 → Notify (Telegram): «Итерация 3: ошибка 4.2°, столкновений 0. Политика развёрнута»
   — нода disabled до назначения кред telegramApi (включить в UI)
```

### Почему цикл перезапускается: reset-трюк

Split In Batches v3 не стартует новый цикл по новым данным — node context
персистентен в рамках исполнения, и после `done` новые items лишь дописываются в
выход `done` (проверено по исходникам n8n). Решение в `workflow.json`: параметр
`options.reset` задан **выражением** `={{ $json.new_iteration === true }}`:

* новая итерация (после `/finetune`, у эпизодов `new_iteration: true`) → `reset=true` → свежий цикл;
* возврат батча из тела цикла (`new_iteration: false`) → `reset=false` → обычное продолжение.

### Ошибки и NaN

* HTTP-нод `/simulate` настроена `onError: continueRegularOutput`: падение воркера или
  NaN в метриках не роняет воркфлоу — эпизод помечается `failed_retryable` и
  автоматически перезапускается **с тем же episode_id** (воркер может дедуплицировать),
  попытки видны в журнале (`attempt_no`). Проверка: `MUJOCO_NAN_RATE=0.35 python mock_worker.py`.
* Падение воркфлоу целиком ловит ветка **On Workflow Error** (Error Trigger →
  Format Error → Telegram + Postgres). Сработает после п.2 раздела «Импорт».

## Контракты API

### `GET /config` (training worker)
Ответ: `config.yaml` целиком как JSON. Ноды воркфлоу читают пороги только отсюда.

### `POST /process_video` (training worker)
```json
{"job_id": "...", "iteration": 1, "video_url": "demo://|https://...", "video_base64": null,
 "params": {"seed": 7, "n_trajectories": 50, "side": "auto"}}
```
Ответ: `{"job_id", "iteration", "config_used": {...}, "fps", "reference": {...},
"trajectories": [{"trajectory_id", "n_frames", "timestamps_s", "angles_deg", "augmentation"}],
"pipeline_metrics": {...}}` — 50 траекторий углов (градусы, J1–J6 по timestep).

### `POST /simulate` (MuJoCo worker)
```json
{"episode": {"job_id", "iteration", "trajectory_id", "episode_id", "attempt_no",
             "angles_deg", "thresholds", "training_cfg"}}
```
Ответ: `{"episode": {...эхо...}, "sim_metrics": {"pose_error_deg", "collisions",
"jerk_total", "jerk_mean", "frac_within_limits", "max_limit_violation_deg", "steps"}, "ok": bool}`.
Ошибки → HTTP 5xx `{"error": {"message": "..."}}` (воркфлоу перезапустит эпизод).

### `POST /finetune` (training worker)
```json
{"job_id", "iteration", "failed_episode_ids": [...], "failures": [...],
 "good_trajectories": [{"trajectory_id", "angles_deg"}]}
```
Ответ: `{"job_id", "iteration", "model_version", "config_used", "trajectories": [50 новых]}`.

### `POST /deploy` (ROS2-мост)
```json
{"job_id", "iteration", "summary": {"mean_pose_error_deg", "total_collisions",
 "n_passed", "n_failed"}, "trajectories_count": 50, "thresholds": {...}}
```
Ответ: `{"deploy_id", "status": "deployed", "robot", "summary": {...эхо...}}`.

### Эндпоинты, заменившие Python Code nodes (v3)

| Эндпоинт | Вход | Выход | Заменяет |
|---|---|---|---|
| `GET /init_job` | — | `{job_id, iteration: 1, started_at}` | Init Job (Python) |
| `POST /check_episode` | item ответа `/simulate` (`{episode, sim_metrics}` или `{error}`) | episode-поля top-level + `{status: passed\|failed\|failed_retryable, attempt_no, reasons[]}` — супerset `{passed, retry, reason}`: `passed = status=="passed"`, `retry = status=="failed_retryable"`, `reason = reasons[0]` | Check Episode (Python) |
| `POST /aggregate_iteration` | `{items: [50 результатов]}` | агрегат итерации + `finetune_request` | Aggregate Iteration + Select Failures (Python) |
| `POST /build_deploy_result` | ответ `/deploy` | `{notification_text, log_row}` | Build Deploy Result (Python) |
| `POST /build_fail_result` | агрегат итерации | `{notification_text, log_row}` | Build Fail Result (Python) |
| `POST /format_error` | item Error Trigger | `{notification_text, log_row}` | Format Error (Python) |
| `POST /log_iteration` | `{notification_text, log_row{details_hex}}` | `{ok, status, file}` — append в `logs/training_iterations.jsonl` (1 строка = 1 запись, `details_hex` декодируется в `details`) | Log Iteration (Postgres) → JSONL |

Кроме того, `/process_video` и `/finetune` теперь возвращают готовый массив
`episodes` (бывшие Build Episodes / Prepare Next Iteration) — n8n разворачивает его
нодой Split Out.

## Журнал — JSONL у воркера (без Postgres)

Нода `Log Iteration (HTTP)` шлёт `POST /log_iteration`; воркер дописывает строку в
`logs/training_iterations.jsonl` (папка создаётся автоматически; путь переопределяется
`TRAINING_LOG_PATH`). Формат строки:

```json
{"logged_at": "…", "notification_text": "Итерация 1: ошибка 4.52°, …",
 "job_id": "…", "iteration": 1, "status": "deployed",
 "mean_pose_error_deg": 4.52, "total_collisions": 0,
 "n_passed": 50, "n_failed": 0, "all_passed": true,
 "details": {"deploy_response": {…}, "completed_at": "…"}}
```

`details` — полный контекст (ответ деплоя / агрегат итерации / событие ошибки),
в ноде передаётся как `details_hex` (hex без запятых — безопасно для любых транспортов).
Порядок хвоста: **журнал до уведомления** — Telegram-нода затирает item своим ответом,
поэтому `log_row` уходит в JSONL раньше `Notify`. Сама нода `Notify (Telegram)` помечена
`disabled: true`: пока не назначены креды `telegramApi`, она пропускает item насквозь
и демо проходит без ошибок; после настройки — включить в UI. `Log Iteration` тоже
`onError: continueRegularOutput` (падение воркера не роняет цикл).

Pinned-данные для демо: в `workflow.json` запинен вход webhook'а (`video_url: "demo://"`,
`params`) — при открытии воркфлоу в редакторе входные данные видны без запуска.
В production-запусках через активный webhook pin'ы игнорируются n8n.

Группировка нод по фазам: в JSON-схеме n8n визуальных групп нет (только sticky) —
фазы помечены sticky-заметками: init → process → simulate → aggregate → deploy/notify.

## Пороги — только в config.yaml

```yaml
thresholds:
  pose_error_max_deg: 8.0      # IF-проверка эпизодов и валидность
  max_collisions: 0
training:
  max_iterations: 5            # максимум итераций finetune → simulate
  max_episode_retries: 2       # ретраи эпизода при NaN/сбое MuJoCo
```

Изменили `config.yaml` воркера — воркфлоу подхватит на следующем запуске (он читает
`GET /config` в начале каждого исполнения). Ничего править в нодах не нужно.

## Логика — в воркере: почему без Code nodes

Нативный python task runner n8n работает **только на Linux** (изоляция задач через
`multiprocessing` forkserver, которого не существует на Windows), а npm-установка
n8n под Windows вообще не содержит пакет раннера. Поэтому в v3:

- **В n8n ноль Code nodes** — только HTTP-ноды, Split Out / Aggregate, IF, Split In Batches;
  работает на любом n8n (Windows/npm, docker, cloud).
- **Вся логика — в воркере** (`/init_job`, `/check_episode`, `/aggregate_iteration`,
  `/build_deploy_result`, `/build_fail_result`, `/format_error`, плюс `episodes` в
  ответах `/process_video` и `/finetune`). Логика перенесена из Code-нод 1:1 по смыслу.
- Контекст передаётся через item payload и `$('Node')`-ссылки в выражениях
  (`$env` доступен в выражениях HTTP-нод).
- Fan-out (1 item → 50 эпизодов) и fan-in (50 → 1) делают стандартные ноды
  **Split Out** и **Aggregate** — HTTP-нода работает per-item и не умеет ни того, ни другого.

## Верификация без n8n

```bash
python build_workflow.py      # перегенерация workflow.json
python verify_workflow.py     # структура + симуляция датафлоу на реальном mock_worker
```

Сценарии: **A** happy path (деплой с 1-й итерации), **B** строгий порог → finetune-цикл
(сходимость на 2-й итерации), **C** NaN-ретраи эпизодов, **D** исчерпание
`max_iterations` → ветка failed, **E** ветка Error Trigger. Плюс совместимость обоих
Python-рантаймов. Симулятор воспроизводит семантику Split In Batches v3 по исходникам n8n.

## Проверка живого прогона (2026-11-15, локальный n8n 2.44.5 на Windows)

Реальный прогон на инстансе `http://localhost:5678` (npm-установка, БД `~/.n8n/database.sqlite`,
n8n под watchdog-службой `%LOCALAPPDATA%\n8n-service\start-n8n.cmd`).

### История: почему v3 без Python

Первая живая проверка (v2, 9 Python Code nodes) упала на первой же Code-ноде:
`Python runner unavailable: Python 3 is missing from this system`. Цепочка доказательств:
npm-установка n8n не содержит `@n8n/task-runner-python`; пакет из монорепо (tag `n8n@2.44.5`)
+ venv на 3.13.17 падает с `ImportError: ForkServerProcess` — `forkserver` POSIX-only,
python task runner n8n **архитектурно Linux-only**. Решение — v3: логика перенесена в
воркер, в n8n ноль Code nodes (см. «Логика — в воркере»). Живой прогон также поймал баг,
который не заметила оффлайн-симуляция: `POST /process_video` терял `job_id` после
`GET /config` (`$json` — это ответ конфига); исправлено на `$('Init Job (HTTP)')`-ссылки.

### Живой прогон v3 (workflow id `iz0s1xLoXepB1l0Y`) — полный цикл ✅

| Шаг | Проверка | Результат |
|---|---|---|
| Импорт | `n8n import:workflow --input=workflow.json` | `Successfully imported 1 workflow` |
| Активация | `n8n update:workflow --id=<id> --active=true` + рестарт службы | active, webhook зарегистрирован |
| Mock-воркер | `venv/Scripts/python.exe mock_worker.py` → `curl /init_job` | 200, `job_id` в ответе |
| Webhook | `POST /webhook/train` `{"video_url":"demo://"}` | **200**, исполнение 456 |
| Полный цикл | исполнение 456, все ноды в `runData` | **21/21 нод отработали**: `/init_job` → `/config` → `/process_video` → Split Out (50 эпизодов) → 5 батчей × (`/simulate` ×10 + `/check_episode` ×10) → Aggregate → `/aggregate_iteration` → IF (all_passed=true) → `/deploy` → `/build_deploy_result` |

Ценз запросов воркера за прогон (из лога): `50 × POST /simulate`, `50 × POST /check_episode`,
по одному `/init_job` (GET), `/config`, `/process_video`, `/aggregate_iteration`, `/deploy`,
`/build_deploy_result`. Метрики итерации 1: **50/50 passed, pose_error 4.51°, 0 коллизий**.
Текст уведомления собран воркером: «Итерация 1: ошибка 4.51°, столкновений 0. Политика
развёрнута на mock-robot-1 (deploy_id 43420bc2-…)».

Единственная ошибка исполнения — `Log Iteration (Postgres): Node does not have any
credentials set` (креды Postgres/Telegram не назначены; обе ноды помечены
`onError: continueRegularOutput`, поэтому **деплой и уведомление не блокируются**).
`job_id` не приходит в HTTP-ответе webhook'а — ответ асинхронный по дизайну
(`responseMode: onReceived`); `job_id` виден в данных исполнения и в логах воркера.

### Env-переменные в watchdog-скрипте — важная деталь

Env задаётся в `%LOCALAPPDATA%\n8n-service\start-n8n.cmd` **внутри цикла `:loop`**:
cmd перечитывает строки цикла при каждом рестарте n8n, поэтому правки env подхватываются
простым kill процесса n8n — сам watchdog перезапускать не нужно. Если ставить `set`
до `:loop`, они выполняются один раз при старте watchdog (логон Windows) и новые
переменные не подхватятся, пока не перезапущен сам watchdog.

### Команды воспроизведения

```bash
n8n import:workflow --input=workflow.json          # импорт (или через API: POST /api/v1/workflows)
n8n update:workflow --id=<id> --active=true        # активация (2.x: флаг --active; команда deprecated)
python mock_worker.py                              # воркер :8000 (в venv)
curl http://localhost:8000/init_job                # → 200 {job_id, iteration, started_at}
curl -X POST http://localhost:5678/webhook/train \
  -H "Content-Type: application/json" -d '{"video_url":"demo://"}'   # → 200

# разбор исполнения (n8n 2.x не имеет export:execution; sqlite, read-only; в 2.44
# данные лежат в компактном формате — массив со строковыми ссылками-индексами):
python - <<'PY'
import sqlite3, os
con = sqlite3.connect('file:'+os.path.expanduser('~/.n8n/database.sqlite')+'?mode=ro', uri=True)
print(con.execute("SELECT id,status FROM execution_entity ORDER BY id DESC LIMIT 5").fetchall())
PY
```

### Статус Telegram/Postgres

- **Postgres**: убран из воркфлоу — журнал идёт в JSONL у воркера (`/log_iteration`).
- **Telegram**: креды `telegramApi` не назначены — нода `Notify` помечена `disabled: true`
  (пропускает item насквозь, демо без ошибок); после настройки кред включить в UI.
- Все пороги/лимиты — в `config.yaml` воркера; в коде нод их нет.

### Вердикт

**workflow работает на Windows-n8n, полный цикл без ошибок** ✅: импорт, активация,
webhook, 50 эпизодов → simulate/check → агрегация → деплой → JSONL-журнал → уведомление
(disabled) — без единой Python-ноды и без ошибок нод в исполнении (статус `success`).
Для переносимости на Linux/docker ничего менять не нужно: тот же workflow.json и тот же воркер.

### Живой прогон v4 (id `Idigy3f9JdBGkF02`) — полный цикл без ошибок ✅

Исполнение 460: статус **success** за ~20 секунд, 17/17 нод (с учётом кратностей ×5/×6)
без единой ошибки нод. Ценз запросов воркера: `50 × /simulate`, `50 × /check_episode`,
по одному `/init_job`, `/config`, `/process_video`, `/aggregate_iteration`, `/deploy`,
`/build_deploy_result`, `/log_iteration`. JSONL: 1 строка
(`status=deployed`, `job_id=9bbb7ff7…`, `mean_pose_error_deg=4.52`, `total_collisions=0`,
`deploy_id=b45dd881…`). Pinned-данные webhook'а видны в редакторе при открытии воркфлоу.

⚠️ Урок из прогона: после изменения `mock_worker.py` перезапускайте воркер —
первый v4-запрос попал на старый процесс без `/log_iteration` и 404 был проглочен
`onError: continueRegularOutput` (исполнение «успешно», но журнала нет).

### Живой прогон с реальным ML-ядром (MuJoCo + CMA-ES, exec 463) ✅

Воркер перезапущен с ML-ядром (mujoco 4.0.1 + cma 4.5.0) и строгим порогом
`pose_error_max_deg: 2.0` (через `TRAINING_CONFIG`), webhook → полный цикл:

```
итерация 1: all_passed=false  pe=2.713°  passed=1/50   collisions=0
   → POST /finetune (CMA-ES на 72-dim Фурье-политике, best cost 3.727)
итерация 2: all_passed=true   pe=1.720°  passed=50/50  collisions=0
   → POST /deploy → deployed (deploy_id ee0b7bfe…)
```

Статус исполнения — **success**, ошибок нод нет. `pose_error` между итерациями
снизился на 37% — реальная оптимизация по реальной физике (компенсация
гравитационного провиса константами политики + сглаживание траектории базисом).

⚠️ Урок: первый строгий прогон упал на транзиентном `ECONNREFUSED` (localhost
при интенсивном прогоне отказал в accept'е). Фикс на двух уровнях: нодам
`/simulate` и `/check_episode` добавлен `retryOnFail` (3 попытки, 1 с), воркеру —
`request_queue_size = 64`. Обновление workflow выполнено **по месту через API**
(`PUT /api/v1/workflows/{id}`) — без создания новой копии.





## ML-ядро — реальное (MuJoCo + CMA-ES)

Псевдо-физика и «затухающий шум» заменены на настоящие компоненты (контракты API
не изменились — workflow.json тот же):

- **`/simulate` — реальный MuJoCo**: модель [arm_6dof.xml](arm_6dof.xml) (MJCF,
  кинематика 1:1 с `RobotChain` — FK-сверка до 1e-9 в тестах). Траектория
  прогоняется через позиционные сервоприводы (120 Гц, 4 подшага на кадр 30 fps):
  `pose_error` = ошибка слежения под гравитацией, `collisions` = проникающие
  контакты (стойка/самоколлизии), `jerk` = по фактическим углам суставов.
  Детерминировано: одинаковый вход → одинаковые метрики.
- **`/finetune` — реальный CMA-ES**: политика = траектория в усечённом
  Фурье-базисе (72-dim = 6 суставов × 12 мод), cost = MuJoCo-симуляция
  (ошибка слежения + штраф за рывок + коллизии). c0 = проекция текущего базиса
  (выход гарантированно не хуже входа), тёплый старт с прошлой итерации (job_id).
  Живой прогон со строгим порогом: итерация 1 — 2.71° (1/50 passed) → finetune →
  итерация 2 — 1.72° (50/50 passed) → деплой.

### Ограничения ML-ядра

- policy-состояние (коэффициенты Фурье) живёт **в памяти воркера** под job_id —
  рестарт воркера сбрасывает политику (итерации нового job начнут с проекции базиса).
- **`/deploy` — заглушка** (робота/ROS2 нет): возвращает deploy_id и эхо summary.
  Честно помечено; замена на ROS2-мост не затрагивает остальные ноды.
- `MUJOCO_NAN_RATE` — test-harness для ретраев n8n (инжекция NaN в метрику),
  не физика; по умолчанию 0.
- Модель упрощена: капсульные звенья без моторов/датчиков, стойка = цилиндр;
  для конкретного железа заменить MJCF на модель реального робота
  (например, из MuJoCo Menagerie) — `/simulate` не изменится.

## Замена оставшихся заглушек

| Компонент | Статус | Замена при необходимости |
|---|---|---|
| `/simulate` | ✅ реальный MuJoCo (`arm_6dof.xml`) | модель реального робота — тот же JSON |
| `/finetune` | ✅ реальный CMA-ES | RL/IL поверх того же контракта |
| `/deploy` | ⚠️ заглушка (deploy_id + эхо) | ROS2-мост (`ros2 action`/`service` за HTTP) — тот же JSON |
| `Notify (Telegram)` | disabled до кред | Slack — нода `n8n-nodes-base.slack`, текст тот же `notification_text` |
| Журнал | ✅ JSONL у воркера (`/log_iteration`) | Google Sheets/Postgres — заменить одну ноду |
