---
title: "Approvals (разрешения)"
description: "Approval в raytsystem — это точная, истекающая запись, привязанная к типу действия, хэшу полезной нагрузки и конкретным целям. Что требует разрешения и почему старое approval нельзя переиспользовать."
audience: [operator]
status: stable
feature_flags: [runtime_execution_enabled, external_mcp_execution_enabled, external_notifications_enabled]
related_commands:
  - "uv run raytsystem proposal import"
  - "uv run raytsystem mcp approve"
  - "uv run raytsystem package approve"
  - "uv run raytsystem workflow approve"
related_pages:
  - /security/overview
  - /security/emergency-controls
  - /security/defaults
  - /observability/policy-simulator
  - /reference/feature-flags
source_of_truth:
  - path: src/raytsystem/authority.py
  - path: src/raytsystem/contracts/workflows.py
  - path: src/raytsystem/workflows/service.py
  - path: docs/10-execution-security.md
last_verified_against: "schema v1.4.0"
---

# Approvals (разрешения)

## Что это

Approval — это точная, истекающая запись, которая разрешает ровно одно действие. Она
привязана к конкретным полям и проверяется fail-closed резолвером `AuthorityResolver`,
который берёт записи только из доверенных локальных хранилищ. Источник:
`src/raytsystem/authority.py`.

Для approval-узлов workflow публичный `ApprovalAuthorityService` сначала читает текущий
`waiting`-шаг, его run и зарегистрированный immutable gate из `PlatformStore`. Клиент передаёт
только run, node, approver и idempotency key: хэш входа, роль и срок действия нельзя подменить
параметрами вызова.

## Когда использовать

Approval нужен всякий раз, когда действие выходит за границу «безопасного по умолчанию»:
включение адаптера, любой сетевой egress, promotion в канонический корпус, публикация,
push, удаление, оплата или egress приватного (private-corpus) содержимого. Реальное
исполнение рантайма с egress к внутреннему/приватному провайдеру дополнительно требует
неистёкшего approval. Источник:
`docs/10-execution-security.md`.

## К чему привязано approval

Резолвер считает approval валидным, только если совпадает всё сразу:

- тип действия (`action`);
- хэш полезной нагрузки (`payload_sha256` / `artifact_sha256`) — то есть привязка к
  конкретному содержимому;
- назначение (`destination`), например конкретный провайдер или получатель;
- цель — один из идентификаторов employee, task, run или workspace;
- требуемый scope (`required_scope` должен быть подмножеством scope разрешения);
- срок действия: `approved_at <= now < expires_at`;
- при необходимости — версия/хэш политики (`policy_sha256`).

Дополнительно резолвер пересобирает `approval_id` из полей записи и сверяет его — это
защищает от подделки. Источник:
`src/raytsystem/authority.py`.

## Почему старое approval нельзя переиспользовать

Approval привязано к хэшу полезной нагрузки. Если payload изменился, хэш не совпадёт, и
резолвер выбросит ошибку `Approval does not match the exact action scope`. Точно так же
отклоняется approval с истёкшим сроком, с другим destination, с целью вне привязки или с
недостаточным scope. Это исключает повторное использование «почти подходящего» разрешения.

## Approval для workflow

`ApprovalAuthorityService.inspect_pending(workflow_run_id, node_id, at=None)` возвращает
immutable `PendingWorkflowApproval`: точные идентификаторы run, step, node и gate, action и target,
workflow revision, проверенный хэш входа, scope hash, policy version, требуемую роль и абсолютный
UTC-срок действия.

`ApprovalAuthorityService.list_pending(limit=100, cursor=None, at=None)` возвращает замороженный
`PendingWorkflowApprovalPage` и перечисляет ожидающие approval напрямую из канонических run,
revision, step и gate записей. Метод не использует ограниченные `WorkflowService.snapshot()`
списки и поэтому включает активные запуски старых revision после публикации новой. Порядок
стабилен; `next_cursor` непрозрачен и подписан локальным ключом workspace. Все страницы одной
обходной последовательности несут одинаковые `snapshot_id` и `observed_at`. Если записи меняются
между страницами, cursor повреждён или caller пытается сменить время наблюдения, метод возвращает
явную consistency error вместо пропуска или дублирования approvals.

`ApprovalAuthorityService.issue_approval(..., approver, idempotency_key, at=None)` повторно читает
те же доверенные записи и создаёт стандартный `ApprovalRecord`. Запись approval и две стороны
idempotency binding — pending target и caller key — сохраняются одной транзакцией. Точный повтор
при всё ещё ожидающем шаге и действующем gate возвращает тот же `approval_id`; смена approver,
key, run/node binding, входа или gate отклоняется без второй записи. Истёкший gate, неверный node
и шаг не в состоянии `waiting` также отклоняются.

Issuance не меняет состояние workflow. Публичные переходы
`WorkflowService.grant_approval(..., idempotency_key=...)` и
`deny_approval(..., expected=pending, idempotency_key=...)` требуют непустой точный ключ; deny
дополнительно требует тот самый типизированный `PendingWorkflowApproval`, который наблюдал caller.
При первом deny движок под тем же `BEGIN IMMEDIATE` заново выводит live binding из канонических
run/revision/step/gate записей и сравнивает весь контракт до перехода, event или receipt. При
первом вызове
движок одной транзакцией `BEGIN IMMEDIATE` фиксирует переход шага/запуска, audit event и
immutable receipt. Receipt связывает решение с run, node, step, входным хэшем, actor,
approval ID (только для grant), gate ID/action/role, версией и хэшем политики и сроком gate.

Если вызывающая сторона упала после commit, повтор с тем же ключом и теми же параметрами сначала
проверяет receipt и точное terminal-состояние/event, затем возвращает исходный `WorkflowRun` без
второго перехода. Другой ключ после terminal-перехода, переиспользование ключа с изменённой
привязкой и orphan/corrupt receipt отклоняются fail-closed. Grant повторно проверяет канонический
gate и точный accepted `ApprovalRecord`; deny не создаёт и не потребляет approval record. Exact
deny replay обязан повторить исходный `expected`: тот же ключ с изменённым expected binding
отклоняется.

Grant receipts и authority issuance receipts, созданные до добавления полей revision/policy,
читаются совместимо и всё равно перепроверяются по текущим каноническим записям. Старый deny
receipt не содержит доказательства `expected` и намеренно не преобразуется в exact deny: его retry
после обновления завершается fail-closed; terminal-состояние и исходное событие проверяйте через
workflow audit, не создавая второе решение.

## Пример

Проверка разрешений и решений политики выполняется автоматически в тех операциях, которые
их требуют. Для отдельных потоков есть явные команды подтверждения, например:

```bash
uv run raytsystem mcp approve
uv run raytsystem package approve
uv run raytsystem workflow approve <run_id> <node_id> \
  --approval-id <approval_id> --idempotency-key <stable_retry_key>
```

Оценить, какое решение вынесет политика для гипотетического действия, можно в симуляторе
политики (`uv run raytsystem policy simulate`).

## Ожидаемый результат

При совпадении всех полей действие разрешается ровно один раз в рамках привязки. При любом
расхождении оно закрывается наглухо (fail closed).

## Ограничения и безопасность

- Экстренное действие `revoke_pending_approvals` инвалидирует все approvals, выданные до
  времени его активации: резолвер сверяет `approved_at` с временем отзыва. Источник:
  `src/raytsystem/authority.py`.
- Защитные circuit breakers защёлкиваются (latch) в открытом состоянии и закрываются только
  свежим точечным approval — см. `emergency-controls.md`.
- Approval не заменяет решение политики: для реального исполнения нужны и `ALLOW`, и
  подходящее approval одновременно.

## Частые ошибки

- Пытаться применить approval после правки payload — хэш не совпадёт.
- Ожидать, что approval «широкого» scope закроет действие вне его привязки к цели/destination.
- Повторять workflow approve с новым ключом после неясного ответа: используйте исходный
  `--idempotency-key`, иначе terminal-шаг не будет принят за успешный replay.
- Начинать следующую страницу pending approvals заново после изменения store: продолжение старого
  cursor намеренно вернёт consistency error; начните новый обход с `cursor=None`.

## Связанные страницы

- `overview.md`
- `emergency-controls.md`
- `observability/policy-simulator.md`
- `defaults.md`

## Источники истины

- `src/raytsystem/authority.py`
- `src/raytsystem/contracts/workflows.py`
- `src/raytsystem/workflows/service.py`
- `docs/10-execution-security.md`
