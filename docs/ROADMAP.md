# Roadmap

Where the project is, what comes next, and ready-to-paste briefs for each
stage. The briefs are written for an agentic coding session (Claude Code cloud
session or similar) but read fine as a human task list too.

## Done

| Stage | What | Where |
|---|---|---|
| 0 | Forecasting service prototype: FastAPI + MLflow, notebook training, Prometheus/Grafana manifests, Helm chart, CI | branch `milestone/v0.1.0` |
| 1 | **KEDA external scaler** closes the loop forecast → replicas; demo workload; failure policy (reactive fallback, KEDA `fallback` on Prometheus errors) | `scaler/`, `demo/`, `helm/predictive-scaler/` — PR #1 |
| 2 | **Retraining pipeline** with a level-tracking gate (the notebook's model was pinned to its training range), delta-target candidates, 30-minute horizon, fair champion comparison on unseen data, MLflow aliases, nightly CronJob | `training/`, `k8s/retrain-cronjob.yaml` — PR #2 |

## Next

Stages 3 and 4 are independent and can run in parallel. Stage 5 needs a real
cluster. Stage 6 is polish and can run any time after 5.

### Stage 3 — Observability of forecast quality and drift

```
Контекст: проект predictive-scaling-system. Уже сделано: KEDA external
scaler (scaler/) и пайплайн переобучения training/ с многошаговым
прогнозом и MLflow aliases. Создай ветку feat/observability от main и
открой PR.

Задача: наблюдаемость качества прогнозов и дрейфа модели.

1. В serving/app.py — метрика ошибки прогноза: храни недавние прогнозы
   в памяти (по шагу горизонта); когда приходит окно, содержащее
   фактическое значение для прошлого прогноза, обновляй гистограмму
   forecast_abs_error и gauge forecast_rolling_mae (label horizon_step).
   Без внешних хранилищ, с лимитом памяти. Тесты в tests/.
2. monitoring/prometheus-rules.yaml: алерты ForecastDriftHigh (rolling
   MAE выше порога 30 мин), ScalerFallbackRate (доля source="observed"
   в predictive_scaler_decisions_total > 50% за 15 мин),
   ScalerPrometheusErrors. У каждого runbook_url.
3. Grafana-дашборды как код (JSON в monitoring/dashboards/, provisioning
   через ConfigMap в monitoring/grafana.yaml):
   - "Predictive scaling": observed / predicted / reported demand и
     desired replicas по каждому ScaledObject, fallback rate, latency
     прогноза;
   - "Model quality": MAE по шагам горизонта, версия модели, reloads.
   Проверяй JSON скриптом в CI (python, без внешних образов).
4. docs/runbooks/: по файлу на каждый алерт — что значит, как
   проверить, как исправить.
5. Обнови README (Limitations, Future Work, структура).

Ограничения среды: Docker Hub и GHCR недоступны, кластер не поднять.
Проверяй тестами и линтом (ruff 0.4.4), сборку образов — через CI в PR.
Следи за CI через gh api и чини, пока не станет зелёным. Коммить и
пушить после каждого шага.
```

### Stage 4 — Portable deployment: kind, Ingress, GHCR images

```
Контекст: проект predictive-scaling-system; манифесты и start.sh
привязаны к Minikube (NodePort, minikube docker-env, захардкоженный IP
192.168.49.2). Создай ветку feat/kind-deploy от main и открой PR.

Задача: сделать деплой переносимым и воспроизводимым в CI.

1. Публикация образов в GHCR из CI (forecast-serving, predictive-scaler,
   cpu-burner, forecast-training) с тегами sha и latest при пуше в main;
   на PR — только сборка.
2. Единый Helm umbrella-чарт helm/predictive-stack/ (или kustomize)
   для всей системы: MLflow, Prometheus (с pod discovery и RBAC), Grafana,
   forecast-serving, predictive-scaler, KEDA как зависимость, демо-
   workload и ScaledObject как опция. Ingress вместо NodePort, образы
   из GHCR, namespace-RBAC.
3. scripts/up.sh: kind create cluster → helm install → ожидание
   readiness → обучение модели из notebooks/data-jan-2022.json
   (training/) → /reload → smoke-тест /predict и gRPC скейлера.
4. GitHub Actions workflow e2e.yml: запускает scripts/up.sh на kind,
   даёт нагрузку на cpu-burner 10 минут со stepSeconds=30 и проверяет,
   что predictive_scaler_decisions_total{source="forecast"} растёт и
   HPA менял replicas. Логи и метрики — в artifacts.
5. Обнови README Quick Start и удали ссылки на Minikube IP.

Ограничения среды сессии: Docker Hub/GHCR недоступны, кластер локально
не поднять — всё, что требует Docker, проверяй через GitHub Actions.
Итерируй по логам CI (gh api) до зелёного. Коммить и пушить по шагам.
```

### Stage 5 — Benchmark: predictive vs reactive HPA

Needs stage 4 (a cluster in CI).

```
Контекст: проект predictive-scaling-system, есть e2e на kind
(scripts/up.sh, .github/workflows/e2e.yml). Создай ветку
feat/benchmark от main и открой PR.

Задача: измерить, что даёт предиктивное масштабирование.

1. bench/: генератор нагрузки (k6 или python) с суточным паттерном,
   сжатым во времени (24 ч → 40 мин, stepSeconds=30 для скейлера и
   обучения), с резкими утренними рампами.
2. Два прогона на одном кластере: (a) обычный HPA по CPU 70%,
   (b) predictive-scaler с mode=max, forecastAggregation=max.
   Перед (b) обучи модель на данных прогона-разогрева через training/.
3. Метрики из Prometheus после каждого прогона: p95 latency cpu-burner
   во время рамп, доля запросов > 500 мс, replica-minutes,
   время от начала рампы до первого scale-up.
4. bench/report.py: таблица и PNG-графики (нагрузка, реплики, p95 для
   обоих режимов на одной оси) → docs/benchmark.md и artifacts в CI.
5. Workflow benchmark.yml (ручной запуск + еженедельно). Результаты
   и честные выводы (включая случаи, где прогноз не помог) — в README.

Коммить и пушить по шагам; CI — единственный способ проверки.
```

### Stage 6 — Polish for portfolio

```
Контекст: проект predictive-scaling-system завершён функционально.
Создай ветку chore/polish от main и открой PR.

1. Ревью всего репозитория свежим взглядом: ошибки, мёртвый код,
   несоответствия README и кода, дубли между serving и training.
   Исправь, допиши тесты.
2. Обнови ARCHITECTURE.md: схема системы с скейлером, пайплайном и
   наблюдаемостью; таблица решений (почему KEDA, почему delta-target,
   почему aliases).
3. CHANGELOG.md по стадиям; тег v1.0.0 через release (вручную владельцем).
4. Короткое описание проекта для CV/LinkedIn (docs/pitch.md):
   3 предложения о задаче, 3 — о решении, 3 — о результате с цифрами
   из docs/benchmark.md.
```

## Deferred ideas

- Additional signals (memory, queue depth) as features — needs a workload
  where CPU is not the bottleneck.
- Weekly seasonality once more than a month of data exists.
- Multi-tenant scaler config via CRD instead of trigger metadata.
