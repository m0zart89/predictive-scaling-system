# Predictive autoscaling for Kubernetes — the project in nine sentences

**Problem.** Kubernetes autoscalers react to load that has already arrived, so
every predictable ramp-up of a business-hours workload starts with a minute or
two of under-provisioned, slow service. Forecasting the load is the obvious
fix, but a forecast only matters if it reaches the replica count, stays
correct as the workload changes, and fails safe when it is wrong. I wanted to
build that whole loop end to end and measure what it actually buys.

**Solution.** A KEDA external scaler (Python, gRPC) reads recent demand from
Prometheus, asks a FastAPI forecast service for the next 30 minutes, and
reports `max(forecast, observed)`, so a bad forecast can cost money but never
availability. A nightly retraining pipeline promotes a model through MLflow
aliases only if it beats the current one on data neither has seen and passes
a level-tracking gate I added after finding that the original Random Forest
could not forecast above its training range. The service scores its own past
forecasts for drift alerts, and the whole stack ships as one Helm chart that a
single script brings up on kind, with an e2e test and a benchmark in GitHub
Actions.

**Result.** In a benchmark against a plain CPU HPA on a compressed day of
traffic, the predictive scaler had the morning ramp's capacity in place in
60 s instead of 165 s and cut p95 latency during that ramp by 8%, at the cost
of 13% more replica-minutes. On the afternoon ramp and at peak it made no
difference, because the HPA had kept its capacity, and the forecast follows a
ramp rather than anticipating it; time-of-day features are the measured next
step. The first benchmark run also caught a real units bug between training
and scaling, which is now checked in CI.

---

## Коротко по-русски

**Задача.** Автоскейлеры Kubernetes реагируют на нагрузку, которая уже
пришла, поэтому каждая предсказуемая утренняя рампа начинается с минуты-двух
медленного сервиса. Прогноз нагрузки помогает, только если он доходит до
числа реплик, остаётся точным при изменении нагрузки и безопасно деградирует
при ошибке. Я построил этот цикл целиком и измерил, что он даёт.

**Решение.** KEDA external scaler на Python (gRPC) берёт историю спроса из
Prometheus, запрашивает прогноз на 30 минут у FastAPI-сервиса и сообщает
`max(прогноз, факт)`: ошибка прогноза может стоить денег, но не доступности.
Ночной пайплайн переобучения продвигает модель через aliases MLflow, только
если она лучше текущей на данных, которых не видела ни одна из них, и
проходит проверку слежения за уровнем — исходный Random Forest не умел
прогнозировать выше диапазона обучения. Сервис сам оценивает свои прошлые
прогнозы для алертов о дрейфе, а вся система ставится одним Helm-чартом и
одним скриптом на kind, с e2e-тестом и бенчмарком в GitHub Actions.

**Результат.** В бенчмарке против обычного HPA по CPU на сжатом дне трафика
предиктивный скейлер набрал нужную для утренней рампы ёмкость за 60 с вместо
165 с и снизил p95 на этой рампе на 8% ценой 13% лишних replica-minutes. На
дневной рампе и на пике разницы не было: HPA сохранил ёмкость, а прогноз
следует за начавшейся рампой, а не предвосхищает её — следующий измеренный
шаг это признаки времени суток. Первый же прогон бенчмарка поймал реальную
ошибку единиц между обучением и скейлингом, теперь она проверяется в CI.
