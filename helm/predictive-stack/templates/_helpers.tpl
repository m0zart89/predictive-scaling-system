{{/*
Common labels for resources this chart owns (subcharts label their own).
*/}}
{{- define "predictive-stack.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: predictive-stack
{{- end }}

{{/*
Image reference for one of the project's images: <repository>:<tag>, where
tag falls back to global.imageTag. Usage: include "predictive-stack.image" (dict "image" .Values.demo.image "global" .Values.global)
*/}}
{{- define "predictive-stack.image" -}}
{{- printf "%s:%s" .image.repository (.image.tag | default .global.imageTag | default "latest") }}
{{- end }}

{{/*
Prometheus endpoints. prometheusUrl: what the scaler and Grafana query inside
the cluster (external.url with mode=external, else the chart's Service).
trainingPrometheusUrl: where the training jobs read history (prometheus.trainingUrl
when set, e.g. Thanos Query, else prometheusUrl).
*/}}
{{- define "predictive-stack.prometheusUrl" -}}
{{- if eq .Values.prometheus.mode "external" -}}
{{- required "prometheus.external.url is required with prometheus.mode=external" .Values.prometheus.external.url -}}
{{- else -}}
{{- printf "http://prometheus.%s.svc.cluster.local:9090" .Release.Namespace -}}
{{- end -}}
{{- end }}

{{- define "predictive-stack.trainingPrometheusUrl" -}}
{{- .Values.prometheus.trainingUrl | default (include "predictive-stack.prometheusUrl" .) -}}
{{- end }}

{{/*
Every scaled workload (entries of .Values.workloads plus the demo when it is
scaled predictively) as a YAML list of {name, query, capacity}; the demand
recording rules and the training query are built from it.
*/}}
{{- define "predictive-stack.demandWorkloads" -}}
{{- $out := list }}
{{- range .Values.workloads }}
{{- $out = append $out (dict "name" .name "query" .query "capacity" (.capacity | default $.Values.scalerDefaults.capacity)) }}
{{- end }}
{{- if and .Values.demo.enabled (eq .Values.demo.scaling "predictive") }}
{{- $out = append $out (dict "name" "cpu-burner" "query" .Values.demo.query "capacity" .Values.demo.predictive.capacity) }}
{{- end }}
{{- dict "workloads" $out | toYaml }}
{{- end }}

{{/*
Recording rules: workload:demand{workload="<name>"} = the workload's demand
query, in the query's units, every minute. The scaler divides by capacity
itself, the training job's query does the same, so capacity stays in one place.
*/}}
{{- define "predictive-stack.demandRules" -}}
{{- $ws := (include "predictive-stack.demandWorkloads" . | fromYaml).workloads }}
groups:
  - name: predictive-stack.demand
    interval: 1m
    rules:
{{- range $ws }}
      - record: workload:demand
        labels:
          workload: {{ .name | quote }}
        expr: {{ .query | quote }}
{{- end }}
{{- if not $ws }}
      []
{{- end }}
{{- end }}

{{/*
The query a ScaledObject or the training job uses for a workload: the recorded
series with prometheus.recordDemand, else the raw query.
*/}}
{{- define "predictive-stack.demandQuery" -}}
{{- if .root.Values.prometheus.recordDemand -}}
workload:demand{workload="{{ .name }}"}
{{- else -}}
{{ .query }}
{{- end -}}
{{- end }}
