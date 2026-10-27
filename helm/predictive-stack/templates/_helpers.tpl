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
