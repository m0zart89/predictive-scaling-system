{{/*
One ScaledObject on the predictive scaler. Used for every entry of
.Values.workloads and for the demo workload, so there is one place that knows
the trigger metadata. Usage:

  include "predictive-stack.scaledobject" (dict "root" . "w" $workload)

Required keys of w: name, deployment, query, targetValue. Everything else
falls back to .Values.scalerDefaults.
*/}}
{{- define "predictive-stack.scaledobject" -}}
{{- $root := .root }}
{{- $w := .w }}
{{- $d := $root.Values.scalerDefaults }}
{{- range $k := list "name" "deployment" "query" "targetValue" }}
{{- if not (hasKey $w $k) }}{{ fail (printf "workload %q: %s is required" (get $w "name" | default "?") $k) }}{{ end }}
{{- end }}
{{- $fallback := $w.fallback | default $d.fallback }}
# Scales {{ $w.deployment }} on forecast demand:
#   desiredReplicas = ceil(reportedDemand / targetValue),
#   reportedDemand  = max(forecast, observed) in the query's units (mode=max).
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {{ $w.name }}
  labels:
    {{- include "predictive-stack.labels" $root | nindent 4 }}
    predictive-stack/workload: {{ $w.name }}
spec:
  scaleTargetRef:
    name: {{ $w.deployment }}
  minReplicaCount: {{ $w.minReplicas | default $d.minReplicas }}
  maxReplicaCount: {{ $w.maxReplicas | default $d.maxReplicas }}
  pollingInterval: {{ $w.pollingInterval | default $d.pollingInterval }}
  cooldownPeriod: {{ $w.cooldownPeriod | default $d.cooldownPeriod }}
  # Used when the scaler returns an error (Prometheus down, no samples):
  # hold a safe replica count instead of guessing.
  fallback:
    failureThreshold: {{ $fallback.failureThreshold }}
    replicas: {{ $fallback.replicas }}
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
            - type: Percent
              value: 100
              periodSeconds: 15
        scaleDown:
          stabilizationWindowSeconds: {{ $w.scaleDownStabilizationSeconds | default $d.scaleDownStabilizationSeconds }}
          policies:
            - type: Pods
              value: 1
              periodSeconds: 60
  triggers:
    - type: external
      metricType: AverageValue
      metadata:
        scalerAddress: predictive-scaler.{{ $root.Release.Namespace }}.svc.cluster.local:6000
        prometheusServerAddress: {{ $w.prometheusServerAddress | default (printf "http://prometheus.%s.svc.cluster.local:9090" $root.Release.Namespace) }}
        # Total demand, independent of replica count (aggregate to one series).
        query: {{ $w.query | quote }}
        forecastServiceURL: http://forecast-serving.{{ $root.Release.Namespace }}.svc.cluster.local:8080
        targetValue: {{ $w.targetValue | quote }}
        # model input = demand / capacity: the model is trained on that series
        # (training.workload), so the two must agree
        capacity: {{ $w.capacity | default $d.capacity | quote }}
        timeSteps: {{ $w.timeSteps | default $d.timeSteps | quote }}
        stepSeconds: {{ $w.stepSeconds | default $d.stepSeconds | quote }}
        mode: {{ $w.mode | default $d.mode }}
        forecastAggregation: {{ $w.forecastAggregation | default $d.forecastAggregation }}
        activationValue: {{ $w.activationValue | default $d.activationValue | quote }}
        cacheSeconds: {{ $w.cacheSeconds | default $d.cacheSeconds | quote }}
        timeoutSeconds: {{ $w.timeoutSeconds | default $d.timeoutSeconds | quote }}
{{- end }}

{{/*
The workload the model is trained on: training.workload by name, else the
first of .Values.workloads, else the demo. Returns a dict with query and
capacity (as a YAML-encoded dict, decode with fromYaml).
*/}}
{{- define "predictive-stack.trainingWorkload" -}}
{{- $found := dict }}
{{- $name := .Values.training.workload }}
{{- range .Values.workloads }}
{{- if or (and $name (eq .name $name)) (and (not $name) (not $found)) }}
{{- $found = dict "name" .name "query" .query "capacity" (.capacity | default $.Values.scalerDefaults.capacity) }}
{{- end }}
{{- end }}
{{- if and $name (not $found) }}{{ fail (printf "training.workload %q is not in .Values.workloads" $name) }}{{ end }}
{{- if and (not $found) .Values.demo.enabled }}
{{- $found = dict "name" "cpu-burner" "query" .Values.demo.query "capacity" .Values.demo.predictive.capacity }}
{{- end }}
{{- $found | toYaml }}
{{- end }}
