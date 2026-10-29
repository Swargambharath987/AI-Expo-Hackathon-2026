#!/usr/bin/env bash
# Set the server's Anthropic API key — on Cloud Run and in the local .env.
#
#   ./scripts/set_llm_key.sh
#
# Prompts for the key with hidden input (it never appears on screen or in
# shell history), checks it with Anthropic, then updates:
#   - Cloud Run service "duty-line" (new revision, no rebuild — ~1 minute)
#   - LLM_API_KEY in the local .env
set -euo pipefail

SERVICE="duty-line"
REGION="us-central1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"

read -rsp "Paste your Anthropic API key (input hidden), then press Enter: " KEY
echo
KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"

if [[ ! "$KEY" =~ ^sk-ant- ]]; then
  echo "That doesn't look like an Anthropic key (should start with sk-ant-). Nothing changed."
  exit 1
fi

echo "Checking the key with Anthropic..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/models \
  -H @<(printf 'x-api-key: %s\nanthropic-version: 2023-06-01\n' "$KEY"))
if [[ "$STATUS" != "200" ]]; then
  echo "Anthropic rejected the key (HTTP $STATUS). Nothing changed."
  exit 1
fi
echo "Key accepted."

echo "Updating Cloud Run ($SERVICE)... this takes about a minute."
gcloud run services update "$SERVICE" --region "$REGION" \
  --update-env-vars "LLM_API_KEY=$KEY" --quiet >/dev/null
echo "Cloud Run updated."

if [[ -f "$ENV_FILE" ]] && grep -q '^LLM_API_KEY=' "$ENV_FILE"; then
  # Rewrite via a temp file (BSD/GNU sed differ on -i); keep permissions tight
  umask 077
  awk -v k="$KEY" '/^LLM_API_KEY=/{print "LLM_API_KEY=" k; next} {print}' "$ENV_FILE" > "$ENV_FILE.tmp"
  mv "$ENV_FILE.tmp" "$ENV_FILE"
  echo "Local .env updated."
fi

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)")
echo
echo "Done. Health check:"
curl -s "$URL/api/health"
echo
