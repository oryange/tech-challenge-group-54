#!/usr/bin/env bash
# Provisiona os recursos Azure do projeto (idempotente). Requer: az login
set -euo pipefail

RG="${RG:-rg-medwatch-g54}"
LOCATION="${LOCATION:-brazilsouth}"
SKU="${SKU:-F0}"            # F0 = gratuito (limite de 1 por tipo por assinatura)
SPEECH="${SPEECH:-medwatch-speech-g54}"
LANGUAGE="${LANGUAGE:-medwatch-language-g54}"

az group create -n "$RG" -l "$LOCATION" -o none

create() {  # $1=nome $2=kind
  if ! az cognitiveservices account show -n "$1" -g "$RG" &>/dev/null; then
    az cognitiveservices account create -n "$1" -g "$RG" -l "$LOCATION" \
      --kind "$2" --sku "$SKU" --custom-domain "$1" --yes -o none
  fi
}
create "$SPEECH" SpeechServices
create "$LANGUAGE" TextAnalytics

key() { az cognitiveservices account keys list -n "$1" -g "$RG" --query key1 -o tsv; }

cat <<ENV
# --- copie para o .env ---
AZURE_SPEECH_KEY=$(key "$SPEECH")
AZURE_SPEECH_REGION=$LOCATION
AZURE_LANGUAGE_ENDPOINT=$(az cognitiveservices account show -n "$LANGUAGE" -g "$RG" --query properties.endpoint -o tsv)
AZURE_LANGUAGE_KEY=$(key "$LANGUAGE")
ENV
