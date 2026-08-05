#!/usr/bin/env bash
# Gera a CA e o certificado TLS do servidor para o gRPC (cliente oficial sem patch).
# Uso: bash scripts/generate_certs.sh  (executar no mesmo diretório da repo)
set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/certs"
DAYS_CA=3650
DAYS_SERVER=825
SERVER_IP="${ALPHAGENOME_SERVER_IP:-10.9.8.193}"

mkdir -p "$CERT_DIR"

# 1. CA auto-assinada
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout "$CERT_DIR/ca.key" -out "$CERT_DIR/ca.crt" \
  -days "$DAYS_CA" -subj "/CN=AlphaGenome DGX CA"

# 2. Chave do servidor + CSR
openssl req -newkey rsa:4096 -nodes \
  -keyout "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" \
  -subj "/CN=$SERVER_IP"

# 3. Certificado do servidor assinado pela CA, com SAN = IP + localhost
openssl x509 -req -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" -CAcreateserial \
  -out "$CERT_DIR/server.crt" -days "$DAYS_SERVER" \
  -extfile <(printf "subjectAltName=IP:%s,DNS:localhost,IP:127.0.0.1" "$SERVER_IP")

chmod 600 "$CERT_DIR/ca.key" "$CERT_DIR/server.key"

echo "Certificados gerados em $CERT_DIR:"
ls -la "$CERT_DIR"
