## Construção de um Servidor para rodar o AlphaGenome

  ┌─────────────────────────────────┐
  │      Cliente (SDK Python)       │
  │     `alphagenome_dgx.py`        │
  └────────────────┬────────────────┘
                   │  gRPC / Protobuf
                   ▼
  ┌─────────────────────────────────┐
  │     Servidor (server.py)        │
  │   gRPC Service + JAX Engine     │
  └────────────────┬────────────────┘
                   │  VRAM / CUDA
                   ▼
  ┌─────────────────────────────────┐
  │   NVIDIA GPU / AlphaGenome SDK  │
  └─────────────────────────────────┘

Esse projeto inclui arquivos que permitem rodar o servidor do AlphaGenome localmente (ex: server.py)

Para rodar o servidor, primeiro ative o ambiente virtual (como descrito no README.md):

```bash
conda activate alphagenome

python3 server.py

# Em outro terminal, rode seu cliente. Ou, rode algum teste, como test_sdk
python3 test_sdk.py
```


