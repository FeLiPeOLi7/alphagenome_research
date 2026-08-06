## Construção de um Servidor para rodar o AlphaGenome

```text
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
```

Esse projeto inclui arquivos que permitem rodar o servidor do AlphaGenome localmente (ex: `server.py`).

Para rodar o servidor, primeiro ative o ambiente virtual (como descrito no `README.md`):

```bash
conda create -n alphagenome python=3.11 -y

conda activate alphagenome

pip install -e .

python3 server.py

# Em outro terminal, rode seu cliente. Ou, rode algum teste, como test_sdk
python3 tests/test_sdk.py
```


## Utilizando o AlphagenomeViewer

Para utilizar a parte gráfica da aplicação é simples, basta instalar a biblioteca e modificar para onde o cliente AlphaGenome está apontando:

```bash
pip install alphagenome-viewer
```

```text

# Utilize o seu editor de preferencia e busque onde a biblioteca alphagenome-viewer foi instalada
# e.g.
nvim /home/SEU_USUARIO/miniconda3/envs/alphagenome/lib/python3.11/site-packages/app/services/alphagenome.py  

#Modifique o construtor da classse AlphaGenomeServicer para apontar para o servidor local (ao qual não precisa de uma chave API)
class AlphaGenomeService:
    """Service for interacting with AlphaGenome API."""

    def __init__(self, api_key: str):
        """Initialize service with API key."""
        key = api_key if api_key else "dummy"
        self.client = dna_client.create(api_key=key, address="10.x.x.xxx:50051") #Coloque o endereço em que o server.py está rodando
        self._load_gtf()

```

---

Após isso, basta rodar o alphagenome-viewer em conjunto com o server.py

```bash
alphagenome-viewer
```

Repositório do Alphagenome-Viewer: https://github.com/Abrar-Abir/alphagenome-viewer
