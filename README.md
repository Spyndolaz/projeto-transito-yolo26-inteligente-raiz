# Câmera Inteligente de Trânsito com YOLO26

Sistema local para uma maquete de trânsito: YOLO26 detecta, ByteTrack mantém IDs e o motor de tráfego interpreta fluxo, paradas e congestionamento. Um agente local baseado em regras registra experiências e conduz aprendizado contínuo **controlado**. Uma previsão nunca vira rótulo sem confirmação humana.

## Arquitetura

```text
câmera/vídeo → YOLO26 + ByteTrack → contagem/trilhas/paradas → eventos + memória
                                      ↓
                               agente de regras → casos incertos → revisão humana
                                      ↓                                 ↓
                          treinamento de candidato ← dataset versionado ← aprovados
                                      ↓
                         avaliação → promoção ou rejeição → histórico
```

O código existente permanece em `utils/`: `pipeline.py` executa a visão, `tracking.py` guarda trajetórias, `counting.py` conta cruzamentos e `traffic.py` calcula o indicador heurístico. Os novos módulos têm responsabilidades separadas:

- `traffic/events.py`: eventos JSON Lines em `logs/events.jsonl`.
- `agent/`: agente local e memória persistente em `data/memory/memory.jsonl`.
- `learning/`: seleção, coleta, revisão, versões de dataset, avaliação e modelos.
- `models/production`, `models/candidates`, `models/archive`: versões de pesos sem exclusão automática.

## Instalação no Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

As classes e IDs do dataset são preservados: `0 car`, `1 bike`, `2 Big car`, `3 people`.

## Detecção e demonstração

Treine uma primeira vez para produzir `runs/transito/weights/best.pt`:

```powershell
python scripts/treinar.py
python scripts/detectar_webcam.py
python scripts/detectar_video.py --source video.mp4
python scripts/detectar_maquete.py
```

A tela mostra modelo, estado do agente, objetos, IDs, confiança, trilhas, fluxo, parados e o nível de congestionamento. Pressione `q` ou `Esc` para encerrar. Ajuste câmera, linhas, zonas e limiares em `config/config.py`.

Velocidade somente é estimada quando `CALIBRATION_DISTANCE_METERS` e `CALIBRATION_DISTANCE_PIXELS` estiverem definidos; sem eles, não há km/h. O congestionamento é marcado como **estimativa heurística**, não como medida cientificamente validada.

## Aprendizado contínuo controlado

Com `LEARNING_ENABLED=True`, baixa ou média confiança pode gerar um frame em `data/learning_queue/pending/`, acompanhado de JSON com a previsão. O `LearningSampler` limita a frequência por classe para não salvar sequências quase idênticas. Eventos e memória sobrevivem ao encerramento.

Revise os casos, fornecendo caixa YOLO confirmada para um caso aprovado:

```powershell
python scripts/revisar_casos.py
python scripts/criar_dataset.py
```

Confirmados vão para `approved`; rejeitados para `rejected`; incompletos ou duvidosos para `uncertain`. `criar_dataset.py` cria `data/dataset_v001`, `dataset_v002` etc., mantendo intocado o dataset original e gravando `manifest.json`.

Treine, avalie e promova explicitamente:

```powershell
python scripts/treinar_candidato.py --data data/dataset_v001/data.yaml --version v002
python scripts/avaliar_modelo.py --candidate models/candidates/candidate_v002.pt
python scripts/promover_modelo.py --candidate models/candidates/candidate_v002.pt --report logs/evaluation_candidate_v002.json --version v002
python scripts/rollback_modelo.py --model models/archive/v001.pt --version v001
```

O avaliador compara precision, recall, mAP50 e mAP50-95. A promoção só acontece se o relatório contiver a melhoria mínima configurada (`MIN_MODEL_IMPROVEMENT`); caso contrário, o candidato é arquivado como rejeitado. As decisões e métricas são registradas em `models/history.jsonl` e os pesos anteriores são preservados em `models/archive`.

## Limites atuais e próximos passos

O núcleo é offline e não depende de Waze, GPU nem API de LLM. Waze deve ser acrescentado apenas como contexto externo, nunca como substituto da visão. A revisão é por terminal e o usuário fornece a caixa YOLO confirmada; dashboard web e rotulagem gráfica são próximos incrementos. Os limiares, zonas, velocidade e congestionamento precisam ser calibrados na câmera e maquete reais.

## Testes

```powershell
python -m unittest discover -s tests -v
```
