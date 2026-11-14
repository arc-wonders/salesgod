# Voice deployment (voice-a100-4)

Live at https://voice.arkinkansra.in. The server is GCP `voice-a100-4` in project `voice-models-511206`, zone `asia-southeast1-c`, static IP `34.143.141.218`. voice-a100-1, -2 and -3 are not touched.

```
Plivo (+91 80655 89999) ──answer/hangup webhooks──► Caddy :443 ─► uvicorn voice.server :8080
                         ◄──bidirectional WS audio──►   VAD (Silero) → Whisper large-v3 (GPU)
                                                        → ACIE (domain FSM + Qwen3-14B-AWQ via vLLM :8000)
                                                        → Fish Audio s2.1-pro TTS (API) → mu-law → Plivo
```

| Unit | What |
|---|---|
| `salesgod-llm` | vLLM serving Qwen3-14B-AWQ as `acie-expression` on 127.0.0.1:8000 (0.6 of the GPU) |
| `salesgod-voice` | FastAPI console, Plivo webhooks and call sessions on 127.0.0.1:8080 |
| `caddy` | TLS for voice.arkinkansra.in |

- Secrets live in `/opt/salesgod/.env` (mode 600, never in git): Plivo auth, Fish key, console password, session secret.
- Call logs are in `/opt/salesgod/app/data/voice/calls/<id>.json`, with each customer turn's audio in `<id>/customer_NN.wav`.
- Products are in `data/voice/products.json` and are edited from the console.

## Update code
From a laptop: tar the changed paths, `gcloud compute scp` them to `/tmp`, untar into `/opt/salesgod/app` as `salesgod`, then run `sudo systemctl restart salesgod-voice`.

## Test without a phone
```
sudo -u salesgod bash -lc 'cd /opt/salesgod/app && set -a && . /opt/salesgod/.env && set +a && \
  VOICE_DATA_DIR=/tmp/sim /opt/salesgod/venv/bin/python -m voice.simcall llm claude-pro'
```
`voice.selftest` runs TTS → phone codec → VAD → Whisper → ACIE on a scripted conversation. `voice.simcall` runs a full call through the real CallSession, including barge-in.
