# Boscosoft Agents — Docker Deployment

Two independent FastAPI microservices, each in its own container, unified
behind a single nginx gateway on port 80.

```
deploy/
├── agent1/              Task & Estimation Validation Agent
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example     copy to .env and fill in your real key
├── agent2/               Timesheet Execution Review Agent
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── nginx/
│   └── nginx.conf        reverse proxy: routes /agent1/* and /agent2/*
└── docker-compose.yml
```

## 1. Set up environment variables

```bash
cp agent1/.env.example agent1/.env
cp agent2/.env.example agent2/.env
```

Edit both `.env` files and set your real `GROQ_API_KEY` (and `GROQ_MODEL`
if you want to override the default). Each agent has its own `.env`, so
they can use the same or different Groq models/keys independently.

**Never commit the real `.env` files to git.**

## 2. Build and run

```bash
docker compose up -d --build
```

This builds both agent images, starts them, and starts nginx in front.

## 3. Verify

```bash
curl http://localhost/agent1/health
curl http://localhost/agent2/health
```

Both should return a JSON status.

## 4. Endpoints (through the gateway)

**Agent 1 — Task Validation**
- `POST /agent1/api/v1/task/validate`
- `POST /agent1/api/v1/backlog/validate`

**Agent 2 — Timesheet Review**
- `POST /agent2/api/v1/manager/timesheet-review`

## 5. Logs / management

```bash
docker compose logs -f agent1
docker compose logs -f agent2
docker compose logs -f nginx

docker compose restart agent1   # restart just one service, the other stays up

docker compose down             # stop everything
```

## Notes for whoever deploys this (e.g. your TL)

- **Independent scaling/restart**: `agent1` and `agent2` are separate
  containers. A crash or redeploy of one does not affect the other.
- **Separate secrets**: each service has its own `.env`, even though both
  currently use the same variable names (`GROQ_API_KEY`, `GROQ_MODEL`).
  They can be pointed at different Groq models without touching each other.
- **Swagger docs caveat**: FastAPI's auto-generated `/docs` (Swagger UI)
  pages reference absolute asset paths, so `http://yourhost/agent1/docs`
  may not load its static assets correctly behind the stripped `/agent1/`
  prefix. If Swagger UI is needed in production, expose each agent
  directly on its own subdomain/port instead of path-prefixed routing,
  or add a small nginx rewrite for `/docs` and `/openapi.json`.
- **File upload size**: nginx is configured with `client_max_body_size 20M`
  to allow Agent 1's Excel upload endpoint. Increase if larger sprint
  files are expected.
- **HTTPS**: this config is HTTP-only (port 80). For production, put this
  behind a TLS-terminating load balancer, or add a certbot/Let's Encrypt
  container and extend `nginx.conf` with a `listen 443 ssl` block.
