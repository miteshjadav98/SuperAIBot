#!/bin/bash
#
# Deployment script for the Super AI Bot platform.
#
# Target: the SAME Azure VM that already hosts AIKhataBook. This script is
# written to be a good neighbour on a shared box:
#   * uses uniquely-named systemd services (aibot-*) and an nginx site (aibot)
#   * uses non-default app ports so it can't collide with AIKhataBook
#   * NEVER removes the default nginx site or touches AIKhataBook / Redis
#
# Architecture (4 processes, only the frontend is public):
#   1. LangGraph server  (127.0.0.1:2024) - agent graphs + token streaming
#   2. FastAPI gateway   (127.0.0.1:8010) - control plane / PDF RAG (internal)
#   3. Next.js frontend  (127.0.0.1:3100) - chat UI, proxied by nginx
#   4. Prompt service    (127.0.0.1:8020) - versioned prompt management;
#      Swagger UI exposed at https://$DOMAIN/prompt-api/docs
#
# Public URL: https://aibot.miteklabs.tech  ->  nginx  ->  Next.js :3100
# The browser only ever talks to the Next.js app; its built-in /api route
# passes requests through to the LangGraph server server-side.
#
# Run from the project root (where this file lives):
#   chmod +x deployment_script.sh && ./deployment_script.sh

set -e

# ---- Config ---------------------------------------------------------------
DOMAIN="aibot.miteklabs.tech"
LANGGRAPH_PORT=2024
GATEWAY_PORT=8010          # FastAPI gateway (internal only)
PROMPTS_PORT=8020          # Prompt management service (Swagger at /prompt-api/docs)
FRONTEND_PORT=3100         # Next.js (avoid AIKhataBook's likely :3000)
APP_DIR="$(pwd)"
SERVICE_PREFIX="aibot"

echo "==> Deploying Super AI Bot from $APP_DIR to https://$DOMAIN"

# ---- 1. System dependencies ----------------------------------------------
echo "==> Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx curl

# Node.js (for the Next.js frontend) — install only if missing so we don't
# disturb a Node version AIKhataBook may already depend on.
if ! command -v node >/dev/null 2>&1; then
    echo "==> Node.js not found, installing Node 20 LTS via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
else
    echo "==> Node.js already present: $(node -v)"
fi

# ---- 2. Environment file check -------------------------------------------
if [ ! -f "$APP_DIR/.env" ]; then
    echo "ERROR: $APP_DIR/.env is missing."
    echo "       Copy .env.example to .env and fill in the real API keys first."
    exit 1
fi

# .env is gitignored, so a fresh `git pull` never updates it. Auth + MongoDB
# were added after the first deploy — fail early if this box's .env predates them.
for required in MONGODB_URI AUTH_SECRET; do
    if ! grep -qE "^${required}=.+" "$APP_DIR/.env"; then
        echo "ERROR: $required is missing from $APP_DIR/.env."
        echo "       This box's .env is out of date. Add (see .env.example):"
        echo "         MONGODB_URI  — Atlas connection string (Connect > Drivers)"
        echo "         MONGODB_DB   — e.g. superbot"
        echo "         AUTH_SECRET  — python -c \"import secrets; print(secrets.token_hex(32))\""
        echo "       and remove the old REDIS_* lines. Then re-run this script."
        exit 1
    fi
done

# ---- 3. Python virtual environment ---------------------------------------
echo "==> Creating Python virtual environment (.venv)..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/backend/requirements.txt"
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/prompt_service/requirements.txt"

# ---- 4. Frontend build ----------------------------------------------------
# NEXT_PUBLIC_* vars are baked in at build time, so the production env file
# must exist BEFORE `npm run build`. The browser calls the same-origin /api
# route; that route proxies to LangGraph server-side via LANGGRAPH_API_URL.
echo "==> Writing frontend production env..."
LANGSMITH_KEY="$(grep -E '^LANGSMITH_API_KEY=' "$APP_DIR/.env" | head -1 | cut -d= -f2- | tr -d "\"'" || true)"
# NEXT_PUBLIC_GATEWAY_URL is called DIRECTLY from the browser (login/register,
# the agent dropdown, PDF upload), so it must be a public, same-origin path.
# nginx proxies https://$DOMAIN/gateway -> the internal FastAPI gateway.
cat <<EOF > "$APP_DIR/frontend/.env.production.local"
NEXT_PUBLIC_API_URL=https://$DOMAIN/api
NEXT_PUBLIC_GATEWAY_URL=https://$DOMAIN/gateway
NEXT_PUBLIC_ASSISTANT_ID=superbot
LANGGRAPH_API_URL=http://127.0.0.1:$LANGGRAPH_PORT
LANGSMITH_API_KEY=$LANGSMITH_KEY
EOF

echo "==> Installing frontend dependencies and building..."
cd "$APP_DIR/frontend"
npm ci
npm run build
cd "$APP_DIR"

# ---- 5. systemd: LangGraph server (agent graphs + streaming) -------------
echo "==> Configuring $SERVICE_PREFIX-langgraph service..."
cat <<EOF | sudo tee /etc/systemd/system/$SERVICE_PREFIX-langgraph.service
[Unit]
Description=Super AI Bot - LangGraph server
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/backend
Environment="PATH=$APP_DIR/.venv/bin"
EnvironmentFile=$APP_DIR/.env
# --allow-blocking: pdf_chatbot does legitimate blocking PDF parse/embed work
# in a before_agent hook; without this the dev server's blocking detector trips.
ExecStart=$APP_DIR/.venv/bin/langgraph dev --host 127.0.0.1 --port $LANGGRAPH_PORT --no-browser --allow-blocking
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ---- 6. systemd: FastAPI gateway (internal control plane) -----------------
echo "==> Configuring $SERVICE_PREFIX-backend service..."
cat <<EOF | sudo tee /etc/systemd/system/$SERVICE_PREFIX-backend.service
[Unit]
Description=Super AI Bot - FastAPI gateway
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/backend
Environment="PATH=$APP_DIR/.venv/bin"
Environment="PYTHONPATH=$APP_DIR/backend"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port $GATEWAY_PORT
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ---- 6b. systemd: Prompt management service --------------------------------
echo "==> Configuring $SERVICE_PREFIX-prompts service..."
cat <<EOF | sudo tee /etc/systemd/system/$SERVICE_PREFIX-prompts.service
[Unit]
Description=Super AI Bot - Prompt management service
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/prompt_service
Environment="PATH=$APP_DIR/.venv/bin"
# ROOT_PATH makes Swagger UI work behind the stripped /prompt-api nginx prefix.
Environment="ROOT_PATH=/prompt-api"
# Reuses MONGODB_URI/MONGODB_DB from the shared .env. Set PROMPT_API_KEY there
# to require X-API-Key on all /prompts endpoints (recommended: it's public).
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port $PROMPTS_PORT
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ---- 7. systemd: Next.js frontend ----------------------------------------
echo "==> Configuring $SERVICE_PREFIX-frontend service..."
NODE_BIN_DIR="$(dirname "$(command -v node)")"
cat <<EOF | sudo tee /etc/systemd/system/$SERVICE_PREFIX-frontend.service
[Unit]
Description=Super AI Bot - Next.js frontend
After=network.target $SERVICE_PREFIX-langgraph.service

[Service]
User=$USER
WorkingDirectory=$APP_DIR/frontend
Environment="PATH=$NODE_BIN_DIR:/usr/bin:/bin"
Environment="PORT=$FRONTEND_PORT"
Environment="HOSTNAME=127.0.0.1"
ExecStart=$(command -v npm) run start
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ---- 8. Nginx reverse proxy (aibot.miteklabs.tech -> Next.js) -------------
echo "==> Configuring nginx site for $DOMAIN..."
cat <<EOF | sudo tee /etc/nginx/sites-available/$SERVICE_PREFIX
server {
    listen 80;
    server_name $DOMAIN;

    # Allow large PDF uploads to the pdf_chatbot agent
    client_max_body_size 100M;

    # FastAPI gateway (auth, agent list, PDF RAG). Called directly from the
    # browser via NEXT_PUBLIC_GATEWAY_URL=https://$DOMAIN/gateway. The trailing
    # slash on proxy_pass strips the /gateway prefix (/gateway/auth/login ->
    # /auth/login on the gateway).
    location /gateway/ {
        proxy_pass http://127.0.0.1:$GATEWAY_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # PDF ingestion (MarkItDown parse + per-image vision calls + embeddings
        # + waiting for the Atlas index to catch up) can take minutes for large
        # or image-heavy files. Raise the proxy timeouts well above nginx's 60s
        # default so slow uploads aren't cut off with a 504 while the backend is
        # still working.
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Prompt management service. The trailing slash strips the /prompt-api
    # prefix; the service's ROOT_PATH=/prompt-api puts it back in the OpenAPI
    # spec, so Swagger UI at /prompt-api/docs works including "Try it out".
    location /prompt-api/ {
        proxy_pass http://127.0.0.1:$PROMPTS_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location / {
        proxy_pass http://127.0.0.1:$FRONTEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Streaming (SSE) + WebSocket support for live agent tokens
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
EOF

# Enable ONLY our site. We deliberately do NOT remove the default site or any
# other config so AIKhataBook keeps serving on this shared box.
sudo ln -sf /etc/nginx/sites-available/$SERVICE_PREFIX /etc/nginx/sites-enabled/
echo "==> Testing nginx config..."
sudo nginx -t

# ---- 9. Start services ----------------------------------------------------
echo "==> Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_PREFIX-langgraph $SERVICE_PREFIX-backend $SERVICE_PREFIX-prompts $SERVICE_PREFIX-frontend
sudo systemctl restart $SERVICE_PREFIX-langgraph
sudo systemctl restart $SERVICE_PREFIX-backend
sudo systemctl restart $SERVICE_PREFIX-prompts
sudo systemctl restart $SERVICE_PREFIX-frontend
sudo systemctl reload nginx

# ---- 10. SSL via Let's Encrypt -------------------------------------------
echo "==> Setting up SSL for $DOMAIN..."
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d $DOMAIN \
    --non-interactive \
    --agree-tos \
    --register-unsafely-without-email \
    --redirect

echo "==> Verifying SSL auto-renewal..."
sudo certbot renew --dry-run

echo ""
echo "Deployment complete! Your Super AI Bot should now be live at:"
echo "    https://$DOMAIN"
echo "Prompt management Swagger UI (test the API in the browser):"
echo "    https://$DOMAIN/prompt-api/docs"
echo ""
echo "  * To protect the prompt API, add PROMPT_API_KEY=<secret> to .env and"
echo "    run: sudo systemctl restart $SERVICE_PREFIX-prompts"
echo "    Then click Authorize in Swagger and paste the key."
echo "Reminders:"
echo "  * Point DNS: an A record for $DOMAIN -> this VM's public IP."
echo "  * Open ports 80 and 443 in the Azure Network Security Group."
echo "  * Check logs:  sudo journalctl -u $SERVICE_PREFIX-langgraph -f"
echo "                 sudo journalctl -u $SERVICE_PREFIX-frontend  -f"
