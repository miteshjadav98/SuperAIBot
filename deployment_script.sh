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
# Architecture (3 processes, only the frontend is public):
#   1. LangGraph server  (127.0.0.1:2024) - agent graphs + token streaming
#   2. FastAPI gateway   (127.0.0.1:8010) - control plane / PDF RAG (internal)
#   3. Next.js frontend  (127.0.0.1:3100) - chat UI, proxied by nginx
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

# ---- 3. Python virtual environment ---------------------------------------
echo "==> Creating Python virtual environment (.venv)..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/backend/requirements.txt"

# ---- 4. Frontend build ----------------------------------------------------
# NEXT_PUBLIC_* vars are baked in at build time, so the production env file
# must exist BEFORE `npm run build`. The browser calls the same-origin /api
# route; that route proxies to LangGraph server-side via LANGGRAPH_API_URL.
echo "==> Writing frontend production env..."
LANGSMITH_KEY="$(grep -E '^LANGSMITH_API_KEY=' "$APP_DIR/.env" | head -1 | cut -d= -f2- | tr -d "\"'" || true)"
cat <<EOF > "$APP_DIR/frontend/.env.production.local"
NEXT_PUBLIC_API_URL=https://$DOMAIN/api
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
sudo systemctl enable $SERVICE_PREFIX-langgraph $SERVICE_PREFIX-backend $SERVICE_PREFIX-frontend
sudo systemctl restart $SERVICE_PREFIX-langgraph
sudo systemctl restart $SERVICE_PREFIX-backend
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
echo ""
echo "Reminders:"
echo "  * Point DNS: an A record for $DOMAIN -> this VM's public IP."
echo "  * Open ports 80 and 443 in the Azure Network Security Group."
echo "  * Check logs:  sudo journalctl -u $SERVICE_PREFIX-langgraph -f"
echo "                 sudo journalctl -u $SERVICE_PREFIX-frontend  -f"
