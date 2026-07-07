#!/bin/bash

# Exit on any error
set -e

echo "Starting deployment setup for PDF Chatbot..."

# 1. Update and install dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx

# 2. Setup project directory
APP_DIR=$(pwd)
echo "Setting up application in $APP_DIR"

# 3. Setup Python Virtual Environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Setup systemd service for FastAPI (Backend)
echo "Configuring FastAPI service..."
cat <<EOF | sudo tee /etc/systemd/system/chatbot-backend.service
[Unit]
Description=PDF Chatbot FastAPI Backend
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
Environment="PYTHONPATH=$APP_DIR"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 5. Setup systemd service for Streamlit (Frontend)
echo "Configuring Streamlit service..."
cat <<EOF | sudo tee /etc/systemd/system/chatbot-frontend.service
[Unit]
Description=PDF Chatbot Streamlit Frontend
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/frontend
Environment="PATH=$APP_DIR/venv/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 6. Configure Nginx Reverse Proxy for Streamlit
echo "Configuring Nginx..."
cat <<EOF | sudo tee /etc/nginx/sites-available/chatbot
server {
    listen 80;
    server_name chatbot.miteklabs.tech;
    
    # Allow large PDF uploads
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Streamlit WebSockets support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
# Remove default nginx site if it conflicts (optional, but good practice if only hosting one thing, though main site might be here. The user said main site is miteklabs.tech, which might be on this server or another. Assuming this droplet is just for chatbot or handles both via virtual hosts)
# sudo rm -f /etc/nginx/sites-enabled/default

# 7. Start and Enable Services
echo "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable chatbot-backend
sudo systemctl enable chatbot-frontend
sudo systemctl restart chatbot-backend
sudo systemctl restart chatbot-frontend
sudo systemctl restart nginx

# 8. Setup SSL with Let's Encrypt (Certbot)
echo "Setting up SSL with Let's Encrypt..."
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate and auto-configure Nginx for HTTPS
# --non-interactive: don't prompt for input
# --agree-tos: agree to Let's Encrypt TOS
# --redirect: automatically redirect HTTP to HTTPS
sudo certbot --nginx -d chatbot.miteklabs.tech \
    --non-interactive \
    --agree-tos \
    --register-unsafely-without-email \
    --redirect

echo "Verifying SSL auto-renewal..."
sudo certbot renew --dry-run

echo "Deployment complete! Your chatbot should now be live at https://chatbot.miteklabs.tech"
echo "Note: Make sure your DNS records for chatbot.miteklabs.tech point to this server's IP."
