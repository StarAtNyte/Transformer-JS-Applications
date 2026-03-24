#!/bin/bash
set -e

echo "==> Creating /etc/nginx/includes/tfjs.conf"
sudo tee /etc/nginx/includes/tfjs.conf > /dev/null << 'EOF'
location = /tfjs {
    return 301 /tfjs/;
}

location /tfjs/ {
    alias /home/at-office/Projects/Nitiz/TFJS/;
    try_files $uri $uri/ $uri/index.html =404;

    location ~* \.(png|jpg|jpeg|gif|webp|svg|ico)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    location ~* \.(css|js|wasm|bin|onnx)$ {
        expires 7d;
        add_header Cache-Control "public";
    }
}
EOF

echo "==> Adding tfjs include to compute.explorug.online (both server blocks)"
sudo sed -i 's|include /etc/nginx/includes/maps.conf;|include /etc/nginx/includes/maps.conf;\n    include /etc/nginx/includes/tfjs.conf;|g' \
    /etc/nginx/sites-enabled/compute.explorug.online

echo "==> Disabling old tfjs site"
sudo rm -f /etc/nginx/sites-enabled/tfjs

echo "==> Removing tfjs dnsmasq entries"
sudo sed -i '/tfjs\.at\.online/d' /etc/dnsmasq.conf
sudo sed -i '/tfjs\.local\.online/d' /etc/dnsmasq.conf

echo "==> Testing nginx config"
sudo nginx -t

echo "==> Reloading nginx"
sudo systemctl reload nginx

echo "==> Restarting dnsmasq"
sudo systemctl restart dnsmasq

echo ""
echo "Done! Access your apps at: https://compute.explorug.online/tfjs/"
