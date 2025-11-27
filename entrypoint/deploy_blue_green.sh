#!/bin/bash
set -e

# Configuration
NGINX_CONTAINER="nginx_lb"
CONFIG_FILE="./docker/nginx/nginx.conf"

# 1. Detect Current Color
# We grep the nginx.conf to see which server is currently active
if grep -q "server fastapi_blue:8000;" "$CONFIG_FILE"; then
    CURRENT_COLOR="blue"
    NEW_COLOR="green"
    NEW_HOST="fastapi_green"
    NEW_PORT="8002"
else
    CURRENT_COLOR="green"
    NEW_COLOR="blue"
    NEW_HOST="fastapi_blue"
    NEW_PORT="8001"
fi

echo "🔵 Current active color: $CURRENT_COLOR"
echo "🟢 Deploying to: $NEW_COLOR"

# 2. Build and Start the New Color
# We force a recreate to ensure it picks up the latest code/model
echo "🚀 Starting $NEW_HOST container..."
docker compose up -d --build --no-deps $NEW_HOST

# 3. Health Check (Wait for the new container to be ready)
echo "❤️ Checking health of $NEW_HOST..."
MAX_RETRIES=12 # 1 minute total (12 * 5s)
COUNT=0
SUCCESS=false

while [ $COUNT -lt $MAX_RETRIES ]; do
    # curl the internal port (mapped to localhost)
    if curl -s "http://localhost:$NEW_PORT/health" | grep -q "ok"; then
        echo "✅ $NEW_COLOR is healthy!"
        SUCCESS=true
        break
    fi
    echo "⏳ Waiting for $NEW_COLOR to come up... ($COUNT/$MAX_RETRIES)"
    sleep 5
    COUNT=$((COUNT+1))
done

if [ "$SUCCESS" = false ]; then
    echo "❌ Deployment failed: $NEW_COLOR did not become healthy."
    echo "Rollback: Leaving $CURRENT_COLOR active."
    exit 1
fi

# 4. Flip the Switch (Update Nginx Config)
echo "🔄 Switching traffic to $NEW_COLOR..."

# Use sed to replace the active server line in the config file
# This replaces 'server fastapi_XXX:8000;' with 'server fastapi_YYY:8000;'
if [ "$NEW_COLOR" == "green" ]; then
    sed -i '' 's/server fastapi_blue:8000;/server fastapi_green:8000;/g' $CONFIG_FILE
else
    sed -i '' 's/server fastapi_green:8000;/server fastapi_blue:8000;/g' $CONFIG_FILE
fi

# 5. Reload Nginx
# This is a seamless reload; existing connections complete, new ones go to new server
docker exec $NGINX_CONTAINER nginx -s reload

echo "🎉 Deployment Complete! Traffic is now routed to $NEW_COLOR."
echo "ℹ️  Old environment ($CURRENT_COLOR) is still running. Run 'docker stop fastapi_$CURRENT_COLOR' to save resources if needed."