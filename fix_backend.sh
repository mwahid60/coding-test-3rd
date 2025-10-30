#!/bin/bash

echo "🔧 Fixing Backend Issues..."
echo ""

# Step 1: Stop backend
echo "1️⃣ Stopping backend container..."
podman stop fund-backend 2>/dev/null || true
podman rm fund-backend 2>/dev/null || true

# Step 2: Rebuild backend
echo ""
echo "2️⃣ Rebuilding backend..."
cd backend
podman build -t fund-backend .
cd ..

# Step 3: Restart services
echo ""
echo "3️⃣ Starting services..."
podman-compose up -d

# Step 4: Wait for backend to start
echo ""
echo "4️⃣ Waiting for backend to start (15 seconds)..."
sleep 15

# Step 5: Check backend status
echo ""
echo "5️⃣ Checking backend status..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Checking logs..."
    echo ""
    podman logs fund-backend --tail 30
    exit 1
fi

# Step 6: Check services
echo ""
echo "6️⃣ Service Status:"
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Step 7: Create default fund
echo ""
echo "7️⃣ Creating default fund..."
curl -s -X POST "http://localhost:8000/api/funds/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Demo Fund",
    "gp_name": "Demo GP",
    "fund_type": "Private Equity",
    "vintage_year": 2023
  }' | grep -q "id" && echo "✅ Default fund created!" || echo "⚠️  Fund may already exist"

echo ""
echo "✨ Done! Backend should be ready now."
echo ""
echo "📋 Next steps:"
echo "   1. Open http://localhost:3000/upload"
echo "   2. Upload a PDF file"
echo "   3. Check processing status"
echo ""
