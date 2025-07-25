# Docker Permissions Setup

This document explains how Arc-Fusion automatically handles Docker permissions to prevent write errors in `logs/` and `data/` volumes.

## 🎯 Problem Solved

Before these improvements, permission errors were common:
```
Permission denied: 'data/golden_dataset.jsonl'
```

## ✅ Automatic Solution

### 1. Automatic Setup Script

**`./docker-up.sh`** - Main script that handles everything automatically:
```bash
./docker-up.sh        # Start services
./docker-up.sh build  # Force rebuild + start  
./docker-up.sh down   # Stop services
```

### 2. Dynamic Permissions

The system automatically detects your UID/GID and passes them to the Docker container:

**Host (your system):** `adrien (UID=1000, GID=1000)`  
**Container:** `appuser (UID=1000, GID=1000)` ← Same permissions!

### 3. Volumes with Proper Permissions

```yaml
volumes:
  - ./logs:/app/logs:rw    # System logs
  - ./data:/app/data:rw    # Evaluation datasets
```

Both directories are created with `chmod 755` and correct ownership.

## 🔧 How It Works

### Dynamic Dockerfile
```dockerfile
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser && \
    mkdir -p /app/logs /app/data && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app/logs /app/data
```

### Docker Compose with Variables
```yaml
build: 
  context: .
  args:
    USER_ID: ${USER_ID:-1000}
    GROUP_ID: ${GROUP_ID:-1000}
```

### Automatic Setup Script
The `./scripts/setup_permissions.sh` script:
1. Detects your current UID/GID
2. Creates `logs/` and `data/` directories
3. Sets correct permissions
4. Generates `.env.docker` with your variables

## 🚀 Usage

### New Installation
```bash
git clone <repo>
cd arc-fusion
./docker-up.sh build
```

### Daily Development
```bash
./docker-up.sh        # Start
./docker-up.sh down   # Stop
```

### Debugging Permissions
```bash
# Check permissions
ls -la logs/ data/

# Check user inside container
docker compose exec app id

# Manually recreate permissions
./scripts/setup_permissions.sh
```

## 📁 File Structure

```
arc-fusion/
├── logs/                    # System logs (Docker volume)
├── data/                    # Evaluation datasets (Docker volume)
│   └── golden_dataset.jsonl
├── docker-up.sh            # Main script
├── scripts/
│   └── setup_permissions.sh # Permission setup
├── .env.docker             # Auto-generated variables
└── Dockerfile              # Build with dynamic UID/GID
```

## ✨ Benefits

- **Zero manual configuration** - Everything is automatic
- **No permission errors** - UID/GID are synchronized
- **Cross-platform** - Works on Linux/macOS/WSL
- **Simplified development** - One script for everything
- **Persistent volumes** - Your data survives rebuilds

## 🔄 Migration from Old Method

If you were still using `docker compose` manually:

```bash
# Old way
docker compose down
docker compose up --build

# New way  
./docker-up.sh build
```

The new script does exactly the same + automatic permission setup! 