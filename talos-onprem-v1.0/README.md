# Talos Backend - On-Premise Deployment Guide

## Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 20GB disk space

## Installation

1. Copy .env.example to .env and configure:
```bash
   cp .env.example .env
   nano .env  # Edit with your values
```

2. Run install script:
```bash
   chmod +x scripts/install.sh
   ./scripts/install.sh
```

3. Access the API at http://localhost:8000

## Backup
```bash
./scripts/backup.sh
```

## Upgrade
```bash
./scripts/upgrade.sh
```