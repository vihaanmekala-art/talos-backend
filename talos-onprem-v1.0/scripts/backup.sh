#!/bin/bash
set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "💾 Backing up database..."
docker-compose exec -T postgres pg_dump -U talos_user talos_prod > "$BACKUP_DIR/talos_backup_$TIMESTAMP.sql"

echo "✅ Backup saved to $BACKUP_DIR/talos_backup_$TIMESTAMP.sql"