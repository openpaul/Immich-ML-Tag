# Immich ML Tag

Trains classifiers on your Immich tags using CLIP embeddings from smart-search. Automatically predicts tags for new photos.

**Status:** Early development. Expect breaking changes.

> **[!WARNING]**
> This tool connects directly to Immich's PostgreSQL database to read CLIP embeddings.
> While the connection is opened in **read-only mode**, direct database access is inherently risky.
> I make no guarantees, use at your own risk. Please report if you have noticed issues.

We need this direct database access to get the embeddings. If the API starts exposing the embeddings this can be ported.


## How it works

1. You manually tag some photos in Immich (minimum 10 per tag)
2. This tool trains a logistic regression model per tag using CLIP embeddings
3. New photos get auto-tagged with `TagName_predicted` suffix
4. Review predictions and move correct ones to the real tag

## Requirements

- Immich with smart-search enabled (for CLIP embeddings)
- PostgreSQL port exposed (the tool reads embeddings directly from the database)
- Immich API key

## Installation

### Docker Compose (recommended)

You can simply add the Github based docker to your docker compose.
It will then run alongside your immich install.

Add this to your Immich `docker-compose.yml`:

```yaml
  immich-ml-tag:
    container_name: immich_ml_tag
    image: ghcr.io/openpaul/immich-ml-tag:latest
    volumes:
      - ml-tag-data:/data/ml_resources
    environment:
      - DB_HOST=database
      - DB_PORT=5432
      - DB_USERNAME=${DB_USERNAME}
      - DB_PASSWORD=${DB_PASSWORD}
      - DB_DATABASE_NAME=${DB_DATABASE_NAME}
      - API_KEY=${IMMICH_ML_TAG_API_KEY}
      - URL=http://immich-server:2283
      - ML_RESOURCE_PATH=/data/ml_resources
      # Optional settings
      # - TRAIN_TIME=02:00
      # - INFERENCE_INTERVAL=5
      # - THRESHOLD=0.5
      # - MIN_SAMPLES=10
    depends_on:
      - database
      - immich-server
    restart: unless-stopped
```

Add the volume:

```yaml
volumes:
  ml-tag-data:
```

Add to your `.env`:

```
IMMICH_ML_TAG_API_KEY=your_api_key_here
```

Get your API key from the immich UI. Its needs these permissions as of now (checked: December 2025):

- asset
  - asset.read
- tag
  - tag.create
  - tag.delete
  - tag.read   
  - tag.asset


### Manual installation

```bash
git clone git@github.com:openpaul/Immich-ML-Tag.git
pip install ./immich-ml-tag
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | PostgreSQL port |
| `DB_USERNAME` | `postgres` | Database user |
| `DB_PASSWORD` | - | Database password |
| `DB_DATABASE_NAME` | `immich` | Database name |
| `API_KEY` | - | Immich API key |
| `URL` | `https://immich.example.com` | Immich server URL |
| `ML_RESOURCE_PATH` | `./ml_resources` | Path to store models |
| `TRAIN_TIME` | `02:00` | Daily training time (HH:MM) |
| `INFERENCE_INTERVAL` | `5` | Minutes between inference runs |
| `THRESHOLD` | `0.5` | Prediction probability threshold |
| `MIN_SAMPLES` | `10` | Minimum tagged photos required to train |

## Usage

### Serve mode (default in Docker)

Runs as a daemon with scheduled training and inference:

```bash
immich-ml-tag serve --train-on-start
```

### Manual commands

```bash
# Train models for all tags
immich-ml-tag train

# Force retrain even if data hasn't changed
immich-ml-tag train --force

# Run inference on new photos
immich-ml-tag inference

# Run inference on all photos
immich-ml-tag inference --full
```

## Negative examples

Create a tag called `z_ml_negative_examples` to group negative example tags. For each tag you want to train, you can create a child tag under `z_ml_negative_examples` with the same name to provide explicit negative examples.

Example:
- `Dogs` (your regular tag with dog photos)
- `z_ml_negative_examples/Dogs` (photos that look like dogs but aren't)

## Known Issues

- If a photo is tagged with both the manual tag and the `_predicted` tag, removing only the `_predicted` tag won't trigger retraining. Remove the `_predicted` tag to fix.
- The Immich UI can be slow to update while bulk tagging is in progress. Wait for tagging to complete.

## Get involved

This project needs eyes and hands. Do review the code, suggest a refactoring, fix a bug. I would be happy if someone else thought this was usefull.

It also needs documentation and screenshots so it's a bit more clear what the project does.

## AI Disclaimer

Claude Opus 4.5 was used to generate parts of this codebase.