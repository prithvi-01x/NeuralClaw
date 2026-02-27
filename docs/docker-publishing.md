# Publishing NeuralClaw Docker Image

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- A Docker Hub account (or GitHub Container Registry / other registry)

---

## 1. Docker Hub

### Login

```bash
docker login
# Enter your Docker Hub username and password/access token
```

### Tag the Image

```bash
# Format: docker tag <local-image> <dockerhub-username>/<repo-name>:<tag>
docker tag neuralclaw:latest yourusername/neuralclaw:latest
docker tag neuralclaw:latest yourusername/neuralclaw:1.0.0
```

### Push

```bash
docker push yourusername/neuralclaw:latest
docker push yourusername/neuralclaw:1.0.0
```

### Pull (from another machine)

```bash
docker pull yourusername/neuralclaw:latest
docker run -it --env-file .env yourusername/neuralclaw:latest
```

---

## 2. GitHub Container Registry (ghcr.io)

### Login

```bash
# Create a PAT at https://github.com/settings/tokens with write:packages scope
echo $GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### Tag & Push

```bash
docker tag neuralclaw:latest ghcr.io/yourusername/neuralclaw:latest
docker tag neuralclaw:latest ghcr.io/yourusername/neuralclaw:1.0.0

docker push ghcr.io/yourusername/neuralclaw:latest
docker push ghcr.io/yourusername/neuralclaw:1.0.0
```

### Make Public (optional)

Go to `https://github.com/users/yourusername/packages/container/neuralclaw/settings` → change visibility to **Public**.

---

## 3. Build & Push in One Step

```bash
# Docker Hub
docker buildx build -t yourusername/neuralclaw:latest --push .

# GitHub Container Registry
docker buildx build -t ghcr.io/yourusername/neuralclaw:latest --push .
```

---

## 4. Multi-Architecture Build (amd64 + arm64)

```bash
# Create a builder (one-time)
docker buildx create --name multiarch --use

# Build and push for both architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t yourusername/neuralclaw:latest \
  -t yourusername/neuralclaw:1.0.0 \
  --push .
```

---

## 5. CI/CD Automation (GitHub Actions)

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Image

on:
  push:
    tags: ["v*"]

jobs:
  docker:
    runs-on: ubuntu-latest
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

Then push a tag to trigger it:

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## Quick Reference

| Registry | Image Format |
|---|---|
| Docker Hub | `yourusername/neuralclaw:tag` |
| GitHub (ghcr) | `ghcr.io/yourusername/neuralclaw:tag` |

> **Tip:** Replace `yourusername` with your actual registry username throughout this guide.
