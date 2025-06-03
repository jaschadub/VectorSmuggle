# Deployment Guide

## Overview

This guide covers production deployment of VectorSmuggle using Docker and Kubernetes.

## Docker Deployment

### Production Build
```bash
docker build -t vectorsmuggle:latest .
```

### Environment Configuration
```bash
# Create production environment file
cp .env.example .env.production

# Configure for production
export OPENAI_API_KEY=sk-...
export VECTOR_DB=qdrant
export QDRANT_URL=https://your-qdrant-instance.com
```

### Run Production Container
```bash
docker run -d \
  --name vectorsmuggle \
  --env-file .env.production \
  -p 8080:8080 \
  vectorsmuggle:latest
```

## Docker Compose Deployment

### Development Environment
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Environment
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Persistent storage available

### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace vectorsmuggle

# Apply all manifests
kubectl apply -f k8s/ -n vectorsmuggle

# Check deployment status
kubectl get pods -n vectorsmuggle
kubectl rollout status deployment/vectorsmuggle -n vectorsmuggle
```

### Configuration Management
```bash
# Create secrets
kubectl create secret generic vectorsmuggle-secrets \
  --from-literal=openai-api-key=sk-... \
  -n vectorsmuggle

# Apply configuration
kubectl apply -f k8s/configmap.yaml -n vectorsmuggle
```

## Automated Deployment

### Using Deployment Scripts
```bash
# Full automated deployment
./scripts/deploy/deploy.sh --environment production --platform kubernetes --build

# Health check and validation
./scripts/deploy/health-check.sh --detailed --export health-report.json
```

## Security Considerations

### Container Security
- Non-root user execution
- Read-only filesystem
- Minimal base image
- Security scanning

### Network Security
- TLS encryption
- Network policies
- Rate limiting
- Ingress security headers

### Secrets Management
- Kubernetes secrets
- External secret management
- Rotation policies
- Access controls

## Monitoring and Observability

### Health Checks
```bash
# Application health
curl http://localhost:8080/health

# Detailed status
./scripts/deploy/health-check.sh --detailed
```

### Logging
- Structured JSON logging
- Centralized log aggregation
- Log retention policies
- Security event monitoring

### Metrics
- Prometheus integration
- Grafana dashboards
- Resource monitoring
- Performance tracking

## Scaling

### Horizontal Scaling
```bash
kubectl scale deployment vectorsmuggle --replicas=3 -n vectorsmuggle
```

### Resource Management
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

## Backup and Recovery

### Data Backup
```bash
# Backup vector store data
kubectl exec -it vectorsmuggle-pod -- /backup-script.sh

# Backup configuration
kubectl get configmap vectorsmuggle-config -o yaml > config-backup.yaml
```

### Disaster Recovery
- Multi-region deployment
- Data replication
- Automated failover
- Recovery procedures

## Troubleshooting

### Common Issues
- Pod startup failures
- Service connectivity
- Resource constraints
- Configuration errors

### Debug Commands
```bash
# Check pod logs
kubectl logs -f deployment/vectorsmuggle -n vectorsmuggle

# Describe resources
kubectl describe pod vectorsmuggle-xxx -n vectorsmuggle

# Execute into container
kubectl exec -it vectorsmuggle-pod -- /bin/bash
```

See [troubleshooting guide](../technical/troubleshooting.md) for detailed debugging procedures.