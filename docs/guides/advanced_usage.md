# Advanced Usage Guide

## Complex Steganographic Scenarios

### Multi-Layer Obfuscation
```bash
python scripts/embed.py --techniques noise,rotation,fragmentation --noise-level 0.02 --fragment-models ada-002,3-small
```

### Time-Delayed Exfiltration
```bash
python scripts/embed.py --timing-mode business_hours --delay-range 300-900
```

## Advanced Evasion Techniques

### Full Evasion Mode
```bash
python scripts/embed.py --evasion-mode maximum --legitimate-ratio 0.9 --cover-story "quarterly analysis"
```

### Custom Behavioral Profiles
```bash
python scripts/embed.py --user-profile researcher --activity-pattern academic
```

## Multi-Format Processing

### Batch Document Processing
```bash
python scripts/embed.py --directory /path/to/documents --recursive --formats pdf,docx,xlsx,csv
```

### Database Exfiltration
```bash
python scripts/embed.py --database /path/to/database.sqlite --tables sensitive_data,user_info
```

## Advanced Query Techniques

### Multi-Strategy Search
```bash
python scripts/query.py --strategy hybrid --context-reconstruction --cross-reference
```

### Forensic Data Recovery
```bash
python scripts/query.py --mode recovery --export-format json --include-metadata
```

## Production Deployment

### Kubernetes with Monitoring
```bash
kubectl apply -f k8s/ -n vectorsmuggle
./scripts/deploy/health-check.sh --detailed
```

### Security Hardening
```bash
./scripts/deploy/deploy.sh --environment production --security-scan --monitoring
```

See [deployment guide](deployment.md) for complete production setup.