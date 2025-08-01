# Nenemi API

This repository provides a FastAPI service for planning travel packages. The code
is now modular with separate packages for API routes, services, models and
utilities.

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn api.trip_planner:app --reload
```

## Testing

```bash
make test
```
