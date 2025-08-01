.PHONY: run test lint

run:
uvicorn api.trip_planner:app --reload

test:
pytest -q

lint:
flake8
