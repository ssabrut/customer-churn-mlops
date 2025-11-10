deploy:
	- uv pip compile pyproject.toml -o requirements.txt
	- docker build -t churn-mlops-image:latest .
	- docker compose up -d --build

down:
	- docker compose down -v && docker system prune -a --volumes --force