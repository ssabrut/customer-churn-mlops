.PHONY: deploy down

deploy:
	- sed -i '' 's/^IS_DOCKER=.*/IS_DOCKER=true/g' .env
	- uv pip compile --python-version 3.10 pyproject.toml -o requirements.txt
	- docker compose --env-file .env up -d --build

down:
	- sed -i '' 's/^IS_DOCKER=.*/IS_DOCKER=false/g' .env
	- docker compose down -v && docker system prune -a --volumes --force