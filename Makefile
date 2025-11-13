.PHONY: deploy down

deploy:
	# --- CRITICAL: Set IS_DOCKER=true for Docker execution ---
	sed -i '' 's/^IS_DOCKER=.*/IS_DOCKER=true/g' .env
	
	- uv pip compile --python-version 3.10 pyproject.toml -o requirements.txt
	- cd feature_repo; feast apply; cd ..
	- docker compose up -d --build

down:
	# --- Reset IS_DOCKER=false for local development after shutdown ---
	sed -i '' 's/^IS_DOCKER=.*/IS_DOCKER=false/g' .env
	
	- docker compose down -v && docker system prune -a --volumes --force