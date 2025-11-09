deploy:
	- docker compose up -d --build

down:
	- docker compose down -v && docker system prune -a --volumes --force