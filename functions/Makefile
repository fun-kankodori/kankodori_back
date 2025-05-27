.PHONY: back
back:
	uvicorn main:app --port 3110 --reload

.PHONY: publish
publish:
	ngrok http 3110 --region jp