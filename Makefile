.PHONY: install chat ui audit fetch lint test docs serve-docs clean

install:
	cd agent && pip install -r requirements.txt

chat:
	cd agent && python cli.py

ui:
	cd agent && streamlit run app.py

fetch:
	cd agent && python fetch_sources.py

audit:
	cd agent && python -c "from wiki_tool import audit_wiki; print(audit_wiki())"

lint:
	ruff check agent/
	python -m compileall -q agent/

test:
	pytest tests/ -v

docs:
	python scripts/wikilinks_to_md.py --dry-run

serve-docs:
	python scripts/wikilinks_to_md.py --no-backup
	mkdocs serve

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name "*.md.bak" -delete
