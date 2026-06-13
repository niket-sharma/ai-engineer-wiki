.PHONY: install chat ui audit fetch lint test docs serve-docs clean \
        interview assess maintain validate

interview:
	cd agent && python cli.py interview --topic $(TOPIC)

assess:
	cd agent && python cli.py assess

maintain:
	cd agent && python cli.py maintain --no-pr

validate:
	python3 scripts/validate_wiki.py

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
