PYTHON := .venv/bin/python

.PHONY: install data model report all dashboard test

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	PYTHONPATH=. $(PYTHON) -m src.data_pipeline

model:
	PYTHONPATH=. $(PYTHON) -m src.train_model

report:
	PYTHONPATH=. $(PYTHON) -m src.build_report

all:
	PYTHONPATH=. $(PYTHON) -m src.run_all

dashboard:
	PYTHONPATH=. .venv/bin/streamlit run dashboard/app.py

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q

