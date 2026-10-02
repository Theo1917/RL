test:
	python -m pytest tests -q

experiments:
	python experiments/run_small.py --seed 42

demo: experiments
	python demo.py --seed 42

web:
	python -m webapp.server --host 127.0.0.1 --port 8000