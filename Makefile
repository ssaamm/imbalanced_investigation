datasets: prepare_jobs.py prepare_keel.py
	venv/bin/python prepare_jobs.py
	venv/bin/python prepare_keel.py

clean:
	rm -rf datasets targets.json
