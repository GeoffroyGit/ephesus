# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* ephesus/*.py

black:
	@black scripts/* ephesus/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr ephesus-*.dist-info
	@rm -fr ephesus.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload

##### Docker - - - - - - - - - - - - - - - - - - - - - - - - -

# project id
PROJECT_ID=crypto-galaxy-351308
# docker image name
DOCKER_IMAGE_NAME=ephesus-api

docker_build:
	docker build --tag eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run:
	docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_push:
	docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_deploy:
	gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1
