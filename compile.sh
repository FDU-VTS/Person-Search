python -O -m compileall .
find . -name '*.pyc' -exec rename 's/.cpython-36.opt-1//' {} \;
find . -name '*.pyc' -execdir mv {} .. \;
find . -name '*.py' -type f -print -exec rm {} \;
find . -name '__pycache__' -exec rmdir {} \;