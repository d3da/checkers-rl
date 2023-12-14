Setup:

(Optional) setup a python virtual environment and activate it
$ python3 -m venv venv/
$ source venv/bin/activate

Install requirements
$ bash ./install_reqs.sh

Run the train loop
$ python3 ./train.py

Run the train loop with pdb debugger
$ python3 -m pdb ./train.py
(then type "c" to start the program)
