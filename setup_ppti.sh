# export PYTHONPATH="/Vrac/TDMPC:$PYTHONPATH"


# TODO, check if TDMPC dir exists and if it's mine.
# If exists and mine, go in
# If exists, create another dir named $(whoami)_TDMPC
# Else, create it

python3 -m pip install PyOpenGL_accelerate --no-cache-dir --target=/Vrac/TDMPC
python3 -m pip install bbrl_gymnasium --no-cache-dir --target=/Vrac/TDMPC

scp 21304195@ssh.ufr-info-p6.jussieu.fr:/users/Etu5/21304195/Documents/S2/PAND/New/Archive/outputs/2024-05-15/23-00-51 /Users/eldjeee/Documents/Master/S2/PROJET-ANDROIDE