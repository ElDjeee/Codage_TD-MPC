# TD-MPC
This project is part of an IT project unit of the ANDROIDE master's program. Its purpose is to train us in carrying out a team project from start to finish, from analyzing the subject to software development, as well as experimenting with the work done. Our project topic consists in implementing and evaluating a reinforcement learning algorithm, TD-MPC, within a specific library, BBRL.

TD-MPC is a combination of Temporal Difference (TD) learning and Model Predictive Control (MPC). This approach utilizes Temporal Difference predictions to enhance planning and action execution in Model Predictive Control, enabling better anticipation and adaptation to environmental changes for decision optimization.
BBRL, which stands for BlackBoard Reinforcement Learning, is a simple and flexible library for reinforcement learning, derived from SaLinA. 

## Librairy

BBRL: Inspired by SaLinA, source code: https://github.com/osigaud/bbrl
Gymnasium: OpenAI library, source code: https://github.com/Farama-Foundation/Gymnasium


## Source code TD-MPC

Our project is inspired by the research conducted by Hansen et al. 
See the source code below.: https://github.com/nicklashansen/tdmpc


## Installation

The project is available on [GitHub](https://github.com/ElDjeee/Codage_TD-MPC).

Before continuing, you need to install [Python3](https://www.python.org/downloads/).

Then, follow these steps in a terminal to install the project:

```bash
git clone https://github.com/ElDjeee/Codage_TD-MPC.git
cd Codage_TD-MPC
python3 -m pip install -r requirements.txt
```

Once this is done, you can run the project with this command:
```bash
python3 src/train.py task=quadruped-run modality=pixels
```

You can replace the task with any tasks inside the tasks.txt file
