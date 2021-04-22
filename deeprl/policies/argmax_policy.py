import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        """
        TODO: return the action that maximizes the Q-value 
        at the current observation as the output.
        """
        scores = self.critic.qa_values(observation)
        action = np.argmax(scores, axis=-1)
        """
        END CODE
        """
        return action.squeeze()
