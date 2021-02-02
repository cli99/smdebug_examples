from smdebug.rules import Rule, invoke_rule
from smdebug.trials import create_trial
# from smdebug import rule_configs
import numpy as np
import matplotlib.pyplot as plt


class CustomGradientRule(Rule):
    def __init__(self, base_trial, threshold=10.0):
        super().__init__(base_trial)
        self.threshold = float(threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step, "mean", abs=True)
            if abs_mean > self.threshold:
                return True
        return False


smdebug_dir = './output/mnist'

trial = create_trial(path=smdebug_dir)
# print((trial.tensor_names()))

rule_obj = CustomGradientRule(trial, threshold=0.0001)
invoke_rule(rule_obj, start_step=0, end_step=None)

# values = trial.tensor('CrossEntropyLoss_output_0').values()
# values_eval = np.array(list(values.items()))
# fig = plt.figure()
# plt.plot(values_eval[:, 1])
# fig.suptitle('Validation Accuracy', fontsize=20)
# plt.xlabel('Intervals of sampling', fontsize=18)
# plt.ylabel('Acuracy', fontsize=16)
# fig.savefig('temp.jpg')