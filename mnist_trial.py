from smdebug.rules import Rule, invoke_rule
from smdebug.trials import create_trial
# from smdebug import rule_configs


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


trial = create_trial(path=f'./smd_output/mnist')
print(trial.tensor_names())
print(trial.tensor('Net_conv1.bias').values())
# rule_obj = CustomGradientRule(trial, threshold=0.0001)
# invoke_rule(rule_obj, start_step=0, end_step=None)
