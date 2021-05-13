from collections import defaultdict

class Checkpoint():

    def __init__(self):
        #self.args = args.__dict__
        self.G_losses = defaultdict(list)
        self.D_losses = defaultdict(list)
        self.losses_ts = []
        self.metrics_val = defaultdict(list)
        self.metrics_train = defaultdict(list)
        self.sample_ts = []
        self.restore_ts = []
        self.norm_g = []
        self.norm_d = []
        self.counters = {
            "t": None,
            "epoch": None,
        }
        self.g_state = None
        self.g_optim_state = None
        self.d_state = None
        self.d_optim_state = None
        self.g_best_state = None
        self.d_best_state = None
        self.best_t = None
        self.g_best_nl_state = None
        self.d_best_state_nl = None
        self.best_t_nl = None

    def load_checkpoint(self, config):
        self.args = config.__dict__
        self.G_losses = config.G_losses
        self.D_losses = config.D_losses
        self.losses_ts = config.losses_ts
        self.metrics_val = config.metrics_val
        self.metrics_train = config.metrics_train
        self.sample_ts = config.sample_ts
        self.restore_ts = config.restore_ts
        self.norm_g = config.norm_g
        self.norm_d = config.norm_d
        self.counters = config.counters
        self.g_state = config.g_state
        self.g_optim_state = config.g_optim_state
        self.d_state = config.d_state
        self.d_optim_state = config.d_optim_state
        self.g_best_state = config.g_best_state
        self.d_best_state = config.d_best_state
        self.best_t = config.best_t
        self.g_best_nl_state = config.g_best_nl_state
        self.d_best_state_nl = config.d_best_state_nl
        self.best_t_nl = config.best_t_nl


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm