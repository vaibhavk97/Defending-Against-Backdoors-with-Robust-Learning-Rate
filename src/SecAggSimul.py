class SecAggSimul():
    def __init__(self):
        self.agg_updates = 0
        self.agg_ndata = 0

    def get_average_values(self):
        return self.agg_updates / self.agg_ndata

    def submit_grad_ndata_prod(self, inp):
        self.agg_updates += inp

    def submit_ndata_points(self, inp):
        self.agg_ndata += inp

    def reset_values(self):
        self.agg_updates = 0
        self.agg_ndata = 0
        self.agg_sign = 0
