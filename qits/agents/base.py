class Controller:
    """Interface: called once per simulation step, before the traffic dynamics.
    A controller may write `sim.signals.pending_split` (applied at the next cycle start)
    and `sim.route_p` (routing probabilities over N,E,S,W per node)."""

    name = "base"

    def reset(self, sim):
        pass

    def step(self, sim, t):
        pass


class FixedTimeController(Controller):
    """Pure fixed-time plan: constant split, uniform minimal routing."""

    name = "fixed_time"
