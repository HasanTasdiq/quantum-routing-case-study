"""Case study of concurrent entanglement routing: reimplemented methods against optimization references."""
from .model import REGIMES, make_instance, make_slot, realized, hop_limit, network_seed, slot_seed
from .programs import (routing_program, hindsight_optimum, lp_bound, post_generation_optimum, hindsight_relaxation,
                       value_of, check_paths)
from .methods import METHODS, OTHER_POLICIES
