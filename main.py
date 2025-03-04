#
# Test driver.
#
import predictor
from simulator import simulate

my_predictor = predictor.Predictor("dummy", None)

simulate(2017, 1, my_predictor)
