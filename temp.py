import matplotlib.pyplot as plt
import numpy as np

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = current / rampup_length
        #return float(np.exp(-5.0 * phase * phase))
        return (0.72+0.28*float(np.exp(-5.0 * phase * phase)))*np.log(2)

rampdown_length = 100
x_values = np.linspace(0, rampdown_length, 100)
y_values = [cosine_rampdown(x, rampdown_length) for x in x_values]

plt.plot(x_values, y_values, label="cosine_rampdown")
plt.xlabel("Current")
plt.ylabel("Value")
plt.title("Cosine Rampdown Function")
plt.legend()
plt.grid(True)
plt.show()
#36-48

x_values = np.linspace(0, rampdown_length, 100)
y_values = [sigmoid_rampup(x, rampdown_length) for x in x_values]

plt.plot(x_values, y_values, label="sigmoid_rampdown")
plt.xlabel("Current")
plt.ylabel("Value")
plt.title("sigmoid Rampdown Function")
plt.legend()
plt.grid(True)
plt.show()
