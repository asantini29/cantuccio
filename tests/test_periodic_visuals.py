import numpy as np
import matplotlib.pyplot as plt
from cantuccio import cornerplot

np.random.seed(42)
# Create a wrapped cluster! Half at ~0, half at ~2pi
# Passed entirely unwrapped to cornerplot: True angles!
data1 = np.concatenate([
    np.random.normal(0.1, 0.2, 500),
    np.random.normal(2*np.pi - 0.1, 0.2, 500)
])

# Another dimension that's just normal
data2 = np.random.normal(0, 1, 1000)

samples = {
    "phase": data1,
    "amp": data2
}

periodic = {"phase": (0.0, 2*np.pi)}

fig, ax = cornerplot(
    samples, 
    periodic=periodic,
    kde_kwargs={"fast": True},
    offdiag_mode="contour+kde"
)
fig.savefig("tests/periodic_plot_contour.png")
print("Cornerplot test saved to tests/periodic_plot_contour.png")

fig, ax = cornerplot(
    samples, 
    periodic=periodic,
    kde_kwargs={"fast": True},
    offdiag_mode="hexbin+kde"
)
fig.savefig("tests/periodic_plot_hexbin.png")
print("Cornerplot test saved to tests/periodic_plot_hexbin.png")


# Now let's impose periodicity on a cloud that is centered in the middle of the range, to check that it doesn't mess with non-periodic dimensions.
data3 = np.random.normal(0, 1, 1000)
samples["other"] = data3
periodic["other"] = (-2*np.pi, 2*np.pi)


fig, ax = cornerplot(
    samples, 
    periodic=periodic,
    kde_kwargs={"fast": True},
    offdiag_mode="hexbin+kde"
)
fig.savefig("tests/periodic_plot_extra.png")
print("Cornerplot test saved to tests/periodic_plot_extra.png")