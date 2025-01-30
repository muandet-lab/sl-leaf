"""
1-D case with Taylor's expansions.
"""
import matplotlib.pyplot as plt
import numpy as np


def g(x):
    return x ** 4 - x ** 2 + 1


def f(order: int, xb: np.ndarray, x: np.ndarray):
    assert 1 <= order <= 3
    assert (xb.ndim == 1) and (x.ndim == 1)

    terms = [
        g(xb),  # 0-th
        (4 * xb ** 3 - 2 * xb) * (x - xb),  # 1st
        (12 * xb ** 2 - 2) * np.power(x - xb, 2) / 2,  # 2nd
        (24 * xb) * np.power(x - xb, 3) / 6,  # 3rd
        # 24 * np.power(x - xb, 4) / 24  # 4th
    ]

    y_tilde = np.sum(terms[:order + 1], axis=0)

    return y_tilde


def ug(xb, x, cf):
    return -g(x) - cf * np.square(x - xb)


def uf(order: int, xb, x, cf):
    return -f(order=order, xb=xb, x=x) - cf * np.square(x - xb)


def best_respond_against_ar(xb: np.ndarray, cf: np.ndarray, xr: np.ndarray) -> np.ndarray:
    ub = ug(xb=xb, x=xb, cf=cf)
    ur = ug(xb=xb, x=xr, cf=cf)
    w: np.ndarray[bool] = (ur >= ub)  # compliance statuses
    x = (1 - w) * xb + w * xr
    return x


def best_respond_against_expansions(order: int, xb: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """
    This implicitly uses the knowledge of `g`.
    """
    assert 1 <= order <= 2

    x = None
    if order == 1:
        x = np.power(-2 * cf, -1) * (4 * xb ** 3 - 2 * (1 + cf) * xb)
    elif order == 2:
        numerator = 4 * xb ** 3 + cf * xb
        denominator = 6 * xb ** 2 + cf - 1
        x = np.divide(numerator, denominator)

    return x


def plot_graph(order, xb, x):
    # Generate x values (e.g., from -10 to 10)
    x0 = np.linspace(-2, 2, 100)  # 100 points between -10 and 10
    y0 = g(x0)  # Apply the function to each x value

    # Plot the function
    plt.plot(x0, y0, label='g(x) = $x^4-x^2+1$')
    plt.scatter(xb, g(xb), color='black', label='$x^{(b)}$')
    plt.scatter(x, g(x), color='red', label='$x$')

    plt.ylim(0, 4)
    plt.xlim(-2, 2)

    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.title(f"Quartic function with {order}-th order expansions as surrogate functions.")
    plt.legend()
    plt.grid()

    plt.savefig(f'no-harm-quartic-{order}-th.png')
    return


def plot_bars(r1, r2, uf1, uf2, ug1, ug2, order):
    # # scale the data
    # r_scale = 1 / abs(np.concatenate((r1, r2))).max()
    # uf_scale = 1 / abs(np.concatenate((uf1, uf2))).max()
    # ug_scale = 1 / abs(np.concatenate((ug1, ug2))).max()
    #
    # r1 /= r_scale
    # r2 /= r_scale
    # uf1 /= uf_scale
    # uf2 /= uf_scale
    # ug1 /= ug_scale
    # ug2 /= ug_scale

    # the data
    scenarios = [r"$g(x)$, $\downarrow$ is better",
                 r"$u(f,x)$, $\uparrow$ is better",
                 r"$u(g,x)$, $\uparrow$ is better"]
    v1 = [r1.mean(), uf1.mean(), ug1.mean()]
    v2 = [r2.mean(), uf2.mean(), ug2.mean()]

    std1 = [np.std(l) for l in [r1, uf1, ug1]]
    std2 = [np.std(l) for l in [r2, uf2, ug2]]

    # bar width & positions
    bar_width = 0.35
    xcoo = np.arange(len(scenarios))  # Positions for the bars

    # do plot
    plt.figure(figsize=(10, 6))
    plt.bar(xcoo - bar_width / 2, v1, width=bar_width, label="Before best respond")
    plt.bar(xcoo + bar_width / 2, v2, width=bar_width, label="After best respond")

    # Customize the plot
    plt.xticks(xcoo, scenarios)
    # plt.xlabel("Scenarios")
    plt.ylabel("Mean Values")
    plt.title("Comparison of agents' objectives.")
    plt.legend()
    plt.grid()

    plt.savefig(f"no-harm-quartic-{order}-th-bars.png")
    return


def plot_hist(ug1, ug2, order, title=''):
    plt.figure(figsize=(4, 2.7))
    plt.hist(ug2 - ug1, bins=5, edgecolor='black')
    plt.title(title)
    plt.xlabel(r"$u_t(g,x_t)-u_t(g,x_t^{(b)})$")
    plt.ylabel("Number of agents")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"no-harm-quartic-{order}-th-hist.png")
    return


def show_stats_and_plots(xb, cf, x, order, title=''):
    r1 = g(xb)
    r2 = g(x)
    uf1 = uf(order=order, xb=xb, x=xb, cf=cf)
    uf2 = uf(order=order, xb=xb, x=x, cf=cf)
    ug1 = ug(xb=xb, x=xb, cf=cf)
    ug2 = ug(xb=xb, x=x, cf=cf)
    print(
        f"""
            Agents who lower (or keep) true predictive risk g(x): {(r1 >= r2).mean():.0%},
            Agents who obtain higher surrogate utilities uf: {(uf1 < uf2).mean():.0%},
            Agents who obtain higher (or the same) true utilities ug: {(ug1 <= ug2).mean():.0%} 
            """
    )

    # plot results
    plot_graph(order=order, xb=xb, x=x)
    plot_bars(r1, r2, uf1, uf2, ug1, ug2, order=order)
    plot_hist(ug1=ug1, ug2=ug2, order=order, title=title)
    return


def run():
    n = 100
    # d = 1
    order = 2

    # set random seed
    np.random.seed(222)

    # generate data
    xb = np.random.normal(loc=0, scale=0.4, size=n)
    cf = np.random.uniform(low=1, high=1.2, size=n)

    # with Taylor expansions
    x = best_respond_against_expansions(order=order, xb=xb, cf=cf)
    show_stats_and_plots(xb=xb, cf=cf, x=x, order=order, title="with 2nd-order Taylor expansions")

    # with action recommendations
    xr = np.random.normal(loc=0, scale=0.4, size=n)
    x = best_respond_against_ar(xb=xb, cf=cf, xr=xr)
    ug1 = ug(xb=xb, x=xb, cf=cf)
    ug2 = ug(xb=xb, x=x, cf=cf)

    plot_hist(ug1=ug1, ug2=ug2, order='AR', title="with action recommendations")

    return


if __name__ == '__main__':
    run()
