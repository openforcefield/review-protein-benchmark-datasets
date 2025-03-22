import numpy
from matplotlib import pyplot


def main():
    figure_width = 3.0
    figure_height = 3.0

    pyplot.rcParams.update({"font.size": 10.0})

    # Create simulated NOESY data to be plotted. Generate 1-D peaks at
    # x = 0.1, 0.2, 0.7, and 0.9 and cross peaks at (0.1, 0.7) and (0.2, 0.9).
    N_points = 10001
    peak_width = 0.005
    gaussian_denominator = 2 * peak_width * peak_width

    # (x, height)
    peaks = [(0.1, 0.8), (0.2, 0.9), (0.7, 0.7), (0.9, 0.8)]

    # (x, y, height, variance)
    cross_peaks = [(0.1, 0.7, 0.5, 0.5), (0.2, 0.9, 0.5, 2.0)]

    x = numpy.linspace(0, 1, 10001)
    one_d_spectrum = numpy.zeros(x.size)
    two_d_spectrum = numpy.zeros((x.size, x.size))

    for peak_x, height in peaks:
        gaussian = height * numpy.exp(-numpy.square(x - peak_x) / gaussian_denominator)
        one_d_spectrum += gaussian
        two_d_spectrum += gaussian * gaussian[:, numpy.newaxis]

    for cross_peak_x, cross_peak_y, cross_peak_height, cross_peak_variance in cross_peaks:
        gaussian_x = numpy.exp(
            -numpy.square(x - cross_peak_x)
            / (cross_peak_variance * gaussian_denominator)
        )
        gaussian_y = numpy.exp(
            -numpy.square(x - cross_peak_y)
            / (cross_peak_variance * gaussian_denominator)
        )
        two_d_spectrum += cross_peak_height * (
            gaussian_x * gaussian_y[:, numpy.newaxis]
            + gaussian_x[:, numpy.newaxis] * gaussian_y
        )

    # Create the main figure and subplots using gridspec
    figure = pyplot.figure(figsize=(figure_width, figure_height))
    gridspec = figure.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    main_axes = figure.add_subplot(gridspec[1, 0])
    top_axes = figure.add_subplot(gridspec[0, 0], sharex=main_axes)
    right_axes = figure.add_subplot(gridspec[1, 1], sharey=main_axes)

    # Remove axis ticks and labels
    for axes in main_axes, top_axes, right_axes:
        axes.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    # Remove frame from top and right plots
    for axes in top_axes, right_axes:
        for line in ["top", "right", "bottom", "left"]:
            axes.spines[line].set_visible(False)

    # Create a contour plot for the 2-D spectrum in the main axes
    levels = [1E-6, 1E-4, 1E-2, 0.5]
    contours = main_axes.contour(x, x, two_d_spectrum, levels, colors="black", linewidths=0.5)

    # Create a solid diagonal line in the main axes
    main_axes.plot(x, x, color="black", linewidth=1)

    # Create dashed lines at peak locations in the main axes
    for peak_x, _ in peaks:
        main_axes.axvline(x=peak_x, color="black", linewidth=0.5, linestyle="--")
        main_axes.axhline(y=peak_x, color="black", linewidth=0.5, linestyle="--")

    # Create line plots for the 1-D spectrum in the top and right axes
    top_axes.plot(x, one_d_spectrum, color="black", linewidth=1)
    right_axes.plot(one_d_spectrum, x, color="black", linewidth=1)

    # Label peaks in the top and right axes
    peak_labels = "ABCD"
    for (peak_x, height), peak_label in zip(peaks, "ABCD"):
        top_axes.text(peak_x + 0.0125, 0.625, peak_label)
        right_axes.text(0.625, peak_x + 0.0125, peak_label)


    pyplot.savefig("noesy-2d.png")
    pyplot.close(figure)


if __name__ == "__main__":
    main()
