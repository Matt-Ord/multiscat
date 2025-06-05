import numpy as np
from slate_core import array, plot
from slate_core import metadata as _metadata
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.interpolate import interpolate_potential

if __name__ == "__main__":
    # An example of how to build a simple scattering potential.
    # Intially, we create a potential in a fourier basis, as this
    # is the most common representation we would get from e.g. a DFT calculation.
    vectors = (np.array([np.pi, 0, 0]), np.array([0, np.pi, 0]), np.array([0, 0, 1]))
    metadata = _metadata.spaced_volume_metadata_from_stacked_delta_x(
        vectors,
        (3, 3, 5),
    )
    initial_potential = operator.build.potential_from_function(
        metadata,
        lambda x: np.sin(x[0]) * np.sin(x[1]) * np.exp(-(5 * x[2] ** 2)),
    )
    initial_potential = initial_potential.as_type(np.complex128)

    fig, ax, _anim0 = plot.animate_data_2d(array.extract_diagonal(initial_potential))
    ax.set_title(
        (
            "Initial Potential generated from a function,\n"
            f"in a fourier basis with ({metadata.shape}) points"
        ),
    )
    fig.show()

    lobatto_metadata = scattering_metadata_from_stacked_delta_x(
        vectors,
        (12, 12, 35),
    )
    # TODO: Make this more general, interpolating between  # noqa: FIX002
    # LabelledMetadata and move it into slate_quantum.operator.build,
    # since this is generally useful.
    interpolated = interpolate_potential(lobatto_metadata, initial_potential)
    fig, ax, _anim1 = plot.animate_data_2d(array.extract_diagonal(interpolated))
    ax.set_title(
        (
            "Interpolated scattering potential,\n"
            f"in a lobatto basis with ({lobatto_metadata.shape}) points"
        ),
    )
    fig.show()
    plot.wait_for_close()
