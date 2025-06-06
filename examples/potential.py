import numpy as np
from slate_core import array, plot
from slate_core import metadata as _metadata
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.interpolate import interpolate_potential

if __name__ == "__main__":
    # An example of building a simple scattering potential.
    # Intially, we create a potential in a fourier basis, as this
    # is the most common representation we would get from e.g. a DFT calculation.
    metadata = _metadata.spaced_volume_metadata_from_stacked_delta_x(
        (np.array([np.pi, 0, 0]), np.array([0, np.pi, 0]), np.array([0, 0, 10])),
        (3, 3, 35),
    )
    initial_potential = operator.build.potential_from_function(
        metadata,
        lambda x: np.sin(x[0]) * np.sin(x[1]) * np.exp(-(x[2] ** 2 / 25)),
    ).as_type(np.complex128)

    fig, ax, _anim0 = plot.animate_data_2d(array.extract_diagonal(initial_potential))
    ax.set_title(
        (
            "Initial Potential generated from a function,\n"
            f"in a fourier basis with ({metadata.shape}) points"
        ),
    )
    fig.show()

    # We then interpolate this potential into a lobatto basis.
    lobatto_metadata = scattering_metadata_from_stacked_delta_x(
        _metadata.volume.fundamental_stacked_delta_x(metadata),
        (12, 12, 35),
    )
    interpolated = interpolate_potential(lobatto_metadata, initial_potential)
    fig, ax, _anim1 = plot.animate_data_2d(array.extract_diagonal(interpolated))
    ax.set_title(
        (
            "Interpolated scattering potential,\n"
            f"in a lobatto basis with ({lobatto_metadata.shape}) points"
        ),
    )
    fig.show()

    # Alternatively, we could generate a potential directly in the lobatto basis.
    morse_potential = operator.build.morse_potential(
        lobatto_metadata,
        operator.build.CorrugatedMorseParameters(depth=10, height=0.8, offset=0.8),
    ).as_type(np.complex128)
    fig, ax, _anim2 = plot.array_against_axes_1d(
        array.extract_diagonal(morse_potential),
        axes=(2,),
    )
    ax.set_title(
        (
            "Morse potential,\n"
            f"in a lobatto basis with ({lobatto_metadata.shape}) points"
        ),
    )
    fig.show()

    morse_potential = operator.build.corrugated_morse_potential(
        lobatto_metadata,
        operator.build.CorrugatedMorseParameters(
            depth=10,
            height=0.8,
            offset=0.8,
            beta=4,
        ),
    ).as_type(np.complex128)
    fig, ax, _anim2 = plot.array_against_axes_2d(
        array.extract_diagonal(morse_potential),
        axes=(0, 2),
    )
    ax.set_title(
        (
            "A corrugated Morse potential,\n"
            f"in a lobatto basis with ({lobatto_metadata.shape}) points"
        ),
    )
    fig.show()
    plot.wait_for_close()
