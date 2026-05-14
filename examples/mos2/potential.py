from model import build_mos2_potential
from slate_core import EvenlySpacedLengthMetadata, array, plot
from slate_core import metadata as _metadata

if __name__ == "__main__":
    model_potential = build_mos2_potential(
        (64, 64),
        EvenlySpacedLengthMetadata(50, domain=_metadata.Domain(delta=10e-10)),
    )
    fig, ax, _ = plot.array_against_axes_2d(
        array.extract_diagonal(model_potential),
        idx=(1,),
    )
    fig.show()

    fig, ax, _ = plot.array_against_axes_2d_k(
        array.extract_diagonal(model_potential),
        idx=(1,),
    )
    fig.show()

    plot.wait_for_close()
