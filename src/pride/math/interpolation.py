from scipy import interpolate


class Default1DInterpolator(interpolate.Akima1DInterpolator):

    def __init__(self, x, y, axis: int = 0) -> None:

        super().__init__(x, y, axis=axis, extrapolate=False)

        return None
