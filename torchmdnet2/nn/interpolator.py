import torch.nn as nn


class NaturalCubicSpline(nn.Module):
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.
    """

    def __init__(self, coeffs):
        """
        Arguments:
            coeffs: As returned by `torchcubicspline.natural_cubic_spline_coeffs`.
        """
        super(NaturalCubicSpline, self).__init__()
        t, a, b, c, d = coeffs

        self.register_buffer('_t', t)
        self.register_buffer('_a', a)
        self.register_buffer('_b', b)
        self.register_buffer('_c', c)
        self.register_buffer('_d', d)


    def __len__(self):
        """Method to return basis size"""
        return self._b.shape[-1]

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        # TODO: switch to a binary search not a linear search.
        index = (t.unsqueeze(-1) > self._t).sum(dim=-1) - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    # def _interpret_t_2(self, t):
    #     maxlen = self._b.size(-2) - 1
    #     rng = self._t[-1] - self._t[0]
    #     return fractional_part, index

    def forward(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def backward(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 2 * self._c[..., index, :] + 3 * self._d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv