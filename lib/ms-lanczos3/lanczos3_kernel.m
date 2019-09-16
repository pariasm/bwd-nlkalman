function f = lanczos3_kernel(x)
% See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
% pp. 157-158.

f = (sin(pi*x) .* sin(pi*x/3) + eps) ./ ((pi^2 * x.^2 / 3) + eps);
f = f .* (abs(x) < 3);

