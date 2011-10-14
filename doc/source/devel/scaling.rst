##################################
About scaling images and precision
##################################

We have an original input array of floating point values $f_i : i=0..N-1$

We will convert these to an array of integers $z_i : i=0..N-1$ with:

.. math::

    z_i = (f_i + I) * S

where $I$ is an intercept value and $S$ is a scaling value.

Because we are using floating point values, which are inexact, the actual scale
value we use $S^r$ can be thought of as the true scale value $S$ plus some error
$E$.  Similarly the actual intercept $I^r$ can be thought of as $I$ plus some
error $F$:

.. math::

    z_i = (f_i + I + F) * (S + E)

and the errors introduced by $F$ and $E$:

.. math::

    e_i = (f_i + I + F) * (S + E) - (f_i + I) * S

so:

.. math::

    e_i = E*F + E*I + E*f_i + F*S

The squared error is::

    sq_e_i = sy.simplify(sy.expand(e_i ** 2))

.. in sympy

    sq_e_i_2 = E**2*F**2 + 2*E**2*F*I + \
    E**2*I**2 + \
    2*E*F**2*S + 2*E*F*I*S + \
    F**2*S**2 + \
    2*E**2*F*f_i + \
    2*E**2*I*f_i + \
    2*E*F*S*f_i + \
    E**2*f_i**2

.. math::

    e_i^2 = E^{2} F^{2} + 2 E^{2} F I + 2 E^{2} F f_{i} + E^{2} I^{2} + 2 E^{2}
    I f_{i} + E^{2} f_{i}^{2} + 2 E F^{2} S + 2 E F I S + 2 E F S f_{i} + F^{2}
    S^{2}

If we have an expression for the mean of $f_i : i=0..N-1$ - call it $\mu_f$ and
the mean of $f_i^2 : i=0..N-1$ call this $\mu_f^2$, then the mean error is:

.. math::

    E^{2} F^{2} + 2 E^{2} F I + 2 E^{2} F \mu_f + E^{2} I^{2} + 2 E^{2}
    I \mu_f + E^{2} \mu_f^{2} + 2 E F^{2} S + 2 E F I S + 2 E F S \mu_f + F^{2}
    S^{2}

