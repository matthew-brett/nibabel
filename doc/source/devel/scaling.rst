##################################
About scaling images and precision
##################################

We have an original input array of floating point values $f_i : i=0..N-1$

We will convert these to an array of integers $z_i : i=0..N-1$ with:

.. math::

    z_i = (f_i + I) * S

where $I$ is an intercept value and $S$ is a scaling value.

Because we are using floating point values, which are inexact, the scale value
$S$ can be thought of as the true scale value $S^t$ plus some error $E$.
Similarly the intercept can be thought of as $I^t$ plus some error $F$:

.. math::

    z_i = (f_i + I^t + F) * (S^t + E)

and the errors introduced by $F$ and $E$:

.. math::

    e_i = (f_i + I^t + F) * (S^t + E) - (f_i + I^t) * S^t)

    e_i = E*F + E*It + E*f_i + F*St

Clearly $F$ (the intercept error) contributes.
